"""
Atmos_Physics.py
================
Meteorological equations used by the upper-air CAM model, written in PyTorch so they are
differentiable (the network can learn through them) and run on the GPU.

Three groups:

1. **Thermodynamic diagnostics** (fixed definitions): saturation vapour pressure, mixing ratio,
   specific humidity, dewpoint, relative humidity, potential temperature, equivalent potential
   temperature (Bolton 1980).
2. **Kinematic diagnostics on the 3 km map-projected grid**: divergence, relative/absolute vorticity,
   gradient, Laplacian, Jacobian J(u,v), d/dp, kinematic omega (continuity equation).
3. **Adaptive prognostic equations** (``AdaptiveEquation``): every equation is written as a sum of
   named physical terms, and every term has a learnable coefficient that starts at its textbook value
   of 1.0. Training may move those coefficients (per pressure level) if that makes the forecasts verify
   better; ``coefficient_table()`` shows what the model changed. ``PhysicsCore`` uses them to step the
   atmosphere forward in time (vorticity-divergence form of the primitive equations, as in spectral
   models like the GFS).

The divergence equation (AMS Glossary, "divergence equation"; isobaric, frictionless):

    d/dt(D) = -D^2 + 2 J(u,v) - grad(omega) . dV/dp - beta*u + f*zeta - laplacian(Phi)

with d/dt = d/dt + V.grad + omega d/dp. On a map-projected grid f varies in x as well as y, so the
"beta" term is evaluated in its general form  v df/dx - u df/dy  (= -beta*u when f = f(y) only).

Units: SI throughout (m, s, Pa, K, m/s, s^-1). Pressure levels are given in hPa and converted here.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

G = 9.80665           # gravity (m s^-2)
RD = 287.04           # dry-air gas constant (J kg^-1 K^-1)
CP = 1004.64          # specific heat at constant pressure (J kg^-1 K^-1)
KAPPA = RD / CP       # = 2/7
EPS = 0.622           # Rd / Rv


# ====================================================================== 1. thermodynamics
def saturation_vapor_pressure(t_kelvin):
    """Bolton (1980) saturation vapour pressure over water, hPa."""
    tc = t_kelvin - 273.15
    return 6.112 * torch.exp(17.67 * tc / (tc + 243.5))


def vapor_pressure(t_kelvin, rh_percent):
    return saturation_vapor_pressure(t_kelvin) * torch.clamp(rh_percent, 1e-3, 100.0) / 100.0


def mixing_ratio(e_hpa, p_hpa):
    """kg/kg"""
    return EPS * e_hpa / torch.clamp(p_hpa - e_hpa, min=1e-3)


def specific_humidity_from_rh(t_kelvin, rh_percent, p_hpa):
    e = vapor_pressure(t_kelvin, rh_percent)
    return EPS * e / (p_hpa - (1.0 - EPS) * e)


def rh_from_specific_humidity(q, t_kelvin, p_hpa):
    e = q * p_hpa / (EPS + (1.0 - EPS) * q)
    return 100.0 * e / saturation_vapor_pressure(t_kelvin)


def dewpoint(t_kelvin, rh_percent):
    """Kelvin. Inverse of the Bolton formula."""
    ln_ratio = torch.log(vapor_pressure(t_kelvin, rh_percent) / 6.112)
    return 243.5 * ln_ratio / (17.67 - ln_ratio) + 273.15


def potential_temperature(t_kelvin, p_hpa):
    return t_kelvin * (1000.0 / p_hpa) ** KAPPA


def equivalent_potential_temperature(t_kelvin, rh_percent, p_hpa):
    """Bolton (1980) eq. 43 (with the LCL temperature from eq. 15)."""
    td = dewpoint(t_kelvin, rh_percent)
    r = 1000.0 * mixing_ratio(vapor_pressure(t_kelvin, rh_percent), p_hpa)          # g/kg
    t_lcl = 1.0 / (1.0 / (td - 56.0) + torch.log(t_kelvin / td) / 800.0) + 56.0
    theta_dl = t_kelvin * (1000.0 / p_hpa) ** (0.2854 * (1.0 - 0.28e-3 * r))
    return theta_dl * torch.exp((3.376 / t_lcl - 0.00254) * r * (1.0 + 0.81e-3 * r))


# ====================================================================== 2. kinematics on the grid
def ddx(f, dx):
    """d/dX along the last axis (grid metres): centred inside, one-sided on the edges."""
    inner = (f[..., 2:] - f[..., :-2]) / (2.0 * dx)
    left = (f[..., 1:2] - f[..., 0:1]) / dx
    right = (f[..., -1:] - f[..., -2:-1]) / dx
    return torch.cat([left, inner, right], dim=-1)


def ddy(f, dx):
    inner = (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dx)
    bottom = (f[..., 1:2, :] - f[..., 0:1, :]) / dx
    top = (f[..., -1:, :] - f[..., -2:-1, :]) / dx
    return torch.cat([bottom, inner, top], dim=-2)


def gradient(f, m, dx):
    """True-distance horizontal gradient (df/dx, df/dy) on a conformal grid."""
    return m * ddx(f, dx), m * ddy(f, dx)


def divergence(u, v, m, dx):
    """D = m^2 [d(u/m)/dX + d(v/m)/dY]   (exact form for a conformal map projection)."""
    return m * m * (ddx(u / m, dx) + ddy(v / m, dx))


def vorticity(u, v, m, dx):
    """Relative (vertical) vorticity  zeta = m^2 [d(v/m)/dX - d(u/m)/dY]."""
    return m * m * (ddx(v / m, dx) - ddy(u / m, dx))


def laplacian(f, m, dx):
    """m^2 (f_XX + f_YY), 5-point stencil on the interior; the outer ring copies the neighbouring
    interior value (a one-sided stencil there would turn the edge value into grad(f)/dx, a huge artefact)."""
    inner = (f[..., 1:-1, 2:] + f[..., 1:-1, :-2] + f[..., 2:, 1:-1] + f[..., :-2, 1:-1]
             - 4.0 * f[..., 1:-1, 1:-1]) / (dx * dx)
    shape = inner.shape
    lap = F.pad(inner.reshape(-1, 1, *shape[-2:]), (1, 1, 1, 1), mode="replicate").reshape(*shape[:-2], shape[-2] + 2,
                                                                                            shape[-1] + 2)
    return m * m * lap


def jacobian(a, b, m, dx):
    """J(a, b) = da/dx db/dy - da/dy db/dx."""
    return m * m * (ddx(a, dx) * ddy(b, dx) - ddy(a, dx) * ddx(b, dx))


def ddp(f, p_pa, level_dim: int = -3):
    """d/dp along the level axis (non-uniform spacing). ``p_pa``: 1-D tensor of level pressures (Pa)."""
    f = f.movedim(level_dim, -1)
    p = p_pa.to(f)
    if f.shape[-1] == 1:
        return torch.zeros_like(f).movedim(-1, level_dim)
    inner = (f[..., 2:] - f[..., :-2]) / (p[2:] - p[:-2])
    first = (f[..., 1:2] - f[..., 0:1]) / (p[1] - p[0])
    last = (f[..., -1:] - f[..., -2:-1]) / (p[-1] - p[-2])
    return torch.cat([first, inner, last], dim=-1).movedim(-1, level_dim)


def kinematic_omega(div, p_pa, level_dim: int = -3, omega_top=None):
    """Vertical motion omega (Pa/s) from the continuity equation d(omega)/dp = -D.

    Levels must be ordered from the bottom (high pressure) to the top (low pressure). Integration
    starts at the top level where omega_top (default 0: little vertical motion near 100 hPa) is
    imposed, and works downward with the trapezoidal rule. Upper-level divergence -> omega < 0 (lift).
    """
    d = div.movedim(level_dim, 0)
    p = p_pa.to(div)
    n = d.shape[0]
    omegas = [None] * n
    omegas[-1] = torch.zeros_like(d[-1]) if omega_top is None else omega_top
    for k in range(n - 2, -1, -1):
        omegas[k] = omegas[k + 1] - 0.5 * (d[k] + d[k + 1]) * (p[k] - p[k + 1])
    return torch.stack(omegas, 0).movedim(0, level_dim)


def wind_speed(u, v):
    return torch.sqrt(u * u + v * v + 1e-12)


def gaussian_smooth(f, sigma_points: float):
    """Separable Gaussian smoother over the last two axes (used for display of noisy 3 km fields)."""
    if sigma_points <= 0:
        return f
    radius = max(1, int(math.ceil(2.5 * sigma_points)))
    x = torch.arange(-radius, radius + 1, dtype=f.dtype, device=f.device)
    k = torch.exp(-0.5 * (x / sigma_points) ** 2)
    k = k / k.sum()
    shape = f.shape
    g = f.reshape(-1, 1, *shape[-2:])
    g = F.conv2d(F.pad(g, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    g = F.conv2d(F.pad(g, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return g.reshape(shape)


def soft_limit(x, limit: float):
    """limit * tanh(x / limit): ~x for |x| << limit, never exceeds +-limit (smooth, differentiable)."""
    return limit * torch.tanh(x / limit)


# ====================================================================== Poisson solver (zero Dirichlet)
def _dst1(x):
    """DST-I over the last axis via an FFT of the odd extension."""
    n = x.shape[-1]
    zeros = torch.zeros_like(x[..., :1])
    ext = torch.cat([zeros, x, zeros, -x.flip(-1)], dim=-1)
    return -torch.fft.fft(ext, dim=-1).imag[..., 1:n + 1] / 2.0


def solve_poisson_dirichlet(rhs, dx):
    """Solve (d2/dX2 + d2/dY2) psi = rhs on the interior, psi = 0 on the outer ring of points.

    ``rhs``: [..., ny, nx]. Returns psi with the same shape (boundary ring = 0). Spectral (DST-I)
    solution of the 5-point Laplacian - exact for the discrete operator, differentiable, O(N log N).
    """
    ny, nx = rhs.shape[-2:]
    r = rhs[..., 1:-1, 1:-1]
    hi, wi = r.shape[-2:]
    kx = torch.arange(1, wi + 1, dtype=rhs.dtype, device=rhs.device)
    ky = torch.arange(1, hi + 1, dtype=rhs.dtype, device=rhs.device)
    lam = ((2 * torch.cos(math.pi * kx / (wi + 1)) - 2)[None, :]
           + (2 * torch.cos(math.pi * ky / (hi + 1)) - 2)[:, None]) / (dx * dx)
    spec = _dst1(_dst1(r).transpose(-1, -2)).transpose(-1, -2)
    spec = spec / lam
    psi = _dst1(_dst1(spec).transpose(-1, -2)).transpose(-1, -2) * (2.0 / (wi + 1)) * (2.0 / (hi + 1))
    return F.pad(psi, (1, 1, 1, 1))


# ====================================================================== semi-Lagrangian advection
def semi_lagrangian(fields, u, v, omega, m, dx, p_pa, dt, h_scale=None, v_scale=None):
    """Advect ``fields`` [B, C, L, H, W] by the 3-D wind for ``dt`` seconds.

    Departure points are found with one midpoint iteration and values are interpolated (trilinear,
    ``grid_sample``). Semi-Lagrangian advection is stable for any time step, which is what lets a
    physics step of 15-60 minutes run on a 3 km grid (an explicit Eulerian step would need ~20 s).
    ``h_scale`` / ``v_scale`` ([L] tensors) are the learnable advection coefficients. Points whose
    departure lies outside the domain take the boundary value (no lateral boundary conditions yet).
    """
    b, c, nl, h, w = fields.shape
    dev, dt_ = fields.device, fields.dtype
    p = p_pa.to(fields)
    # level spacing so that omega*dt (Pa) -> fraction of a level index (levels ordered bottom -> top)
    dp = torch.empty_like(p)
    if nl > 1:
        dp[1:-1] = (p[:-2] - p[2:]) / 2.0
        dp[0], dp[-1] = p[0] - p[1], p[-2] - p[-1]
    else:
        dp[:] = 1.0
    hs = torch.ones(nl, device=dev, dtype=dt_) if h_scale is None else h_scale
    vs = torch.ones(nl, device=dev, dtype=dt_) if v_scale is None else v_scale
    # displacement per time step in index units; u,v are grid-relative m/s, grid metres = true metres * m
    di = u * m * dt / dx * hs.view(1, nl, 1, 1)
    dj = v * m * dt / dx * hs.view(1, nl, 1, 1)
    dk = -omega * dt / dp.view(1, nl, 1, 1) * vs.view(1, nl, 1, 1) if nl > 1 else torch.zeros_like(di)

    kk, jj, ii = torch.meshgrid(torch.arange(nl, device=dev, dtype=dt_), torch.arange(h, device=dev, dtype=dt_),
                                torch.arange(w, device=dev, dtype=dt_), indexing="ij")

    def to_grid(i, j, k):
        gx = 2.0 * i / max(w - 1, 1) - 1.0
        gy = 2.0 * j / max(h - 1, 1) - 1.0
        gz = 2.0 * k / max(nl - 1, 1) - 1.0 if nl > 1 else torch.zeros_like(i)
        return torch.stack([gx, gy, gz], dim=-1)

    # (di, dj, dk) = parcel displacement over dt in index units. The level index grows upward (pressure
    # falls), so a sinking parcel (omega > 0) has dk < 0 and its departure point lies above: k_d = k - dk.
    disp = torch.stack([di, dj, dk], dim=1)                        # [B, 3, L, H, W]
    mid = to_grid(ii - 0.5 * di, jj - 0.5 * dj, kk - 0.5 * dk)
    disp_mid = F.grid_sample(disp, mid, mode="bilinear", padding_mode="border", align_corners=True)
    dep = to_grid(ii - disp_mid[:, 0], jj - disp_mid[:, 1], kk - disp_mid[:, 2])
    return F.grid_sample(fields, dep, mode="bilinear", padding_mode="border", align_corners=True)


# ====================================================================== 3. adaptive equations
class AdaptiveEquation(nn.Module):
    """sum_k c[k, level] * term_k, with learnable coefficients c initialised to the physical value 1.

    Option 1 of the design brief: "the model can have a variable it is allowed to change based on the
    weights of each node". Each (term, level) coefficient is a trainable weight; gradient descent moves
    it only if that makes the forecast verify better against the ground truth. ``prior_penalty`` keeps
    coefficients near 1 unless the data clearly disagree, so the textbook equation is the default.
    """

    def __init__(self, name: str, terms: list[str], n_levels: int, per_level: bool = True,
                 trainable: bool = True, descriptions: dict | None = None):
        super().__init__()
        self.name, self.terms = name, list(terms)
        self.descriptions = descriptions or {}
        shape = (len(terms), n_levels if per_level else 1)
        self.coeff = nn.Parameter(torch.ones(shape), requires_grad=trainable)

    def c(self, term: str):
        """Coefficient of ``term`` broadcastable against [B, L, H, W]."""
        row = self.coeff[self.terms.index(term)]
        return row.view(1, -1, 1, 1)

    def forward(self, terms: dict):
        missing = set(self.terms) - set(terms)
        if missing:
            raise KeyError(f"{self.name}: missing terms {sorted(missing)}")
        return sum(self.c(name) * terms[name] for name in self.terms)

    def prior_penalty(self):
        return ((self.coeff - 1.0) ** 2).sum()

    def coefficient_table(self, levels=None) -> dict:
        values = self.coeff.detach().cpu().numpy()
        out = {}
        for k, term in enumerate(self.terms):
            row = values[k]
            out[term] = ({f"{int(p)} hPa": round(float(x), 4) for p, x in zip(levels, row)}
                         if levels is not None and len(row) == len(levels) else [round(float(x), 4) for x in row])
        return out


def equation_terms_divergence(u, v, zeta, div, omega, phi, f, fx, fy, m, dx, p_pa):
    """Right-hand side terms of the divergence equation (AMS Glossary form), each [B, L, H, W]."""
    du_dp, dv_dp = ddp(u, p_pa), ddp(v, p_pa)
    wx, wy = gradient(omega, m, dx)
    return {
        "divergence_squared": -div * div,                          # -(D)^2
        "jacobian": 2.0 * jacobian(u, v, m, dx),                   # +2 J(u, v)
        "tilting": -(wx * du_dp + wy * dv_dp),                     # -grad(omega) . dV/dp
        "beta": v * fx - u * fy,                                   # -beta*u (general form)
        "coriolis_vorticity": f * zeta,                            # +f zeta
        "geopotential_laplacian": -laplacian(phi, m, dx),         # -laplacian(Phi)
    }


def equation_terms_vorticity(u, v, zeta, div, omega, f, fx, fy, m, dx, p_pa):
    du_dp, dv_dp = ddp(u, p_pa), ddp(v, p_pa)
    wx, wy = gradient(omega, m, dx)
    return {
        "stretching": -(zeta + f) * div,                           # -(zeta+f) D
        "tilting": wy * du_dp - wx * dv_dp,                        # k . (dV/dp x grad omega)
        "beta": -(u * fx + v * fy),                                # -V . grad f
    }


DIVERGENCE_TERMS = ["divergence_squared", "jacobian", "tilting", "beta", "coriolis_vorticity",
                    "geopotential_laplacian"]
VORTICITY_TERMS = ["stretching", "tilting", "beta"]


def boundary_taper(h: int, w: int, width: int, device=None, dtype=None):
    """0 on the outer ring -> 1 at ``width`` points from the edge (sin^2 ramp)."""
    jj = torch.arange(h, device=device, dtype=dtype)
    ii = torch.arange(w, device=device, dtype=dtype)
    dj = torch.minimum(jj, (h - 1) - jj)[:, None]
    di = torch.minimum(ii, (w - 1) - ii)[None, :]
    d = torch.clamp(torch.minimum(dj, di) / max(width, 1), 0.0, 1.0)
    return torch.sin(0.5 * math.pi * d) ** 2


def _geo(geo: dict, name: str):
    """Static field as [B or 1, 1, H, W] so it broadcasts against [B, L, H, W]."""
    a = geo[name]
    return a.view(1, 1, *a.shape[-2:]) if a.dim() == 2 else a.unsqueeze(1)


class PhysicsCore(nn.Module):
    """Advances (Z, U, V, T, RH) on pressure levels with the adaptive primitive equations.

    One call = one step of ``dt`` seconds split into ``substeps``:
      1. diagnose D, zeta, omega (continuity), Phi = gZ
      2. semi-Lagrangian advection of zeta, D, T, q by (u, v, omega)          [adaptive: advection]
      3. add the non-advective terms of the divergence equation               [adaptive: divergence]
         and the vorticity equation                                           [adaptive: vorticity]
         and the adiabatic warming/cooling kappa*T*omega/p                    [adaptive: thermodynamic]
      4. invert the zeta / D increments to wind increments (psi, chi Poisson solves, zero boundary)
      5. height tendency from linear balance  dPhi = f * dpsi                  [adaptive: height_tendency]
    The neural network then corrects whatever this physics step gets wrong.
    """

    VARS = ("Z", "U", "V", "T", "RH")

    # Largest increment one substep may add to zeta / D (s^-1) and T (K); a tanh soft limiter leaves
    # realistic values untouched and only caps runaway growth (e.g. the -D^2 term blows up in finite time
    # on noisy 3 km fields or on a blocky first-guess analysis).
    MAX_STEP = {"vorticity": 5e-4, "divergence": 5e-4, "temperature": 3.0}

    def __init__(self, levels_hpa, dx: float, substeps: int = 2, trainable: bool = True,
                 source_smoothing_points: float = 1.5, boundary_relax_points: int = 12):
        super().__init__()
        self.source_smoothing = float(source_smoothing_points)
        self.boundary_relax_points = int(boundary_relax_points)
        self.levels_hpa = tuple(float(p) for p in levels_hpa)
        if list(self.levels_hpa) != sorted(self.levels_hpa, reverse=True):
            raise ValueError("levels must be ordered from the bottom (high pressure) to the top")
        nl = len(self.levels_hpa)
        self.register_buffer("p_pa", torch.tensor(self.levels_hpa) * 100.0)
        self.dx, self.substeps = float(dx), int(substeps)
        self.advection = AdaptiveEquation("advection", ["horizontal", "vertical"], nl, trainable=trainable,
                                          descriptions={"horizontal": "V.grad (semi-Lagrangian)",
                                                        "vertical": "omega d/dp (semi-Lagrangian)"})
        self.divergence_eq = AdaptiveEquation("divergence", DIVERGENCE_TERMS, nl, trainable=trainable, descriptions={
            "divergence_squared": "-(D)^2", "jacobian": "+2J(u,v)", "tilting": "-grad(omega).dV/dp",
            "beta": "-beta*u  (v df/dx - u df/dy)", "coriolis_vorticity": "+f*zeta",
            "geopotential_laplacian": "-laplacian(Phi)"})
        self.vorticity_eq = AdaptiveEquation("vorticity", VORTICITY_TERMS, nl, trainable=trainable, descriptions={
            "stretching": "-(zeta+f)D", "tilting": "k.(dV/dp x grad omega)", "beta": "-V.grad(f)"})
        self.thermo_eq = AdaptiveEquation("thermodynamic", ["adiabatic"], nl, trainable=trainable,
                                          descriptions={"adiabatic": "+kappa*T*omega/p"})
        self.height_eq = AdaptiveEquation("height_tendency", ["balanced"], nl, trainable=trainable,
                                          descriptions={"balanced": "dPhi = f * dpsi (linear balance)"})

    @property
    def equations(self):
        return [self.advection, self.divergence_eq, self.vorticity_eq, self.thermo_eq, self.height_eq]

    def prior_penalty(self):
        return sum(eq.prior_penalty() for eq in self.equations)

    def coefficient_table(self) -> dict:
        return {eq.name: {"terms": eq.coefficient_table(self.levels_hpa), "descriptions": eq.descriptions}
                for eq in self.equations}

    # ------------------------------------------------------------------ diagnostics
    def diagnose(self, state, geo) -> dict:
        """state [B, 5, L, H, W] -> dict of derived fields [B, L, H, W]."""
        z, u, v, t, rh = state.unbind(1)
        m = _geo(geo, "map_factor")
        div = divergence(u, v, m, self.dx)
        zeta = vorticity(u, v, m, self.dx)
        omega = kinematic_omega(div, self.p_pa)
        p_hpa = (self.p_pa / 100.0).view(1, -1, 1, 1)
        return {"divergence": div, "vorticity": zeta, "omega": omega,
                "absolute_vorticity": zeta + _geo(geo, "coriolis"),
                "wind_speed": wind_speed(u, v), "theta": potential_temperature(t, p_hpa),
                "theta_e": equivalent_potential_temperature(t, rh, p_hpa)}

    # ------------------------------------------------------------------ one physics step
    def forward(self, state, geo, dt: float = 3600.0):
        for _ in range(self.substeps):
            state = self._substep(state, geo, dt / self.substeps)
        return state

    def _substep(self, state, geo, dt):
        z, u, v, t, rh = state.unbind(1)
        m = _geo(geo, "map_factor")
        f = _geo(geo, "coriolis")
        fx, fy = gradient(f, m, self.dx)
        p_hpa = (self.p_pa / 100.0).view(1, -1, 1, 1)
        dx = self.dx

        div = divergence(u, v, m, dx)
        zeta = vorticity(u, v, m, dx)
        omega = kinematic_omega(div, self.p_pa)
        q = specific_humidity_from_rh(t, rh, p_hpa)
        # source terms are evaluated on lightly smoothed fields (removes 2-grid-length noise that the
        # nonlinear terms would otherwise amplify); advection and the increments keep full resolution
        sm = lambda a: gaussian_smooth(a, self.source_smoothing)
        us, vs_, zs, ts = sm(u), sm(v), sm(z), sm(t)
        div_s, zeta_s = divergence(us, vs_, m, dx), vorticity(us, vs_, m, dx)
        omega_s = kinematic_omega(div_s, self.p_pa)
        phi_s = G * (zs - zs.mean(dim=(-2, -1), keepdim=True))   # anomaly: same derivatives, better precision

        # 2. advection (semi-Lagrangian, adaptive strength per level)
        adv = semi_lagrangian(torch.stack([zeta, div, t, q], 1), u, v, omega, m, dx, self.p_pa, dt,
                              h_scale=self.advection.coeff[0], v_scale=self.advection.coeff[1])
        zeta_a, div_a, t_a, q_a = adv.unbind(1)

        # 3. sources
        d_terms = equation_terms_divergence(us, vs_, zeta_s, div_s, omega_s, phi_s, f, fx, fy, m, dx, self.p_pa)
        z_terms = equation_terms_vorticity(us, vs_, zeta_s, div_s, omega_s, f, fx, fy, m, dx, self.p_pa)
        lim = self.MAX_STEP
        div_new = div_a + soft_limit(dt * self.divergence_eq(d_terms), lim["divergence"])
        zeta_new = zeta_a + soft_limit(dt * self.vorticity_eq(z_terms), lim["vorticity"])
        t_new = t_a + soft_limit(dt * self.thermo_eq({"adiabatic": KAPPA * ts * omega_s / self.p_pa.view(1, -1, 1, 1)}),
                                 lim["temperature"])
        q_new = torch.clamp(q_a, min=1e-9)

        # Lateral relaxation zone (Davies-style): increments taper to zero toward the domain edge. There are no
        # parent-model boundary conditions yet, so the edge persists and the network corrects it. The taper is
        # applied to the vorticity/divergence increments BEFORE inversion - tapering the wind increments instead
        # would itself create divergence (grad(w) . dV) in the relaxation zone.
        if self.boundary_relax_points > 0:
            w = boundary_taper(*div.shape[-2:], self.boundary_relax_points, div.device, div.dtype)
            div_new, zeta_new = div + w * (div_new - div), zeta + w * (zeta_new - zeta)
            t_new, q_new = t + w * (t_new - t), q + w * (q_new - q)

        # 4. wind increments from vorticity / divergence increments
        m2 = m * m
        dpsi = solve_poisson_dirichlet((zeta_new - zeta) / m2, dx)
        dchi = solve_poisson_dirichlet((div_new - div) / m2, dx)
        psi_x, psi_y = gradient(dpsi, m, dx)
        chi_x, chi_y = gradient(dchi, m, dx)
        u_new = u - psi_y + chi_x
        v_new = v + psi_x + chi_y

        # 5. balanced height tendency
        z_new = z + self.height_eq.c("balanced") * f * dpsi / G

        rh_new = torch.clamp(rh_from_specific_humidity(q_new, t_new, p_hpa), 0.0, 100.0)
        return torch.stack([z_new, u_new, v_new, t_new, rh_new], 1)
