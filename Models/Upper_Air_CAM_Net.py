"""
Upper_Air_CAM_Net.py
====================
The upper-air map model: a CNN + RNN (ConvGRU) network on a 3 km convection-allowing grid that
steps the atmosphere forward in hourly steps, with adaptive physics equations inside the loop.

    radiosonde obs --nearest neighbour--> 3 km grid
          |
          v
    AnalysisNet (CNN U-Net)           obs grid, the same smoothed at 50/100/200 km, distance-to-obs, static
          |                           fields -> correction to the 100 km first guess  ->  analysis (f00)
          |
          v   for each hour (1 .. n_steps, architecture allows up to 48):
    PhysicsCore (Atmos_Physics.py)    adaptive divergence / vorticity / thermodynamic / height-tendency
          |                           equations (learnable term coefficients)      ->  physics first guess
          v
    ForecastCell (CNN encoder -> ConvGRU memory -> CNN decoder)
                                      corrects the physics first guess             ->  state at next hour

State = (Z geopotential height m, U, V grid-relative wind m/s, T K, RH %) on ``levels`` (hPa).
Derived fields (divergence, vorticity, omega, theta, theta-e, wind speed) come from Atmos_Physics.

Import it anywhere (training, tests, deployment):

    from Models.Upper_Air_CAM_Net import UpperAirCAMNet, CAMNetConfig, load_checkpoint
"""
from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models import Atmos_Physics as ap

STATE_VARS = ("Z", "U", "V", "T", "RH")
DIAG_VARS = ("divergence", "vorticity", "omega")
DEFAULT_LEVELS = (500, 400, 300, 250, 200, 150, 100)

# Rough climatology used until real statistics are computed from the training data.
_Z_STD_ATM = {1000: 111, 925: 762, 850: 1457, 700: 3012, 500: 5574, 400: 7185, 300: 9164, 250: 10363,
              200: 11784, 150: 13608, 100: 16180}
_T_STD_ATM = {1000: 288, 925: 284, 850: 279, 700: 269, 500: 252, 400: 241, 300: 229, 250: 221, 200: 217,
              150: 217, 100: 217}


@dataclass
class CAMNetConfig:
    levels: tuple = DEFAULT_LEVELS
    dx: float = 3000.0                 # grid spacing (m) - 3 km CAM grid
    width: int = 32                    # base CNN channels
    hidden: int = 96                   # ConvGRU channels (at 1/8 resolution = 24 km)
    analysis_depth: int = 5            # U-Net depth of the analysis CNN (receptive field > 1000 km)
    use_physics: bool = True
    physics_substeps: int = 2          # 2 x 30 min per hourly step (semi-Lagrangian: stable at any step)
    step_hours: float = 1.0
    max_lead_hours: int = 48           # architecture limit; trained lead is recorded in the checkpoint
    map_level: int = 300
    # The nearest-neighbour obs grid is also given to the analysis CNN smoothed at these scales (km); the
    # analysis is learned as a correction to the ``first_guess_km`` version (a Barnes-like first guess).
    smoothing_scales_km: tuple = (50.0, 100.0, 200.0)
    first_guess_km: float = 100.0

    @property
    def n_state(self) -> int:
        return len(STATE_VARS) * len(self.levels)


def default_normalization(levels) -> dict:
    nl = len(levels)
    mean = np.zeros((5, nl), np.float32)
    std = np.zeros((5, nl), np.float32)
    for k, p in enumerate(levels):
        mean[:, k] = [_Z_STD_ATM.get(int(p), 9000), 12.0, 0.0, _T_STD_ATM.get(int(p), 230), 40.0]
        std[:, k] = [120.0, 15.0, 15.0, 5.0, 25.0]
    delta = np.tile(np.array([[8.0], [2.0], [2.0], [0.4], [6.0]], np.float32), (1, nl))
    diag = np.tile(np.array([[2e-5], [5e-5], [0.3]], np.float32), (1, nl))
    return {"state_mean": mean, "state_std": std, "delta_std": delta, "diag_std": diag}


# ====================================================================== first guess from gridded obs
def fill_missing_obs(obs, levels):
    """NaN (no observation) -> domain mean of that variable/level, or climatology if the whole field is missing.
    Uses fixed climatology (not data statistics) so training samples and deployment fill identically."""
    clim = torch.as_tensor(default_normalization(levels)["state_mean"], dtype=obs.dtype, device=obs.device)
    have = torch.isfinite(obs)
    count = have.sum(dim=(-1, -2), keepdim=True)
    mean = torch.where(have, obs, torch.zeros_like(obs)).sum(dim=(-1, -2), keepdim=True) / count.clamp(min=1)
    mean = torch.where(count > 0, mean, clim.view(*([1] * (obs.dim() - 4)), *clim.shape, 1, 1).expand_as(mean))
    return torch.where(have, obs, mean.expand_as(obs)), have


def smooth_at_scale(x, sigma_km: float, dx: float):
    """Gaussian smoothing of [..., H, W] at ``sigma_km``; large scales are done on a pooled grid (fast, same result
    up to interpolation error). Must be applied to the FULL domain (edges use replicate padding)."""
    sigma = sigma_km * 1000.0 / dx
    shape = x.shape
    y = x.reshape(-1, 1, *shape[-2:])
    pool = max(1, int(sigma // 4))
    if pool > 1:
        y = F.avg_pool2d(F.pad(y, (0, (-shape[-1]) % pool, 0, (-shape[-2]) % pool), mode="replicate"), pool)
    y = ap.gaussian_smooth(y, sigma / pool)
    if pool > 1:
        y = F.interpolate(y, scale_factor=pool, mode="bilinear", align_corners=False)[..., :shape[-2], :shape[-1]]
    return y.reshape(shape)


def first_guess_stack(obs, levels, dx: float, scales_km):
    """obs [B, 5, L, H, W] (NaN = none) -> (filled, have-mask, [B, S, 5, L, H, W] smoothed versions)."""
    filled, have = fill_missing_obs(obs, levels)
    smoothed = torch.stack([smooth_at_scale(filled, s, dx) for s in scales_km], 1)
    return filled, have, smoothed


# ====================================================================== building blocks
def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation), _gn(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.GELU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


def _zero_init(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def _pad_to_multiple(x, multiple: int):
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (h, w)


class UNet(nn.Module):
    """Plain U-Net; the output layer starts at zero so an untrained net adds no correction."""

    def __init__(self, cin: int, cout: int, width: int, depth: int):
        super().__init__()
        chans = [width * min(2 ** i, 8) for i in range(depth + 1)]
        self.depth = depth
        self.inc = ConvBlock(cin, chans[0])
        self.downs = nn.ModuleList([nn.Sequential(nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1), _gn(chans[i + 1]),
                                                  nn.GELU(), ConvBlock(chans[i + 1], chans[i + 1]))
                                    for i in range(depth)])
        self.bottleneck = ConvBlock(chans[-1], chans[-1], dilation=2)
        self.ups = nn.ModuleList([ConvBlock(chans[i + 1] + chans[i], chans[i]) for i in reversed(range(depth))])
        self.out = _zero_init(nn.Conv2d(chans[0], cout, 1))

    def forward(self, x):
        x, (h, w) = _pad_to_multiple(x, 2 ** self.depth)
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        y = self.bottleneck(skips.pop())
        for up in self.ups:
            skip = skips.pop()
            y = up(torch.cat([F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return self.out(y)[..., :h, :w]


class ConvGRUCell(nn.Module):
    """Convolutional GRU: the RNN memory carried from one forecast hour to the next."""

    def __init__(self, cin: int, hidden: int, k: int = 3):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(cin + hidden, 2 * hidden, k, padding=k // 2)
        self.cand = nn.Conv2d(cin + hidden, hidden, k, padding=k // 2)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden, *x.shape[-2:], device=x.device, dtype=x.dtype)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class ForecastCell(nn.Module):
    """CNN encoder (3 km -> 24 km) -> ConvGRU -> CNN decoder with skip connections (24 km -> 3 km)."""

    def __init__(self, cin: int, cout: int, width: int, hidden: int):
        super().__init__()
        c0, c1, c2 = width, width * 2, width * 4
        self.e0 = ConvBlock(cin, c0)
        self.e1 = nn.Sequential(nn.Conv2d(c0, c1, 3, stride=2, padding=1), nn.GELU(), ConvBlock(c1, c1))
        self.e2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(), ConvBlock(c2, c2))
        self.e3 = nn.Sequential(nn.Conv2d(c2, hidden, 3, stride=2, padding=1), nn.GELU(), ConvBlock(hidden, hidden, 2))
        self.gru = ConvGRUCell(hidden, hidden)
        self.d2 = ConvBlock(hidden + c2, c2)
        self.d1 = ConvBlock(c2 + c1, c1)
        self.d0 = ConvBlock(c1 + c0, c0)
        self.out = _zero_init(nn.Conv2d(c0, cout, 1))

    def forward(self, x, h=None):
        x, (hh, ww) = _pad_to_multiple(x, 8)
        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        h = self.gru(self.e3(s2), h)
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        y = self.d2(torch.cat([up(h, s2), s2], 1))
        y = self.d1(torch.cat([up(y, s1), s1], 1))
        y = self.d0(torch.cat([up(y, s0), s0], 1))
        return self.out(y)[..., :hh, :ww], h


# ====================================================================== the model
class UpperAirCAMNet(nn.Module):
    N_STATIC = 6
    N_TIME = 5

    def __init__(self, config: CAMNetConfig | None = None):
        super().__init__()
        self.config = config = config or CAMNetConfig()
        nl, ns = len(config.levels), config.n_state
        self.physics = ap.PhysicsCore(config.levels, config.dx, substeps=config.physics_substeps)
        if config.first_guess_km not in config.smoothing_scales_km:
            raise ValueError("first_guess_km must be one of smoothing_scales_km")
        n_scales = len(config.smoothing_scales_km)
        self.analysis_net = UNet((2 + n_scales) * ns + nl + self.N_STATIC + self.N_TIME, ns, config.width,
                                 config.analysis_depth)
        self.forecast_cell = ForecastCell(2 * ns + len(DIAG_VARS) * nl + self.N_STATIC + self.N_TIME, ns,
                                          config.width, config.hidden)
        for name, value in default_normalization(config.levels).items():
            self.register_buffer(name, torch.tensor(value))
        self.map_level_index = list(config.levels).index(config.map_level) if config.map_level in config.levels else None

    # ------------------------------------------------------------------ normalisation
    def set_normalization(self, stats: dict) -> None:
        for name in ("state_mean", "state_std", "delta_std", "diag_std"):
            if name in stats:
                value = torch.as_tensor(np.asarray(stats[name], np.float32))
                getattr(self, name).copy_(torch.where(value.abs() < 1e-12, torch.ones_like(value), value)
                                          if name != "state_mean" else value)

    def _norm(self, s):
        return (s - self.state_mean[None, :, :, None, None]) / self.state_std[None, :, :, None, None]

    @staticmethod
    def _flat(x):
        return x.reshape(x.shape[0], -1, *x.shape[-2:])

    def _unflat(self, x):
        return x.reshape(x.shape[0], len(STATE_VARS), len(self.config.levels), *x.shape[-2:])

    # ------------------------------------------------------------------ features
    def static_features(self, geo):
        lat, lon = torch.deg2rad(geo["lat"]), torch.deg2rad(geo["lon"])
        feats = [torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon),
                 (geo["map_factor"] - 1.0) * 20.0, geo["coriolis"] / 1e-4]
        return torch.stack(feats, 1)

    def time_features(self, hours_utc, day_of_year, lead_hours, like):
        b, _, h, w = like.shape
        hr = 2 * math.pi * hours_utc / 24.0
        dy = 2 * math.pi * day_of_year / 365.25
        lead = torch.full_like(hr, float(lead_hours) / self.config.max_lead_hours)
        feats = torch.stack([torch.sin(hr), torch.cos(hr), torch.sin(dy), torch.cos(dy), lead], 1)
        return feats[:, :, None, None].expand(b, self.N_TIME, h, w)

    # ------------------------------------------------------------------ analysis
    def first_guess(self, obs):
        cfg = self.config
        return first_guess_stack(obs, cfg.levels, cfg.dx, cfg.smoothing_scales_km)

    def analysis(self, obs, obs_dist, geo, init_hour, day_of_year, smoothed=None):
        """Nearest-neighbour gridded observations (NaN = no data) -> 3 km analysis (f00).

        ``smoothed`` ([B, S, 5, L, H, W]) may be precomputed on the full domain (training crops); otherwise it is
        computed here, which is correct whenever ``obs`` covers the whole domain (deployment)."""
        cfg = self.config
        filled, have = fill_missing_obs(obs, cfg.levels)
        if smoothed is None:
            _, _, smoothed = self.first_guess(obs)
        reference = smoothed[:, list(cfg.smoothing_scales_km).index(cfg.first_guess_km)]
        static = self.static_features(geo)
        x = torch.cat([self._flat(self._norm(filled))] + [self._flat(self._norm(smoothed[:, i]))
                                                         for i in range(smoothed.shape[1])]
                      + [self._flat(have.float()), torch.clamp(obs_dist, max=2000.0) / 500.0, static,
                         self.time_features(init_hour, day_of_year, 0.0, static)], 1)
        correction = self._unflat(self.analysis_net(x)) * self.state_std[None, :, :, None, None]
        return self._clip(reference + correction)

    @staticmethod
    def _clip(s):
        rh = torch.clamp(s[:, 4], 0.0, 100.0)
        return torch.cat([s[:, :4], rh[:, None]], 1)

    # ------------------------------------------------------------------ forecast
    def step(self, state, geo, hidden, hours_utc, day_of_year, lead_hours):
        """One hourly step: physics first guess + ConvGRU correction."""
        dt = 3600.0 * self.config.step_hours
        s_phys = self.physics(state, geo, dt) if self.config.use_physics else state
        diag = self.physics.diagnose(s_phys, geo)
        d = torch.stack([diag[n] for n in DIAG_VARS], 1) / self.diag_std[None, :, :, None, None]
        static = self.static_features(geo)
        x = torch.cat([self._flat(self._norm(state)),
                       self._flat((s_phys - state) / self.delta_std[None, :, :, None, None]),
                       self._flat(torch.clamp(d, -20, 20)), static,
                       self.time_features(hours_utc, day_of_year, lead_hours, static)], 1)
        delta, hidden = self.forecast_cell(x, hidden)
        new = s_phys + self._unflat(delta) * self.delta_std[None, :, :, None, None]
        return self._clip(new), s_phys, hidden

    def forward(self, obs, obs_dist, geo, init_hour, day_of_year, n_steps: int = 2, start_state=None,
                return_physics: bool = False, smoothed=None):
        """Returns {"analysis": [B,5,L,H,W], "states": [B, n_steps+1, 5, L, H, W]} in physical units.

        states[:, 0] is the forecast's starting state (the analysis, or ``start_state`` if given - used
        for teacher forcing from the ground-truth f00 during training); states[:, h] is lead hour h.
        """
        if n_steps > self.config.max_lead_hours / self.config.step_hours:
            raise ValueError(f"n_steps={n_steps} exceeds max_lead_hours={self.config.max_lead_hours}")
        analysis = self.analysis(obs, obs_dist, geo, init_hour, day_of_year, smoothed)
        state = analysis if start_state is None else start_state
        states, physics_only, hidden = [state], [], None
        for h in range(1, n_steps + 1):
            lead = h * self.config.step_hours
            state, s_phys, hidden = self.step(state, geo, hidden, (init_hour + lead) % 24, day_of_year, lead)
            states.append(state)
            if return_physics:
                physics_only.append(s_phys)
        out = {"analysis": analysis, "states": torch.stack(states, 1), "geo": geo}
        if return_physics:
            out["physics_first_guess"] = torch.stack(physics_only, 1) if physics_only else None
        return out

    # ------------------------------------------------------------------ reporting
    def coefficient_table(self) -> dict:
        return self.physics.coefficient_table()

    def derived_fields(self, state, geo) -> dict:
        """All derived quantities (divergence, vorticity, omega, theta, theta-e, wind speed, ...)."""
        with torch.no_grad():
            return self.physics.diagnose(state, geo)


# ====================================================================== loss / verification
def map_model_loss(model: UpperAirCAMNet, out: dict, truth, map_level_weight: float = 2.0,
                   divergence_weight: float = 0.5, coefficient_prior: float = 1e-3, div_smooth_points: float = 2.0,
                   teacher_forced: bool = False, analysis_divergence_factor: float = 0.1):
    """truth: [B, T+1, 5, L, H, W] (f00 .. fT). Frame 0 is scored against the analysis.

    Divergence (a derivative field) is only learnable where the state is already good: it gets full weight on
    forecast frames started from the true f00 (teacher forcing) and ``analysis_divergence_factor`` on the
    analysis and on frames started from it. (With full weight there, the cheapest way to cut the loss is to
    smooth away ALL divergence of the sparse-obs analysis, which also flattens the jets.)"""
    cfg = model.config
    pred = torch.cat([out["analysis"][:, None], out["states"][:, 1:]], 1)
    t = pred.shape[1]
    truth = truth[:, :t]
    std = model.state_std[None, None, :, :, None, None]
    w_lev = torch.ones(len(cfg.levels), device=pred.device)
    if model.map_level_index is not None:
        w_lev[model.map_level_index] = map_level_weight
    se = ((pred - truth) / std) ** 2
    state_loss = (se.mean(dim=(-1, -2)) * w_lev[None, None, None, :]).mean() / w_lev.mean()

    geo = out["geo"]
    b = pred.shape[0]
    m = geo["map_factor"].unsqueeze(1).unsqueeze(1)
    div = lambda s: ap.gaussian_smooth(ap.divergence(s[:, :, 1], s[:, :, 2], m, cfg.dx), div_smooth_points)
    d_std = model.diag_std[0][None, None, :, None, None]
    # Huber: the early (blocky nearest-neighbour) analysis has huge spurious divergence on cell edges;
    # a squared error there would swamp every other gradient
    frame_w = torch.full((t,), analysis_divergence_factor, device=pred.device)
    if teacher_forced:
        frame_w[1:] = 1.0
    per_frame = F.smooth_l1_loss(div(pred) / d_std, div(truth) / d_std, beta=2.0, reduction="none").mean(
        dim=(0, 2, 3, 4))
    div_loss = (per_frame * frame_w).sum() / t
    prior = model.physics.prior_penalty()
    total = state_loss + divergence_weight * div_loss + coefficient_prior * prior
    return total, {"state": float(state_loss.detach()), "divergence": float(div_loss.detach()),
                   "coeff_prior": float(prior.detach()), "frames": t, "batch": b}


def verify_map_level(state, truth, geo, dx: float, level_index: int, div_smooth_points: float = 2.0) -> dict:
    """RMSE at one level: height (m), wind speed (kt), vector wind (m/s), divergence (1e-5 s^-1)."""
    with torch.no_grad():
        m = geo["map_factor"].unsqueeze(1)
        k = level_index
        rm = lambda a, b: float(torch.sqrt(((a - b) ** 2).mean()))
        spd = lambda s: torch.sqrt(s[:, 1, k] ** 2 + s[:, 2, k] ** 2)
        dv = lambda s: ap.gaussian_smooth(ap.divergence(s[:, 1], s[:, 2], m, dx)[:, k], div_smooth_points)
        return {"height_m": rm(state[:, 0, k], truth[:, 0, k]),
                "wind_speed_kt": rm(spd(state), spd(truth)) * 1.943844,
                "vector_wind_ms": float(torch.sqrt(((state[:, 1, k] - truth[:, 1, k]) ** 2
                                                    + (state[:, 2, k] - truth[:, 2, k]) ** 2).mean())),
                "divergence_1e5": rm(dv(state), dv(truth)) * 1e5}


# ====================================================================== checkpoints
def save_checkpoint(model: UpperAirCAMNet, path: str, meta: dict | None = None) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg = asdict(model.config)
    cfg["levels"] = list(cfg["levels"])
    torch.save({"config": cfg, "state_dict": model.state_dict(), "meta": meta or {},
                "coefficients": model.coefficient_table(),
                "saved": datetime.datetime.now(datetime.timezone.utc).isoformat()}, path)


def load_checkpoint(path: str, map_location="cpu"):
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = dict(blob["config"])
    cfg["levels"] = tuple(cfg["levels"])
    model = UpperAirCAMNet(CAMNetConfig(**cfg))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


def geometry_tensors(geometry: dict, device="cpu", batch: int = 1) -> dict:
    """LambertGrid.geometry() numpy dict -> batched torch tensors [B, H, W]."""
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)[None].expand(batch, *v.shape).contiguous()
            for k, v in geometry.items()}
