"""
Model_Grid.py
=============
The 3 km Lambert-conformal grid the upper-air CAM model runs on, shared by the model
(Models/), the data pipeline (Database_Training/) and the map renderer (Maps/).

The default grid *is* the operational HRRR CONUS grid (1799 x 1059 points, 3 km, Lambert
conformal, tangent at 38.5N, LoV 97.5W, spherical earth R = 6 371 229 m). Using the HRRR grid
means HRRR "ground truth" needs no regridding at all, and any window of it (e.g. a region for
training or a demo) is still an exact 3 km sub-grid.

Conventions used everywhere in the project
-----------------------------------------
* Arrays are ``[..., ny, nx]`` with row 0 at the SOUTH edge (y increases with the row index).
* Winds are GRID-RELATIVE (u along +x, v along +y of the grid), which is how the HRRR stores them.
  Use ``earth_to_grid_winds`` / ``grid_to_earth_winds`` to convert observations or to label barbs.
* ``map_factor`` m = (grid distance) / (true distance). True distance between neighbours = dx / m.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np

EARTH_RADIUS_M = 6371229.0      # sphere used by the HRRR / NAM GRIB2 files
OMEGA = 7.292115e-5             # earth rotation rate (s^-1)


@dataclass(frozen=True)
class LambertGrid:
    """A regular grid on a (spherical) Lambert conformal conic projection."""

    nx: int
    ny: int
    dx: float                    # grid spacing in projection metres (dx == dy)
    lat_first: float             # latitude  of grid point [0, 0] (south-west corner)
    lon_first: float             # longitude of grid point [0, 0], degrees east in [-180, 180)
    lov: float = -97.5           # central meridian (orientation longitude)
    latin1: float = 38.5
    latin2: float = 38.5
    radius: float = EARTH_RADIUS_M
    name: str = "hrrr_conus"

    # ------------------------------------------------------------------ projection constants
    @property
    def cone(self) -> float:
        """Cone constant n (= sin(latin) for a tangent projection)."""
        p1, p2 = math.radians(self.latin1), math.radians(self.latin2)
        if abs(p1 - p2) < 1e-10:
            return math.sin(p1)
        return (math.log(math.cos(p1) / math.cos(p2))
                / math.log(math.tan(math.pi / 4 + p2 / 2) / math.tan(math.pi / 4 + p1 / 2)))

    @property
    def _F(self) -> float:
        p1 = math.radians(self.latin1)
        return math.cos(p1) * math.tan(math.pi / 4 + p1 / 2) ** self.cone / self.cone

    @property
    def _rho0(self) -> float:
        return self.radius * self._F / math.tan(math.pi / 4 + math.radians(self.latin1) / 2) ** self.cone

    # ------------------------------------------------------------------ forward / inverse projection
    def project(self, lat, lon):
        """(lat, lon) in degrees -> projection (x, y) in metres (origin: LoV at latitude latin1)."""
        lat = np.radians(np.asarray(lat, dtype=float))
        dlon = (np.asarray(lon, dtype=float) - self.lov + 180.0) % 360.0 - 180.0
        n = self.cone
        rho = self.radius * self._F / np.tan(np.pi / 4 + lat / 2) ** n
        theta = n * np.radians(dlon)
        return rho * np.sin(theta), self._rho0 - rho * np.cos(theta)

    def unproject(self, x, y):
        """Projection (x, y) in metres -> (lat, lon) in degrees."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        n = self.cone
        dy = self._rho0 - y
        rho = np.sign(n) * np.hypot(x, dy)
        theta = np.arctan2(np.sign(n) * x, np.sign(n) * dy)
        lat = 2.0 * np.arctan((self.radius * self._F / rho) ** (1.0 / n)) - np.pi / 2
        lon = (np.degrees(theta) / n + self.lov + 180.0) % 360.0 - 180.0
        return np.degrees(lat), lon

    @property
    def origin_xy(self) -> tuple[float, float]:
        x0, y0 = self.project(self.lat_first, self.lon_first)
        return float(x0), float(y0)

    # ------------------------------------------------------------------ grid coordinates
    def xy_1d(self):
        x0, y0 = self.origin_xy
        return x0 + self.dx * np.arange(self.nx), y0 + self.dx * np.arange(self.ny)

    def xy(self):
        xs, ys = self.xy_1d()
        return np.meshgrid(xs, ys)

    def latlon(self):
        return self.unproject(*self.xy())

    def map_factor(self, lat=None):
        """Scale factor k = n * rho / (R cos(lat)) of the Lambert projection (1.0 on the standard parallels)."""
        if lat is None:
            lat, _ = self.latlon()
        phi = np.radians(np.asarray(lat, dtype=float))
        rho = self.radius * self._F / np.tan(np.pi / 4 + phi / 2) ** self.cone
        return self.cone * rho / (self.radius * np.cos(phi))

    @staticmethod
    def coriolis(lat):
        return 2.0 * OMEGA * np.sin(np.radians(np.asarray(lat, dtype=float)))

    def rotation_angle(self, lon):
        """Angle (radians) between grid +y and true north at longitude ``lon``: n * (lon - LoV)."""
        dlon = (np.asarray(lon, dtype=float) - self.lov + 180.0) % 360.0 - 180.0
        return self.cone * np.radians(dlon)

    def geometry(self) -> dict:
        """Static fields the model/physics need: lat, lon, map factor, Coriolis parameter (float32 [ny, nx])."""
        lat, lon = self.latlon()
        return {"lat": lat.astype(np.float32), "lon": lon.astype(np.float32),
                "map_factor": self.map_factor(lat).astype(np.float32),
                "coriolis": self.coriolis(lat).astype(np.float32)}

    # ------------------------------------------------------------------ windows / comparisons
    def window(self, i0: int, j0: int, nx: int, ny: int, name: str | None = None) -> "LambertGrid":
        """Sub-grid starting at column i0 / row j0 (still exactly the same 3 km grid points)."""
        if i0 < 0 or j0 < 0 or i0 + nx > self.nx or j0 + ny > self.ny:
            raise ValueError(f"window [{i0}:{i0 + nx}, {j0}:{j0 + ny}] is outside the {self.nx}x{self.ny} grid")
        x0, y0 = self.origin_xy
        lat, lon = self.unproject(x0 + i0 * self.dx, y0 + j0 * self.dx)
        return replace(self, nx=int(nx), ny=int(ny), lat_first=float(lat), lon_first=float(lon),
                       name=name or f"{self.name}[{i0}:{i0 + nx},{j0}:{j0 + ny}]")

    def window_around(self, lat: float, lon: float, nx: int, ny: int, name: str | None = None) -> "LambertGrid":
        i, j = self.index_of(lat, lon)
        i0 = int(np.clip(round(i) - nx // 2, 0, self.nx - nx))
        j0 = int(np.clip(round(j) - ny // 2, 0, self.ny - ny))
        return self.window(i0, j0, nx, ny, name)

    def index_of(self, lat, lon):
        """Fractional (i, j) grid index of a lat/lon."""
        x, y = self.project(lat, lon)
        x0, y0 = self.origin_xy
        return (x - x0) / self.dx, (y - y0) / self.dx

    def offset_in(self, parent: "LambertGrid") -> tuple[int, int] | None:
        """(i0, j0) if this grid is an exact window of ``parent`` (same projection & spacing), else None."""
        if not self.same_projection(parent) or abs(self.dx - parent.dx) > 1e-3:
            return None
        i, j = parent.index_of(self.lat_first, self.lon_first)
        ri, rj = int(round(float(i))), int(round(float(j)))
        if abs(i - ri) > 1e-3 or abs(j - rj) > 1e-3:
            return None
        if ri < 0 or rj < 0 or ri + self.nx > parent.nx or rj + self.ny > parent.ny:
            return None
        return ri, rj

    def same_projection(self, other: "LambertGrid") -> bool:
        return (abs(self.lov - other.lov) < 1e-6 and abs(self.latin1 - other.latin1) < 1e-6
                and abs(self.latin2 - other.latin2) < 1e-6 and abs(self.radius - other.radius) < 1e-3)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LambertGrid":
        return cls(**d)

    @property
    def extent_km(self) -> tuple[float, float]:
        return (self.nx - 1) * self.dx / 1000.0, (self.ny - 1) * self.dx / 1000.0


def earth_to_grid_winds(u_earth, v_earth, angle):
    """Rotate earth-relative (east/north) wind components to grid-relative. ``angle`` from rotation_angle()."""
    c, s = np.cos(angle), np.sin(angle)
    return c * u_earth - s * v_earth, s * u_earth + c * v_earth


def grid_to_earth_winds(u_grid, v_grid, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * u_grid + s * v_grid, -s * u_grid + c * v_grid


# The operational HRRR CONUS grid (NCEP grid definition for hrrr.tHHz.wrfprsfFF.grib2).
HRRR_CONUS = LambertGrid(nx=1799, ny=1059, dx=3000.0, lat_first=21.138123, lon_first=-122.719528,
                         lov=-97.5, latin1=38.5, latin2=38.5, name="hrrr_conus")

# Named windows of the HRRR grid (all exactly 3 km). Centre lat/lon, nx, ny.
DOMAIN_PRESETS = {
    "conus": None,                                    # the full 1799 x 1059 grid
    "central_plains": (38.5, -97.5, 512, 448),       # ~1530 x 1340 km, ~15 RAOB sites
    "southern_plains": (34.5, -98.0, 448, 384),
    "northeast": (42.0, -76.0, 448, 384),
    "demo": (38.5, -97.5, 384, 320),                 # small window used by tests / CPU demo
}


def get_domain(name: str = "central_plains") -> LambertGrid:
    if name not in DOMAIN_PRESETS:
        raise ValueError(f"unknown domain '{name}' (choose from {sorted(DOMAIN_PRESETS)})")
    spec = DOMAIN_PRESETS[name]
    if spec is None:
        return HRRR_CONUS
    lat, lon, nx, ny = spec
    return HRRR_CONUS.window_around(lat, lon, nx, ny, name=name)
