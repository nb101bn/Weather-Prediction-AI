"""
Grid_Interpolation.py
=====================
Puts every data source on the model's 3 km grid (Models/Model_Grid.py) with NEAREST-NEIGHBOUR
interpolation:

    value(grid point g) = value(observation k*),   k* = argmin_k  distance(g, x_k)

Distances are great-circle distances. Internally points are placed on the unit sphere
(x, y, z) = (cos(lat)cos(lon), cos(lat)sin(lon), sin(lat)); straight-line (chord) distance c is a
monotonic function of the great-circle distance d = 2R asin(c/2), so the nearest point by chord is the
nearest point on the earth. A k-d tree makes the search O(N log K).

* ``nearest_neighbor_to_grid``  - scattered observations (radiosondes) -> grid, plus the distance to the
  observation that was used (a confidence field the network receives as an input channel).
* ``grid_observations``         - an observation table (all variables/levels) -> [var, level, ny, nx]
  arrays, with earth-relative winds rotated to the grid-relative convention.
* ``regrid_frame``              - a model field on another grid -> our grid (exact window copy when the
  source grid is the same 3 km HRRR grid, nearest neighbour otherwise).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from Models.Model_Grid import EARTH_RADIUS_M, LambertGrid, earth_to_grid_winds, grid_to_earth_winds

STATE_VARS = ("Z", "U", "V", "T", "RH")           # m, m/s (grid-relative), m/s, K, %
STATE_UNITS = {"Z": "m", "U": "m/s", "V": "m/s", "T": "K", "RH": "%"}


@dataclass
class ModelFrame:
    """One forecast hour of gridded model output (the training "ground truth").

    ``fields``: float32 [len(STATE_VARS), len(levels), ny, nx] on ``grid``; winds are grid-relative to
    ``grid`` when ``grid_relative`` is True, earth-relative otherwise.
    """
    fields: np.ndarray
    levels: tuple
    grid: LambertGrid | None
    init_time: datetime.datetime
    fhour: int
    source: str
    grid_relative: bool = True
    lats: np.ndarray | None = None      # needed only when ``grid`` is None (non-Lambert source)
    lons: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def valid_time(self) -> datetime.datetime:
        return self.init_time + datetime.timedelta(hours=self.fhour)


# ====================================================================== nearest neighbour core
def _unit_vectors(lat, lon):
    lat, lon = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def chord_to_km(chord):
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) / 1000.0


def nearest_neighbor_to_grid(obs_lat, obs_lon, values, grid_lat, grid_lon, max_distance_km: float | None = None):
    """Nearest-neighbour interpolation of scattered values to grid points.

    Returns (gridded values, distance_km to the observation used, index of that observation).
    Points farther than ``max_distance_km`` from every observation are NaN.
    """
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values) & np.isfinite(obs_lat) & np.isfinite(obs_lon)
    shape = np.shape(grid_lat)
    if ok.sum() == 0:
        return np.full(shape, np.nan), np.full(shape, np.inf), np.full(shape, -1)
    tree = cKDTree(_unit_vectors(np.asarray(obs_lat)[ok], np.asarray(obs_lon)[ok]))
    chord, idx = tree.query(_unit_vectors(grid_lat, grid_lon).reshape(-1, 3))
    dist = chord_to_km(chord).reshape(shape)
    out = values[ok][idx].reshape(shape)
    used = np.flatnonzero(ok)[idx].reshape(shape)
    if max_distance_km is not None:
        out = np.where(dist <= max_distance_km, out, np.nan)
    return out, dist, used


# ====================================================================== observations -> grid
OBS_COLUMNS = ["station", "lat", "lon", "level", "Z", "U_earth", "V_earth", "T", "RH"]


def grid_observations(obs: pd.DataFrame, grid: LambertGrid, levels, max_distance_km: float | None = None):
    """Grid an observation table (columns OBS_COLUMNS, one row per station and level).

    Returns ``values`` [5, L, ny, nx] (NaN where a variable/level has no observation at all),
    ``distance_km`` [L, ny, nx] (distance to the nearest station reporting wind+height at that level)
    and ``counts`` {(var, level): number of stations used}.
    """
    lat, lon = grid.latlon()
    angle = grid.rotation_angle(lon)
    nl = len(levels)
    values = np.full((len(STATE_VARS), nl, grid.ny, grid.nx), np.nan, dtype=np.float32)
    distance = np.full((nl, grid.ny, grid.nx), np.inf, dtype=np.float32)
    counts = {}
    for k, p in enumerate(levels):
        rows = obs[np.isclose(obs["level"].astype(float), float(p))]
        if rows.empty:
            continue
        olat, olon = rows["lat"].to_numpy(float), rows["lon"].to_numpy(float)
        for var in ("Z", "T", "RH"):
            out, dist, _ = nearest_neighbor_to_grid(olat, olon, rows[var].to_numpy(float), lat, lon, max_distance_km)
            values[STATE_VARS.index(var), k] = out
            counts[(var, int(p))] = int(np.isfinite(rows[var].to_numpy(float)).sum())
            if var == "Z":
                distance[k] = dist
        # wind: pick the nearest station that has BOTH components, then rotate to the grid at the grid point
        wind_ok = np.where(np.isfinite(rows["U_earth"]) & np.isfinite(rows["V_earth"]), 0.0, np.nan)
        _, wdist, used = nearest_neighbor_to_grid(olat, olon, wind_ok, lat, lon, max_distance_km)
        if (used >= 0).any():
            ue = rows["U_earth"].to_numpy(float)[np.clip(used, 0, None)]
            ve = rows["V_earth"].to_numpy(float)[np.clip(used, 0, None)]
            ug, vg = earth_to_grid_winds(ue, ve, angle)
            bad = (used < 0) | ~np.isfinite(wdist)
            if max_distance_km is not None:
                bad = bad | (wdist > max_distance_km)
            values[STATE_VARS.index("U"), k] = np.where(bad, np.nan, ug)
            values[STATE_VARS.index("V"), k] = np.where(bad, np.nan, vg)
            distance[k] = np.minimum(distance[k], wdist)
        counts[("wind", int(p))] = int((np.isfinite(rows["U_earth"]) & np.isfinite(rows["V_earth"])).sum())
    return values, distance, counts


# ====================================================================== model grid -> our grid
def regrid_frame(frame: ModelFrame, grid: LambertGrid) -> ModelFrame:
    """Put a model frame on ``grid`` with grid-relative winds.

    * same 3 km Lambert grid (e.g. HRRR on a window of the HRRR grid): exact copy of the window;
    * anything else: nearest-neighbour regridding, with winds rotated source-grid -> earth -> our grid.
    """
    if frame.grid is not None:
        offset = grid.offset_in(frame.grid)
        if offset is not None:
            i0, j0 = offset
            sub = frame.fields[..., j0:j0 + grid.ny, i0:i0 + grid.nx].copy()
            if not frame.grid_relative:
                iu, iv = STATE_VARS.index("U"), STATE_VARS.index("V")
                _, lon = grid.latlon()
                sub[iu], sub[iv] = earth_to_grid_winds(sub[iu], sub[iv], grid.rotation_angle(lon))
            return ModelFrame(sub.astype(np.float32), frame.levels, grid, frame.init_time, frame.fhour, frame.source,
                              True, meta={**frame.meta, "regrid": "window"})
        src_lat, src_lon = frame.grid.latlon()
    else:
        src_lat, src_lon = frame.lats, frame.lons
    fields = frame.fields.copy()
    iu, iv = STATE_VARS.index("U"), STATE_VARS.index("V")
    if frame.grid_relative and frame.grid is not None:
        fields[iu], fields[iv] = grid_to_earth_winds(fields[iu], fields[iv], frame.grid.rotation_angle(src_lon))
    lat, lon = grid.latlon()
    tree = cKDTree(_unit_vectors(src_lat.ravel(), src_lon.ravel()))
    _, idx = tree.query(_unit_vectors(lat, lon).reshape(-1, 3))
    out = fields.reshape(*fields.shape[:2], -1)[..., idx].reshape(*fields.shape[:2], grid.ny, grid.nx)
    out[iu], out[iv] = earth_to_grid_winds(out[iu], out[iv], grid.rotation_angle(lon))
    return ModelFrame(out.astype(np.float32), frame.levels, grid, frame.init_time, frame.fhour, frame.source, True,
                      meta={**frame.meta, "regrid": "nearest_neighbor"})
