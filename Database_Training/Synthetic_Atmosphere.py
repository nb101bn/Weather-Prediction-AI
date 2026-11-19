"""
Synthetic_Atmosphere.py
=======================
A fake-but-physically-shaped upper troposphere used for OFFLINE development and tests only
(everything it produces is labelled source='synthetic' and the maps carry a SYNTHETIC banner).

It stands in for both data sources of the real pipeline, with the same interfaces:
  * ``fetch_frame(init, fhour, levels, grid)`` -> ModelFrame   (like GribModelFetcher: "HRRR ground truth")
  * ``fetch_obs(init, levels, stations)``       -> obs table    (like RaobObservations: radiosondes)

The atmosphere: a meandering jet (trough/ridge wave moving east) with a jet streak, heights that fall
poleward, geostrophic winds on the 3 km grid, a divergent (ageostrophic) wind placed downstream of the
troughs (upper divergence ahead of troughs, as in the real atmosphere), a few short-lived convective
outflow blobs (CAM-scale divergence at 250-300 hPa), temperatures from the hypsometric equation and a
moist plume ahead of the trough. Each initialisation gets its own random weather (seeded by the date).
"""
from __future__ import annotations

import datetime
import math
import zlib

import numpy as np
import pandas as pd

from Database_Training.Grid_Interpolation import OBS_COLUMNS, STATE_VARS, ModelFrame
from Models.Model_Grid import HRRR_CONUS, LambertGrid, grid_to_earth_winds

G, RD = 9.80665, 287.04
_P = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100], dtype=float)
_Z_STD = np.array([111, 762, 1457, 3012, 5574, 7185, 9164, 10363, 11784, 13608, 16180], dtype=float)
_AMP = np.array([0.25, 0.3, 0.35, 0.45, 0.6, 0.8, 0.95, 1.0, 0.9, 0.65, 0.45])       # wave amplitude vs height
_DIV = np.array([-0.6, -0.8, -1.0, -0.6, -0.1, 0.4, 0.9, 1.0, 0.7, 0.3, 0.05])       # divergent-wind profile
_RH = np.array([75, 72, 68, 55, 45, 42, 38, 32, 20, 10, 5], dtype=float)


def _profile(table, p):
    """ln-p interpolation of one of the vertical profile tables (extrapolated linearly at the ends)."""
    x = np.log(_P[::-1])
    y = table[::-1]
    lp = np.log(np.asarray(p, dtype=float))
    out = np.interp(lp, x, y)
    slope_top = (y[1] - y[0]) / (x[1] - x[0])
    slope_bot = (y[-1] - y[-2]) / (x[-1] - x[-2])
    out = np.where(lp < x[0], y[0] + slope_top * (lp - x[0]), out)
    return np.where(lp > x[-1], y[-1] + slope_bot * (lp - x[-1]), out)


class SyntheticAtmosphere:
    source = "synthetic"

    def __init__(self, seed: int = 0, obs_noise: bool = True):
        self.seed, self.obs_noise = seed, obs_noise

    # ------------------------------------------------------------------ weather of one initialisation
    def _params(self, init_time: datetime.datetime) -> dict:
        rng = np.random.default_rng(zlib.crc32(f"{init_time:%Y%m%d%H}|{self.seed}".encode()))
        wavelength = rng.uniform(2200e3, 3800e3)
        k = 2 * math.pi / wavelength
        blobs = []
        for _ in range(int(rng.integers(1, 5))):
            blobs.append({"lat": rng.uniform(30, 46), "lon": rng.uniform(-108, -84), "radius": rng.uniform(20e3, 45e3),
                          "div": rng.uniform(0.8e-4, 2.5e-4), "start": rng.uniform(-3, 4), "life": rng.uniform(3, 7),
                          "drift_u": rng.uniform(5, 20), "drift_v": rng.uniform(-5, 5)})
        return {
            "jet_lat": rng.uniform(35.5, 44.0), "jet_dz": rng.uniform(260.0, 420.0), "jet_width_deg": rng.uniform(3.0, 4.5),
            "k": k, "meander_deg": rng.uniform(2.0, 6.0), "phase": rng.uniform(0, 2 * math.pi),
            "c": rng.uniform(8.0, 16.0), "streak_x": rng.uniform(-1500e3, 1500e3), "streak_c": rng.uniform(18.0, 30.0),
            "streak_len": rng.uniform(600e3, 1000e3), "streak_amp": rng.uniform(0.25, 0.55),
            "d_amp": rng.uniform(1.5e-5, 3.5e-5), "z_grad": rng.uniform(8.0, 16.0), "amp_trend": rng.uniform(-0.04, 0.04),
            "moist": rng.uniform(25.0, 45.0), "blobs": blobs,
        }

    # ------------------------------------------------------------------ analytic pieces
    def _height(self, prm, lat, X, t, p):
        theta = prm["k"] * (X - prm["c"] * t) + prm["phase"]
        amp = prm["meander_deg"] * (1.0 + prm["amp_trend"] * t / 3600.0)
        lat_c = prm["jet_lat"] + amp * np.sin(theta)
        streak = 1.0 + prm["streak_amp"] * np.exp(-((X - prm["streak_x"] - prm["streak_c"] * t) / prm["streak_len"]) ** 2)
        a = _profile(_AMP, p)
        z = (_profile(_Z_STD, p) - a * 0.5 * prm["jet_dz"] * streak * np.tanh((lat - lat_c) / prm["jet_width_deg"])
             - a * prm["z_grad"] * (lat - 40.0))
        return z, theta, lat_c

    def _chi(self, prm, lat, lon, X, t, p, theta, lat_c):
        """Velocity potential of the divergent wind (D = laplacian(chi))."""
        env = np.exp(-((lat - lat_c) / (1.6 * prm["jet_width_deg"])) ** 2)
        chi = -(prm["d_amp"] / prm["k"] ** 2) * _profile(_DIV, p) * np.cos(theta) * env
        hours = t / 3600.0
        for b in prm["blobs"]:
            age = (hours - b["start"]) / b["life"]
            if not 0.0 < age < 1.0:
                continue
            strength = math.sin(math.pi * age)
            blat = b["lat"] + b["drift_v"] * t / 111e3
            blon = b["lon"] + b["drift_u"] * t / (111e3 * math.cos(math.radians(b["lat"])))
            r2 = ((lat - blat) * 111e3) ** 2 + ((lon - blon) * 111e3 * np.cos(np.radians(lat))) ** 2
            upper = max(_profile(_DIV, p), 0.0) ** 2      # outflow aloft only
            chi = chi - strength * upper * b["div"] * b["radius"] ** 2 / 2.0 * np.exp(-r2 / (2 * b["radius"] ** 2))
        return chi

    def _fields_on(self, prm, grid: LambertGrid, t: float, levels) -> np.ndarray:
        lat, lon = grid.latlon()
        X, _ = grid.xy()
        m = grid.map_factor(lat)
        f = np.where(np.abs(grid.coriolis(lat)) < 3e-5, 3e-5, grid.coriolis(lat))
        out = np.zeros((len(STATE_VARS), len(levels), grid.ny, grid.nx), dtype=np.float64)
        for kk, p in enumerate(levels):
            z, theta, lat_c = self._height(prm, lat, X, t, p)
            dzdy, dzdx = np.gradient(z, grid.dx)
            u = -(G / f) * m * dzdy
            v = (G / f) * m * dzdx
            chi = self._chi(prm, lat, lon, X, t, p, theta, lat_c)
            dcdy, dcdx = np.gradient(chi, grid.dx)
            u, v = u + m * dcdx, v + m * dcdy
            # hypsometric temperature: T = -(g/R) dZ/dln(p)
            zp, _, _ = self._height(prm, lat, X, t, p * 1.05)
            zm, _, _ = self._height(prm, lat, X, t, p / 1.05)
            temp = -(G / RD) * (zp - zm) / (math.log(p * 1.05) - math.log(p / 1.05))
            plume = prm["moist"] * np.exp(-((lat - lat_c + 1.5) / 3.0) ** 2) * np.clip(np.cos(theta), 0, None)
            rh = np.clip(_profile(_RH, p) + plume * (p / 300.0) ** 0.5, 2.0, 100.0)
            out[:, kk] = np.stack([z, u, v, temp, rh])
        return out.astype(np.float32)

    # ------------------------------------------------------------------ public interfaces
    def fetch_frame(self, init_time: datetime.datetime, fhour: int, levels, grid: LambertGrid) -> ModelFrame:
        prm = self._params(init_time)
        fields = self._fields_on(prm, grid, fhour * 3600.0, levels)
        return ModelFrame(fields, tuple(levels), grid, init_time, int(fhour), self.source, True,
                          meta={"note": "SYNTHETIC truth - not real weather"})

    def fetch_obs(self, init_time: datetime.datetime, levels, stations=None) -> pd.DataFrame:
        """Radiosonde-like samples of the synthetic atmosphere at the real RAOB sites (with noise,
        random missing soundings and early balloon bursts)."""
        from Database_Training.Upper_Air_Obs import RAOB_STATIONS, check_observed_init
        check_observed_init(init_time)
        prm = self._params(init_time)
        rng = np.random.default_rng(zlib.crc32(f"obs|{init_time:%Y%m%d%H}|{self.seed}".encode()))
        rows = []
        for wmo in stations or list(RAOB_STATIONS):
            _, _, slat, slon, _ = RAOB_STATIONS[wmo]
            if self.obs_noise and rng.random() < 0.05:
                continue                                   # missing sounding
            local = HRRR_CONUS.window_around(slat, slon, 5, 5)
            vals = self._fields_on(prm, local, 0.0, levels)
            i, j = local.index_of(slat, slon)
            i, j = int(np.clip(round(float(i)), 0, 4)), int(np.clip(round(float(j)), 0, 4))
            col = vals[:, :, j, i].astype(float)
            _, lon_c = local.latlon()
            ue, ve = grid_to_earth_winds(col[1], col[2], local.rotation_angle(lon_c[j, i]))
            burst = rng.random() < 0.1
            for kk, p in enumerate(levels):
                z, t, rh = col[0, kk], col[3, kk], col[4, kk]
                u, v = ue[kk], ve[kk]
                if self.obs_noise:
                    z, t = z + rng.normal(0, 5.0), t + rng.normal(0, 0.5)
                    u, v, rh = u + rng.normal(0, 1.5), v + rng.normal(0, 1.5), float(np.clip(rh + rng.normal(0, 5), 1, 100))
                if burst and p < 200:
                    z = t = rh = u = v = np.nan
                rows.append([wmo, slat, slon, float(p), z, u, v, t, rh])
        table = pd.DataFrame(rows, columns=OBS_COLUMNS)
        table.attrs.update({"source": self.source, "init_time": init_time.isoformat()})
        return table
