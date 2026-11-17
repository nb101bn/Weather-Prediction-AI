"""
Data-Injestor.py
================
Prepares cleaned soundings (Data-Cleaning.py output) for the Point-Forecast NN.

A sounding is reduced to a fixed "column" of rows: the surface plus the mandatory
pressure levels in ``LEVELS_HPA``. Values are interpolated linearly in ln(p), wind
is carried as u/v components (direction is circular and can't be averaged), and
levels that lie below the station's surface are flagged in a mask.

Training pairs are (sounding at init time -> sounding at init + lead). Wyoming soundings
exist at 00Z/12Z only, so a target for e.g. +2 h is *time-interpolated* between the
bracketing soundings. Those samples are counted separately (``n_interpolated``) so
you always know how much of the training signal was real vs interpolated.

Outputs are plain NumPy arrays so this module does not depend on PyTorch.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVELS_HPA = (1000, 925, 850, 700, 500, 400, 300, 250, 200)
ROW_LABELS = ("SFC",) + tuple(f"{p} hPa" for p in LEVELS_HPA)
PROFILE_VARS = ("TEMP", "DWPT", "RELH", "MIXR", "U", "V", "HGHT", "SKNT")
FEATURE_VARS = ("TEMP", "DWPT", "RELH", "MIXR", "U", "V", "HGHT")
# Which profile variable(s) each forecast component predicts.
TARGET_VARS = {
    "temperature": ("TEMP",), "dewpoint": ("DWPT",), "relative_humidity": ("RELH",),
    "mixing_ratio": ("MIXR",), "wind_speed": ("SKNT",), "wind_direction": ("U", "V"), "height": ("HGHT",),
}
DEFAULT_TRAINING_LEADS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 30, 36, 42, 48)
MAX_LEAD_HOURS = 48
N_SCALARS = 9


@dataclass
class Profile:
    time: datetime.datetime
    values: np.ndarray        # [rows, len(PROFILE_VARS)]
    below_ground: np.ndarray  # [rows] bool
    psfc: float

    def var(self, name: str) -> np.ndarray:
        return self.values[:, PROFILE_VARS.index(name)]


def wind_components(direction_deg, speed):
    rad = np.deg2rad(np.asarray(direction_deg, dtype=float))
    speed = np.asarray(speed, dtype=float)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def wind_direction(u, v):
    """Meteorological direction (degrees the wind blows FROM) from u/v."""
    return np.mod(np.rad2deg(np.arctan2(-np.asarray(u, dtype=float), -np.asarray(v, dtype=float))), 360.0)


def _interp_log_p(pres_desc: np.ndarray, values: np.ndarray, targets_hpa) -> np.ndarray:
    """Interpolate in ln(p); outside the observed range the nearest valid value is held constant."""
    ok = ~np.isnan(values)
    if ok.sum() == 0:
        return np.full(len(targets_hpa), np.nan)
    x = np.log(pres_desc[ok])[::-1]
    return np.interp(np.log(np.asarray(targets_hpa, dtype=float)), x, values[ok][::-1])


class SoundingInjestor:
    def __init__(self, levels=LEVELS_HPA, history_step_hours: int = 12):
        self.levels = tuple(levels)
        self.history_step = datetime.timedelta(hours=history_step_hours)
        self.n_rows = len(self.levels) + 1
        self.n_channels = 2 * len(FEATURE_VARS) + 1  # current, 12-h tendency, below-ground mask

    # ------------------------------------------------------------------ profiles
    def to_profile(self, df: pd.DataFrame, time: datetime.datetime) -> Profile:
        df = df.sort_values("PRES", ascending=False)
        pres = df["PRES"].to_numpy(dtype=float)
        psfc = float(pres[0])
        u, v = wind_components(df["DRCT"], df["SKNT"])
        columns = {name: df[name].to_numpy(dtype=float) for name in ("TEMP", "DWPT", "RELH", "MIXR", "HGHT", "SKNT")}
        columns.update({"U": u, "V": v})
        targets = (psfc,) + self.levels
        values = np.stack([_interp_log_p(pres, columns[name], targets) for name in PROFILE_VARS], axis=1)
        below = np.array([False] + [p > psfc for p in self.levels])
        if np.isnan(values).any():
            raise ValueError(f"sounding at {time} is missing an entire variable")
        return Profile(time=time, values=values, below_ground=below, psfc=psfc)

    def profiles_from(self, soundings) -> dict:
        profiles = {}
        for when, df in soundings:
            try:
                profiles[when] = self.to_profile(df, when)
            except ValueError:
                continue
        return profiles

    @staticmethod
    def blend(a: Profile, b: Profile, weight: float, when: datetime.datetime) -> Profile:
        return Profile(time=when, values=(1 - weight) * a.values + weight * b.values,
                       below_ground=a.below_ground | b.below_ground, psfc=(1 - weight) * a.psfc + weight * b.psfc)

    # ------------------------------------------------------------------ features
    def features(self, current: Profile, previous: Profile | None, lead_hours: float):
        idx = [PROFILE_VARS.index(v) for v in FEATURE_VARS]
        now = current.values[:, idx]
        tendency = now - previous.values[:, idx] if previous is not None else np.zeros_like(now)
        x_prof = np.concatenate([now.T, tendency.T, current.below_ground[None, :].astype(float)], axis=0)
        valid = current.time + datetime.timedelta(hours=float(lead_hours))
        doy = 2 * np.pi * valid.timetuple().tm_yday / 365.25
        hv = 2 * np.pi * (valid.hour + valid.minute / 60) / 24
        hi = 2 * np.pi * current.time.hour / 24
        x_scalar = np.array([lead_hours / 24.0, np.sin(doy), np.cos(doy), np.sin(hv), np.cos(hv), np.sin(hi),
                             np.cos(hi), (current.psfc - 850.0) / 150.0, float(previous is not None)])
        return x_prof.astype(np.float32), x_scalar.astype(np.float32)

    def target_profile(self, profiles: dict, valid: datetime.datetime, allow_interpolated: bool,
                       max_gap_hours: float = 12.0):
        """Exact sounding at the valid time, or a time-interpolation between the bracketing ones."""
        if valid in profiles:
            return profiles[valid], False
        if not allow_interpolated:
            return None, False
        hour = datetime.timedelta(hours=1)
        ta = next((valid - k * hour for k in range(1, int(max_gap_hours) + 1) if valid - k * hour in profiles), None)
        tb = next((valid + k * hour for k in range(1, int(max_gap_hours) + 1) if valid + k * hour in profiles), None)
        if ta is None or tb is None or (tb - ta) > datetime.timedelta(hours=max_gap_hours):
            return None, False
        weight = (valid - ta).total_seconds() / (tb - ta).total_seconds()
        return self.blend(profiles[ta], profiles[tb], weight, valid), True

    # ------------------------------------------------------------------ datasets
    def build_training_set(self, soundings, component: str, leads=DEFAULT_TRAINING_LEADS,
                           allow_interpolated: bool = True) -> dict:
        if component not in TARGET_VARS:
            raise ValueError(f"unknown component '{component}'")
        t_idx = [PROFILE_VARS.index(v) for v in TARGET_VARS[component]]
        profiles = self.profiles_from(soundings)
        xp, xs, y, base, mask, init_times, lead_list, interp_flags = [], [], [], [], [], [], [], []
        for t0 in sorted(profiles):
            current, previous = profiles[t0], profiles.get(t0 - self.history_step)
            for lead in leads:
                target, interpolated = self.target_profile(profiles, t0 + datetime.timedelta(hours=lead),
                                                           allow_interpolated)
                if target is None:
                    continue
                a, b = self.features(current, previous, lead)
                xp.append(a)
                xs.append(b)
                base.append(current.values[:, t_idx].T)
                y.append((target.values[:, t_idx] - current.values[:, t_idx]).T)
                mask.append(~(target.below_ground | current.below_ground))
                init_times.append(t0)
                lead_list.append(lead)
                interp_flags.append(interpolated)
        if not xp:
            raise ValueError("no training pairs could be built - fetch a longer, continuous date range")
        interp = np.array(interp_flags)
        return {"x_prof": np.stack(xp), "x_scalar": np.stack(xs), "y_delta": np.stack(y).astype(np.float32),
                "base": np.stack(base).astype(np.float32), "mask": np.stack(mask),
                "init_times": init_times, "leads": np.array(lead_list), "interpolated": interp,
                "n_interpolated": int(interp.sum()), "n_native": int((~interp).sum()),
                "n_soundings": len(profiles), "component": component}

    def build_inference_input(self, soundings, lead_hours: float) -> dict:
        """``soundings``: cleaned (time, df) pairs, any order. Uses the newest as the init time."""
        profiles = self.profiles_from(soundings)
        if not profiles:
            raise ValueError("no usable sounding to initialise the forecast")
        t0 = max(profiles)
        current, previous = profiles[t0], profiles.get(t0 - self.history_step)
        x_prof, x_scalar = self.features(current, previous, lead_hours)
        return {"x_prof": x_prof[None], "x_scalar": x_scalar[None], "current": current, "previous": previous,
                "init_time": t0, "valid_time": t0 + datetime.timedelta(hours=float(lead_hours))}


def normalization_stats(x_prof: np.ndarray, y_delta: np.ndarray, mask: np.ndarray) -> dict:
    """Per-(channel,row) statistics. Delta mean is kept at 0 so an all-zero output == persistence."""
    x_mean, x_std = x_prof.mean(axis=0), x_prof.std(axis=0)
    m = mask[:, None, :].astype(float)
    y_std = np.sqrt((m * y_delta ** 2).sum(axis=0) / np.maximum(m.sum(axis=0), 1.0))
    return {"x_mean": x_mean.astype(np.float32), "x_std": np.where(x_std < 1e-6, 1.0, x_std).astype(np.float32),
            "y_std": np.where(y_std < 1e-6, 1.0, y_std).astype(np.float32)}
