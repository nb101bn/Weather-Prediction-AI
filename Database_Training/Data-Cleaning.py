"""
Data-Cleaning.py
================
Quality control for soundings returned by Data-Fetch.py.

``SoundingCleaner`` runs an ordered list of *steps*. Each step is a function
``step(df, report) -> df | None`` (return None to reject the whole sounding).
To expand the cleaning later, write a new function and register it:

    cleaner = SoundingCleaner()
    cleaner.add_step(my_superadiabatic_check)           # appended to the end
    cleaner.add_step(my_first_check, position=0)        # or inserted anywhere

Every cleaned DataFrame carries ``df.attrs['cleaning']`` describing what was changed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

CORE_COLUMNS = ["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT"]

# Physically plausible ranges; values outside are set to NaN (not clipped).
VALID_RANGES = {
    "PRES": (1.0, 1100.0), "HGHT": (-500.0, 40000.0), "TEMP": (-100.0, 60.0), "DWPT": (-120.0, 40.0),
    "RELH": (0.0, 100.0), "MIXR": (0.0, 40.0), "DRCT": (0.0, 360.0), "SKNT": (0.0, 300.0),
}

Step = Callable[[pd.DataFrame, dict], "pd.DataFrame | None"]


def saturation_vapor_pressure(temp_c):
    """Bolton (1980) saturation vapour pressure in hPa."""
    return 6.112 * np.exp(17.67 * np.asarray(temp_c, dtype=float) / (np.asarray(temp_c, dtype=float) + 243.5))


class SoundingCleaner:
    """Expandable QC pipeline for a single sounding DataFrame."""

    def __init__(self, required_top_hpa: float = 300.0, min_levels: int = 5, steps: list[Step] | None = None):
        self.required_top_hpa = required_top_hpa
        self.min_levels = min_levels
        self.steps: list[Step] = steps if steps is not None else [
            self.coerce_numeric,
            self.range_check,
            self.drop_missing_core,
            self.sort_and_dedupe_pressure,
            self.enforce_dewpoint_le_temperature,
            self.fill_derived_moisture,
            self.check_vertical_coverage,
        ]

    def add_step(self, step: Step, position: int | None = None) -> None:
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)

    def clean(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        attrs = dict(df.attrs)
        report = {"input_rows": int(len(df)), "actions": [], "rejected": None}
        out = df.copy()
        for step in self.steps:
            out = step(out, report)
            if out is None:
                return None
        report["output_rows"] = int(len(out))
        out = out.reset_index(drop=True)
        out.attrs = {**attrs, "cleaning": report}
        return out

    def clean_many(self, soundings):
        """Clean a list of (time, df) pairs; drops rejected soundings and returns (kept, rejected_times)."""
        kept, rejected = [], []
        for when, df in soundings:
            cleaned = self.clean(df)
            (kept.append((when, cleaned)) if cleaned is not None else rejected.append(when))
        return kept, rejected

    # ------------------------------------------------------------------ default steps
    @staticmethod
    def coerce_numeric(df, report):
        for col in CORE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
        return df[CORE_COLUMNS]

    @staticmethod
    def range_check(df, report):
        for col, (lo, hi) in VALID_RANGES.items():
            bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            if bad.any():
                df.loc[bad, col] = np.nan
                report["actions"].append(f"{col}: {int(bad.sum())} out-of-range value(s) set to missing")
        return df

    @staticmethod
    def drop_missing_core(df, report):
        before = len(df)
        df = df[df["PRES"].notna() & df["TEMP"].notna()]
        if len(df) < before:
            report["actions"].append(f"dropped {before - len(df)} level(s) without pressure/temperature")
        return df

    @staticmethod
    def sort_and_dedupe_pressure(df, report):
        before = len(df)
        df = df.sort_values("PRES", ascending=False).drop_duplicates(subset="PRES", keep="first")
        if len(df) < before:
            report["actions"].append(f"removed {before - len(df)} duplicate pressure level(s)")
        return df

    @staticmethod
    def enforce_dewpoint_le_temperature(df, report):
        bad = df["DWPT"].notna() & (df["DWPT"] > df["TEMP"])
        if bad.any():
            df.loc[bad, "DWPT"] = df.loc[bad, "TEMP"]
            report["actions"].append(f"capped {int(bad.sum())} dewpoint(s) at the temperature")
        return df

    @staticmethod
    def fill_derived_moisture(df, report):
        """Recompute RH / mixing ratio from T, Td and p wherever they are missing."""
        e = saturation_vapor_pressure(df["DWPT"])
        rh = 100.0 * e / saturation_vapor_pressure(df["TEMP"])
        mixr = 1000.0 * 0.622 * e / (df["PRES"] - e)
        for col, derived in (("RELH", rh), ("MIXR", mixr)):
            fill = df[col].isna() & pd.Series(derived, index=df.index).notna()
            if fill.any():
                df.loc[fill, col] = np.round(pd.Series(derived, index=df.index)[fill], 2)
                report["actions"].append(f"{col}: derived {int(fill.sum())} missing value(s) from T/Td")
        return df

    def check_vertical_coverage(self, df, report):
        if len(df) < self.min_levels:
            report["rejected"] = f"only {len(df)} usable level(s)"
            return None
        if df["PRES"].min() > self.required_top_hpa:
            report["rejected"] = f"sounding terminated at {df['PRES'].min():.0f} hPa (needs {self.required_top_hpa:.0f})"
            return None
        if df["DWPT"].notna().sum() < 2:
            report["rejected"] = "no usable dewpoint data"
            return None
        return df
