"""
Upper_Air_Obs.py
================
Real upper-air OBSERVATIONS for the map model: radiosondes (RAOBs) launched at 00Z and 12Z.

Pipeline for one initialisation time:
    WyomingUpperAir (Data-Fetch.py, cached + throttled)  ->  SoundingCleaner (Data-Cleaning.py, QC)
    ->  ``sounding_to_levels`` (ln-p interpolation to the model's pressure levels, NO extrapolation)
    ->  observation table (station, lat, lon, level, Z, U_earth, V_earth, T, RH)
which Grid_Interpolation.grid_observations() then puts on the 3 km grid by nearest neighbour.

Only 00Z / 12Z are accepted: those are the "observed initialisations" the model trains on.
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

from Database_Training.Grid_Interpolation import OBS_COLUMNS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBSERVED_INIT_HOURS = (0, 12)
KNOT = 0.514444


def _load(name: str, filename: str):
    """The older Database_Training modules have hyphenated file names, so import them by path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT_ROOT, "Database_Training", filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# CONUS radiosonde sites: WMO id -> (ICAO, name, lat, lon, elevation m). Coordinates are rounded to
# ~0.01 deg (plenty for nearest-neighbour gridding on a 3 km grid from ~400 km station spacing).
RAOB_STATIONS = {
    "72201": ("KEY", "Key West FL", 24.55, -81.79, 2), "72202": ("MFL", "Miami FL", 25.75, -80.38, 4),
    "72206": ("JAX", "Jacksonville FL", 30.50, -81.70, 9), "72210": ("TBW", "Tampa Bay FL", 27.70, -82.40, 13),
    "72214": ("TLH", "Tallahassee FL", 30.45, -84.30, 53), "72208": ("CHS", "Charleston SC", 32.90, -80.03, 15),
    "72215": ("FFC", "Peachtree City GA", 33.36, -84.57, 245), "72230": ("BMX", "Birmingham AL", 33.17, -86.77, 178),
    "72233": ("LIX", "Slidell LA", 30.34, -89.83, 10), "72235": ("JAN", "Jackson MS", 32.32, -90.08, 91),
    "72240": ("LCH", "Lake Charles LA", 30.13, -93.22, 5), "72248": ("SHV", "Shreveport LA", 32.45, -93.84, 85),
    "72249": ("FWD", "Fort Worth TX", 32.83, -97.30, 196), "72250": ("BRO", "Brownsville TX", 25.92, -97.42, 7),
    "72251": ("CRP", "Corpus Christi TX", 27.77, -97.51, 14), "72261": ("DRT", "Del Rio TX", 29.37, -100.92, 313),
    "72265": ("MAF", "Midland TX", 31.95, -102.19, 873), "72274": ("TUS", "Tucson AZ", 32.23, -110.96, 751),
    "72293": ("NKX", "San Diego CA", 32.85, -117.12, 128), "72305": ("MHX", "Newport NC", 34.78, -76.88, 11),
    "72317": ("GSO", "Greensboro NC", 36.10, -79.94, 277), "72318": ("RNK", "Blacksburg VA", 37.20, -80.41, 648),
    "72327": ("OHX", "Nashville TN", 36.25, -86.57, 180), "72340": ("LZK", "Little Rock AR", 34.83, -92.26, 172),
    "72357": ("OUN", "Norman OK", 35.18, -97.44, 357), "72363": ("AMA", "Amarillo TX", 35.23, -101.71, 1095),
    "72364": ("EPZ", "Santa Teresa NM", 31.87, -106.70, 1252), "72365": ("ABQ", "Albuquerque NM", 35.04, -106.62, 1619),
    "72376": ("FGZ", "Flagstaff AZ", 35.23, -111.82, 2179), "72388": ("VEF", "Las Vegas NV", 36.05, -115.18, 697),
    "72393": ("VBG", "Vandenberg CA", 34.75, -120.57, 100), "72402": ("WAL", "Wallops Island VA", 37.93, -75.47, 13),
    "72403": ("IAD", "Sterling VA", 38.98, -77.49, 88), "72426": ("ILN", "Wilmington OH", 39.42, -83.82, 317),
    "72440": ("SGF", "Springfield MO", 37.23, -93.40, 394), "72451": ("DDC", "Dodge City KS", 37.76, -99.97, 790),
    "72456": ("TOP", "Topeka KS", 39.07, -95.63, 268), "72469": ("DNR", "Denver CO", 39.77, -104.87, 1611),
    "72476": ("GJT", "Grand Junction CO", 39.12, -108.53, 1475), "72489": ("REV", "Reno NV", 39.57, -119.80, 1516),
    "72493": ("OAK", "Oakland CA", 37.73, -122.22, 3), "72501": ("OKX", "Upton NY", 40.87, -72.86, 20),
    "72518": ("ALB", "Albany NY", 42.69, -73.83, 93), "72520": ("PIT", "Pittsburgh PA", 40.53, -80.22, 360),
    "72528": ("BUF", "Buffalo NY", 42.94, -78.72, 218), "72558": ("OAX", "Omaha NE", 41.32, -96.37, 350),
    "72562": ("LBF", "North Platte NE", 41.13, -100.68, 847), "72572": ("SLC", "Salt Lake City UT", 40.77, -111.95, 1289),
    "72582": ("LKN", "Elko NV", 40.87, -115.73, 1593), "72597": ("MFR", "Medford OR", 42.37, -122.88, 405),
    "72632": ("DTX", "White Lake MI", 42.70, -83.47, 329), "72634": ("APX", "Gaylord MI", 44.91, -84.72, 448),
    "72645": ("GRB", "Green Bay WI", 44.48, -88.13, 209), "72649": ("MPX", "Chanhassen MN", 44.85, -93.57, 287),
    "72659": ("ABR", "Aberdeen SD", 45.45, -98.41, 397), "72662": ("UNR", "Rapid City SD", 44.07, -103.21, 1029),
    "72672": ("RIW", "Riverton WY", 43.06, -108.48, 1688), "72681": ("BOI", "Boise ID", 43.57, -116.21, 874),
    "72694": ("SLE", "Salem OR", 44.92, -123.02, 61), "72712": ("CAR", "Caribou ME", 46.87, -68.02, 191),
    "72747": ("INL", "International Falls MN", 48.57, -93.38, 359), "72764": ("BIS", "Bismarck ND", 46.77, -100.75, 503),
    "72768": ("GGW", "Glasgow MT", 48.21, -106.63, 693), "72776": ("TFX", "Great Falls MT", 47.46, -111.38, 1130),
    "72786": ("OTX", "Spokane WA", 47.68, -117.63, 728), "72797": ("UIL", "Quillayute WA", 47.95, -124.55, 56),
    "74389": ("GYX", "Gray ME", 43.89, -70.26, 125), "74455": ("DVN", "Davenport IA", 41.61, -90.58, 229),
    "74494": ("CHH", "Chatham MA", 41.67, -69.97, 16), "74560": ("ILX", "Lincoln IL", 40.15, -89.34, 178),
}


def station_table() -> pd.DataFrame:
    return pd.DataFrame([{"station": k, "icao": v[0], "name": v[1], "lat": v[2], "lon": v[3], "elev_m": v[4]}
                         for k, v in RAOB_STATIONS.items()])


def stations_near(grid, margin_km: float = 600.0) -> list[str]:
    """Stations inside the grid or within ``margin_km`` of it (stations just outside still matter for
    nearest-neighbour values near the domain edge)."""
    out = []
    for wmo, (_, _, lat, lon, _) in RAOB_STATIONS.items():
        i, j = grid.index_of(lat, lon)
        m = margin_km * 1000.0 / grid.dx
        if -m <= i <= grid.nx - 1 + m and -m <= j <= grid.ny - 1 + m:
            out.append(wmo)
    return out


def check_observed_init(time: datetime.datetime) -> None:
    if time.hour not in OBSERVED_INIT_HOURS or time.minute or time.second:
        raise ValueError(f"{time:%Y-%m-%d %H:%MZ} is not an observed initialisation - only 00Z and 12Z are used "
                         "(radiosonde launch times)")


def _interp_ln_p(pres, values, targets):
    """ln(p) interpolation; NaN outside the observed pressure range (no extrapolation)."""
    ok = np.isfinite(pres) & np.isfinite(values)
    out = np.full(len(targets), np.nan)
    if ok.sum() < 2:
        return out
    x = np.log(pres[ok])[::-1]           # ascending ln p
    y = values[ok][::-1]
    t = np.log(np.asarray(targets, dtype=float))
    inside = (t >= x.min()) & (t <= x.max())
    out[inside] = np.interp(t[inside], x, y)
    return out


def sounding_to_levels(df: pd.DataFrame, levels) -> pd.DataFrame:
    """Cleaned sounding (Data-Cleaning output: PRES, HGHT, TEMP, DWPT, RELH, MIXR, DRCT, SKNT) ->
    one row per model level with Z (m), U/V earth-relative (m/s), T (K), RH (%). Levels below the
    surface or above the balloon burst are NaN."""
    df = df.sort_values("PRES", ascending=False)
    pres = df["PRES"].to_numpy(float)
    drct, sknt = df["DRCT"].to_numpy(float), df["SKNT"].to_numpy(float) * KNOT
    u = -sknt * np.sin(np.radians(drct))
    v = -sknt * np.cos(np.radians(drct))
    cols = {"Z": df["HGHT"].to_numpy(float), "T": df["TEMP"].to_numpy(float) + 273.15,
            "RH": df["RELH"].to_numpy(float), "U_earth": u, "V_earth": v}
    out = pd.DataFrame({"level": [float(p) for p in levels]})
    for name, values in cols.items():
        out[name] = _interp_ln_p(pres, values, levels)
    return out


class RaobObservations:
    """Radiosonde observations for one initialisation time, from the University of Wyoming archive."""

    source = "wyoming"

    def __init__(self, fetcher=None, cleaner=None, verbose: bool = True):
        if fetcher is None:
            fetcher = _load("data_fetch", "Data-Fetch.py").WyomingUpperAir()
        if cleaner is None:
            # keep soundings that reach at least 500 hPa; levels above the burst simply become NaN
            cleaner = _load("data_cleaning", "Data-Cleaning.py").SoundingCleaner(required_top_hpa=500.0)
        self.fetcher, self.cleaner, self.verbose = fetcher, cleaner, verbose

    def fetch(self, init_time: datetime.datetime, levels, stations=None) -> pd.DataFrame:
        check_observed_init(init_time)
        stations = stations or list(RAOB_STATIONS)
        rows, missing = [], []
        for wmo in stations:
            raw = self.fetcher.fetch(wmo, init_time)
            clean = self.cleaner.clean(raw) if raw is not None else None
            if clean is None:
                missing.append(wmo)
                continue
            lv = sounding_to_levels(clean, levels)
            _, _, lat, lon, _ = RAOB_STATIONS.get(wmo, (None, None, np.nan, np.nan, None))
            lv.insert(0, "lon", lon)
            lv.insert(0, "lat", lat)
            lv.insert(0, "station", wmo)
            rows.append(lv)
        if self.verbose:
            print(f"[RAOB] {init_time:%Y-%m-%d %HZ}: {len(rows)} soundings, {len(missing)} missing")
        if not rows:
            return pd.DataFrame(columns=OBS_COLUMNS)
        table = pd.concat(rows, ignore_index=True)[OBS_COLUMNS]
        table.attrs.update({"source": "wyoming", "init_time": init_time.isoformat(), "missing": missing})
        return table

    fetch_obs = fetch          # common interface with SyntheticAtmosphere.fetch_obs
