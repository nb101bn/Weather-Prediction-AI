"""
Data-Fetch.py
=============
Fetches upper-air soundings (the data behind a Skew-T log-p diagram) from the
University of Wyoming Upper Air archive (WUA).

Every fetcher returns a pandas DataFrame with one row per reported level:

    PRES (hPa) | HGHT (m) | TEMP (C) | DWPT (C) | RELH (%) | MIXR (g/kg) | DRCT (deg) | SKNT (knot)

and metadata in ``df.attrs`` (station, time, source, url).

Two sources share the same interface (``fetch``, ``fetch_range``, ``fetch_latest``):

* ``WyomingUpperAir``  - real data from weather.uwyo.edu (tries the current
  ``wsgi/sounding`` CSV interface first, then the legacy ``cgi-bin/sounding``
  TEXT:LIST page). Results are cached to disk so training runs don't re-download.
* ``SyntheticUpperAir`` - physically-plausible *fake* soundings for offline
  development/testing. Anything produced from it is labelled ``source='synthetic'``.
"""
from __future__ import annotations

import datetime
import io
import math
import os
import re
import time as _time
import zlib

import numpy as np
import pandas as pd
import requests

COLUMNS = ["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT"]
MS_TO_KNOTS = 1.943844
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def nominal_sounding_times(start: datetime.datetime, end: datetime.datetime, step_hours: int = 12):
    """All synoptic launch times (00Z/12Z by default) between start and end, inclusive."""
    current = start.replace(minute=0, second=0, microsecond=0)
    while current.hour % step_hours:
        current += datetime.timedelta(hours=1)
    while current <= end:
        yield current
        current += datetime.timedelta(hours=step_hours)


class UpstreamUnavailable(RuntimeError):
    """weather.uwyo.edu could not be reached at all (DNS / connection / timeout) - stop instead of retrying."""


def _empty_to_none(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty or df["TEMP"].notna().sum() == 0:
        return None
    return df


class WyomingUpperAir:
    """Class for pulling soundings from the University of Wyoming Upper Air archive."""

    BASE_URL = "https://weather.uwyo.edu"
    BUSY_MARKERS = ("too busy", "try again later")

    def __init__(self, cache_dir: str | None = DEFAULT_CACHE_DIR, timeout: float = 30.0,
                 pause_seconds: float = 1.5, max_retries: int = 3, session: requests.Session | None = None,
                 wsgi_src: str | None = None):
        self.cache_dir = os.path.join(cache_dir, "wyoming") if cache_dir else None
        self.timeout = timeout
        self.pause_seconds = pause_seconds  # be polite: UWyo throttles rapid-fire requests
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "Weather-Prediction-AI/point-forecast (research)")
        self.wsgi_src = wsgi_src
        self._last_request = 0.0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ public API
    def fetch(self, station: str, time: datetime.datetime, use_cache: bool = True) -> pd.DataFrame | None:
        """Fetch one sounding. Returns None if the archive has no sounding for that station/time."""
        station = str(station).strip().upper()
        cached = self._read_cache(station, time) if use_cache else None
        if cached is not None:
            return cached
        self.last_error = None
        df, network_failures = None, 0
        for getter in (self._fetch_wsgi_csv, self._fetch_legacy_text):
            try:
                df = getter(station, time)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                network_failures += 1
                self.last_error = f"{getter.__name__}: {exc}"
                df = None
            except (requests.exceptions.RequestException, ValueError) as exc:  # HTTP errors, unparseable pages
                self.last_error = f"{getter.__name__}: {exc}"
                df = None
            if df is not None:
                break
        if df is None and network_failures == 2:
            raise UpstreamUnavailable(f"{self.BASE_URL} unreachable ({self.last_error})")
        if df is None:
            return None
        df.attrs.update({"station": station, "time": time.isoformat(), "source": "wyoming"})
        self._write_cache(station, time, df)
        return df

    def fetch_range(self, station: str, start: datetime.datetime, end: datetime.datetime,
                    step_hours: int = 12, verbose: bool = True) -> list[tuple[datetime.datetime, pd.DataFrame]]:
        """Fetch every sounding between start and end. Missing soundings are skipped."""
        results = []
        for when in nominal_sounding_times(start, end, step_hours):
            df = self.fetch(station, when)
            if verbose:
                print(f"[WUA] {station} {when:%Y-%m-%d %HZ}: {'ok' if df is not None else 'missing'}")
            if df is not None:
                results.append((when, df))
        return results

    def fetch_latest(self, station: str, now: datetime.datetime | None = None, count: int = 2,
                     max_lookback_hours: int = 72) -> list[tuple[datetime.datetime, pd.DataFrame]]:
        """Most recent ``count`` soundings (newest first), looking back at most ``max_lookback_hours``."""
        now = now or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        when = now.replace(minute=0, second=0, microsecond=0)
        when -= datetime.timedelta(hours=when.hour % 12)
        found = []
        while len(found) < count and (now - when) <= datetime.timedelta(hours=max_lookback_hours):
            df = self.fetch(station, when)
            if df is not None:
                found.append((when, df))
            when -= datetime.timedelta(hours=12)
        return found

    # ------------------------------------------------------------------ HTTP helpers
    def _get(self, url: str, params: dict) -> str:
        for attempt in range(1, self.max_retries + 1):
            wait = self.pause_seconds - (_time.monotonic() - self._last_request)
            if wait > 0:
                _time.sleep(wait)
            self._last_request = _time.monotonic()
            response = self.session.get(url, params=params, timeout=self.timeout)
            busy = response.status_code == 503 or any(m in response.text.lower() for m in self.BUSY_MARKERS)
            if busy and attempt < self.max_retries:
                _time.sleep(self.pause_seconds * 2 ** attempt)
                continue
            if response.status_code == 404:
                return ""
            response.raise_for_status()
            return response.text
        return ""

    def _fetch_wsgi_csv(self, station: str, time: datetime.datetime) -> pd.DataFrame | None:
        params = {"datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "id": station, "type": "TEXT:CSV"}
        if self.wsgi_src:
            params["src"] = self.wsgi_src
        return _empty_to_none(self.parse_wsgi_csv(self._get(f"{self.BASE_URL}/wsgi/sounding", params)))

    def _fetch_legacy_text(self, station: str, time: datetime.datetime) -> pd.DataFrame | None:
        params = {"region": "naconf", "TYPE": "TEXT:LIST", "YEAR": time.strftime("%Y"),
                  "MONTH": time.strftime("%m"), "FROM": time.strftime("%d%H"), "TO": time.strftime("%d%H"),
                  "STNM": station}
        return _empty_to_none(self.parse_legacy_text(self._get(f"{self.BASE_URL}/cgi-bin/sounding", params)))

    # ------------------------------------------------------------------ parsers
    @staticmethod
    def parse_legacy_text(page: str) -> pd.DataFrame | None:
        """Parse the fixed-width (7 chars/column) table inside the first <PRE> block of a TEXT:LIST page."""
        match = re.search(r"<pre>(.*?)</pre>", page or "", flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        lines = match.group(1).splitlines()
        header_idx = next((i for i, line in enumerate(lines) if "PRES" in line and "HGHT" in line), None)
        if header_idx is None:
            return None
        names = lines[header_idx].split()
        rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip() or line.strip().startswith("-") or not re.match(r"^\s*\d", line):
                continue  # units row / separators
            fields = [line[i * 7:(i + 1) * 7].strip() for i in range(len(names))]
            rows.append([float(f) if f else np.nan for f in fields])
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=names)
        return df.reindex(columns=COLUMNS)

    @staticmethod
    def parse_wsgi_csv(text: str) -> pd.DataFrame | None:
        """Parse the CSV served by the newer wsgi interface. Column names are matched by keyword
        (e.g. 'pressure_hPa', 'dew point temperature_C', 'wind speed_m/s') so minor header changes survive."""
        if not text or "," not in text.splitlines()[0]:
            return None
        raw = pd.read_csv(io.StringIO(text))
        lower = {c: c.lower().strip() for c in raw.columns}

        def pick(*must, avoid=()):
            for col, name in lower.items():
                if all(m in name for m in must) and not any(a in name for a in avoid):
                    return col
            return None

        mapping = {
            "PRES": pick("pressure"),
            "HGHT": pick("height"),
            "TEMP": pick("temperature", avoid=("dew", "potential", "ice", "virtual")),
            "DWPT": pick("dew"),
            "RELH": pick("relative humidity") or pick("humidity", avoid=("ice",)),
            "MIXR": pick("mixing"),
            "DRCT": pick("direction"),
            "SKNT": pick("speed"),
        }
        if mapping["PRES"] is None or mapping["TEMP"] is None:
            return None
        df = pd.DataFrame({k: pd.to_numeric(raw[v], errors="coerce") if v else np.nan for k, v in mapping.items()})
        speed_col = lower.get(mapping["SKNT"], "") if mapping["SKNT"] else ""
        if "m/s" in speed_col or "m s" in speed_col:
            df["SKNT"] = df["SKNT"] * MS_TO_KNOTS
        return df[COLUMNS]

    # ------------------------------------------------------------------ cache
    def _cache_path(self, station: str, time: datetime.datetime) -> str | None:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, station, f"{time:%Y%m%d%H}.csv")

    def _read_cache(self, station: str, time: datetime.datetime) -> pd.DataFrame | None:
        path = self._cache_path(station, time)
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            df.attrs.update({"station": station, "time": time.isoformat(), "source": "wyoming", "cached": True})
            return df
        return None

    def _write_cache(self, station: str, time: datetime.datetime, df: pd.DataFrame) -> None:
        path = self._cache_path(station, time)
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        # Recent launches can still be incomplete on the UWyo site - only cache soundings that are 6+ hours old.
        if path and now - time >= datetime.timedelta(hours=6):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)


class SyntheticUpperAir:
    """Deterministic fake soundings with the same interface/columns as ``WyomingUpperAir``.

    Used for offline development, tests and the HUD's demo mode. The atmosphere has a seasonal
    cycle, a diurnal surface cycle and slowly-evolving 'weather' anomalies, so there is real
    (but artificial) signal for the network to learn. NEVER use it for an actual forecast.
    """

    LEVELS = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
                       450, 400, 350, 300, 250, 200, 150, 100], dtype=float)

    def __init__(self, seed: int = 7, surface_pressure: float = 930.0):
        self.seed = seed
        self.surface_pressure = surface_pressure

    def _anomaly(self, station: str, hours: float, period: float, salt: int) -> float:
        """Smooth pseudo-random signal: sum of a few sinusoids with station-dependent phases."""
        rng = np.random.default_rng(zlib.crc32(f"{station}|{salt}|{self.seed}".encode()))
        phases, scales = rng.uniform(0, 2 * np.pi, 3), rng.uniform(0.5, 1.0, 3)
        periods = period * np.array([1.0, 0.61, 0.37])
        return float(np.sum(scales * np.sin(2 * np.pi * hours / periods + phases)) / 1.5)

    def fetch(self, station: str, time: datetime.datetime, use_cache: bool = True) -> pd.DataFrame | None:
        station = str(station).strip().upper()
        hours = (time - datetime.datetime(2000, 1, 1)).total_seconds() / 3600.0
        doy = time.timetuple().tm_yday
        season = -math.cos(2 * math.pi * (doy - 15) / 365.25)          # -1 winter ... +1 summer
        diurnal = math.cos(2 * math.pi * (time.hour - 21) / 24.0)      # warmest ~21Z (afternoon, CONUS)
        synoptic = self._anomaly(station, hours, 96.0, 1)               # ~4-day waves
        moist = self._anomaly(station, hours, 60.0, 2)
        jet = self._anomaly(station, hours, 120.0, 3)
        turn = self._anomaly(station, hours, 80.0, 4)

        psfc = self.surface_pressure + 8.0 * synoptic
        pres = np.concatenate([[psfc], self.LEVELS[self.LEVELS < psfc]])
        t_sfc = 12.0 + 13.0 * season + 6.0 * synoptic
        height = 44330.8 * (1 - (pres / 1013.25) ** 0.190263)
        h_sfc = height[0]
        boundary_layer = np.exp(-(height - h_sfc) / 1200.0)            # diurnal heating stays near the ground
        temp = t_sfc - 6.5 * (height - h_sfc) / 1000.0 + 6.0 * diurnal * boundary_layer
        temp = np.maximum(temp, -56.5 + 4.0 * season)                   # crude tropopause
        depression = np.clip(3.0 + 8.0 * (1 - moist) / 2 + 0.0022 * (height - h_sfc) * (1.2 - moist), 0.5, 28)
        dwpt = temp - depression
        speed = np.clip(10 + 25 * (height / 10000.0) * (1.4 + jet) + 5 * synoptic, 2, 160)
        drct = (230 + 40 * turn + 25 * (height / 10000.0) + 20 * synoptic) % 360
        es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
        e = 6.112 * np.exp(17.67 * dwpt / (dwpt + 243.5))
        df = pd.DataFrame({"PRES": np.round(pres, 1), "HGHT": np.round(height), "TEMP": np.round(temp, 1),
                           "DWPT": np.round(dwpt, 1), "RELH": np.round(100 * e / es),
                           "MIXR": np.round(1000 * 0.622 * e / (pres - e), 2), "DRCT": np.round(drct),
                           "SKNT": np.round(speed)})
        df.attrs.update({"station": station, "time": time.isoformat(), "source": "synthetic"})
        return df

    def fetch_range(self, station, start, end, step_hours: int = 12, verbose: bool = False):
        return [(when, self.fetch(station, when)) for when in nominal_sounding_times(start, end, step_hours)]

    def fetch_latest(self, station, now=None, count: int = 2, max_lookback_hours: int = 72):
        now = now or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        when = now.replace(minute=0, second=0, microsecond=0)
        when -= datetime.timedelta(hours=when.hour % 12)
        return [(when - datetime.timedelta(hours=12 * i), self.fetch(station, when - datetime.timedelta(hours=12 * i)))
                for i in range(count)]


def get_fetcher(source: str = "wyoming", **kwargs):
    """Factory used by the model/API: 'wyoming' (real) or 'synthetic' (demo/testing)."""
    source = (source or "wyoming").lower()
    if source == "wyoming":
        return WyomingUpperAir(**kwargs)
    if source == "synthetic":
        return SyntheticUpperAir(**kwargs)
    raise ValueError(f"Unknown data source '{source}' (expected 'wyoming' or 'synthetic')")
