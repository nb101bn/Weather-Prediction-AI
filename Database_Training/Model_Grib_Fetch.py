"""
Model_Grib_Fetch.py
===================
Downloads the "ground truth" for training: 3 km convection-allowing model output on pressure levels.

Two ways to get it (same ``fetch_frame(init, fhour, levels) -> ModelFrame`` interface):

* ``GribModelFetcher``     - GRIB2 files from the public NCEP archives. Only the needed messages are
  downloaded: the ``.idx`` inventory next to every file gives each message's byte offset, so we request
  byte ranges (HTTP Range) for e.g. HGT/UGRD/VGRD/TMP/RH at 7 levels (~35 messages) instead of the
  whole 100-400 MB file. Mirrors are tried in order (AWS open data -> Google -> NOMADS).
* ``AwipsEdexGridFetcher`` - the python-awips DataAccessLayer, taking grids from an EDEX server as they
  are transmitted (real time only; an EDEX server keeps roughly the last day or so).

Training rule (enforced by ``check_training_frame``): only OBSERVED initialisations (00Z and 12Z, the
radiosonde times) and only forecast hours 0-12 (the frames closest to the truth) are used.

Supported models: ``hrrr`` (HRRR CONUS, 3 km - the default; its grid *is* the model grid) and ``nam3k``
(NAM CONUS nest, 3 km). Add more in ``MODELS``.
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import re
import tempfile
from dataclasses import dataclass

import numpy as np
import requests

from Database_Training.Grid_Interpolation import STATE_VARS, ModelFrame
from Models.Model_Grid import EARTH_RADIUS_M, HRRR_CONUS, LambertGrid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, "Database_Training", "cache", "grib")

OBSERVED_INIT_HOURS = (0, 12)
TRAINING_FHOURS = tuple(range(0, 13))        # f00 .. f12

# wgrib2 inventory names (in the .idx files) and ecCodes short names for our state variables
IDX_NAMES = {"Z": "HGT", "U": "UGRD", "V": "VGRD", "T": "TMP", "RH": "RH"}
ECCODES_NAMES = {"gh": "Z", "u": "U", "v": "V", "t": "T", "r": "RH"}

MODELS = {
    "hrrr": {
        "label": "HRRR CONUS 3 km",
        "path": "hrrr.{date}/conus/hrrr.t{hh}z.wrfprsf{fh:02d}.grib2",
        "mirrors": {"aws": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
                    "google": "https://storage.googleapis.com/high-resolution-rapid-refresh",
                    "nomads": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod"},
        "grid": HRRR_CONUS,
    },
    "nam3k": {
        "label": "NAM CONUS nest 3 km",
        "path": "nam.{date}/nam.t{hh}z.conusnest.hiresf{fh:02d}.tm00.grib2",
        "mirrors": {"aws": "https://noaa-nam-pds.s3.amazonaws.com",
                    "nomads": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod"},
        "grid": None,                                  # read from the GRIB2 grid definition
    },
}


class DataUnavailable(RuntimeError):
    """The requested model run / forecast hour is not on any mirror."""


def check_training_frame(init_time: datetime.datetime, fhour: int) -> None:
    """Training uses observed initialisations (00Z/12Z) and forecast hours 0-12 only."""
    if init_time.hour not in OBSERVED_INIT_HOURS or init_time.minute or init_time.second:
        raise ValueError(f"{init_time:%Y-%m-%d %H:%MZ}: only the 00Z and 12Z (observed) initialisations are used "
                         "for training")
    if int(fhour) not in TRAINING_FHOURS:
        raise ValueError(f"f{int(fhour):02d}: training frames are limited to f00-f12")


def observed_init_times(start: datetime.datetime, end: datetime.datetime):
    """Every 00Z and 12Z between start and end (inclusive)."""
    t = start.replace(minute=0, second=0, microsecond=0)
    while t.hour not in OBSERVED_INIT_HOURS:
        t += datetime.timedelta(hours=1)
    while t <= end:
        yield t
        t += datetime.timedelta(hours=12)


# ====================================================================== .idx inventories
@dataclass
class IdxRecord:
    number: int
    offset: int
    variable: str
    level: str
    forecast: str
    end: int | None = None                # last byte (inclusive); None = to the end of the file


def parse_idx(text: str) -> list[IdxRecord]:
    """Parse a wgrib2-style inventory: ``n:offset:d=YYYYMMDDHH:VAR:LEVEL:FCST:...``."""
    records = []
    for line in text.splitlines():
        parts = line.strip().split(":")
        if len(parts) < 6 or not parts[0].strip().isdigit():
            continue
        records.append(IdxRecord(int(parts[0]), int(parts[1]), parts[3], parts[4], parts[5]))
    records.sort(key=lambda r: r.offset)
    for a, b in zip(records, records[1:]):
        a.end = b.offset - 1
    return records


def select_records(records: list[IdxRecord], variables, levels_hpa) -> list[IdxRecord]:
    wanted = {(IDX_NAMES[v], f"{int(p)} mb") for v in variables for p in levels_hpa}
    chosen = [r for r in records if (r.variable, r.level) in wanted]
    found = {(r.variable, r.level) for r in chosen}
    missing = wanted - found
    if missing:
        raise DataUnavailable(f"inventory lacks {sorted(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    # the same field can appear twice in some inventories; keep the first
    seen, unique = set(), []
    for r in chosen:
        if (r.variable, r.level) not in seen:
            seen.add((r.variable, r.level))
            unique.append(r)
    return unique


def byte_ranges(records: list[IdxRecord]) -> list[tuple[int, int | None]]:
    """Merge adjacent messages into as few HTTP Range requests as possible."""
    ranges = []
    for r in sorted(records, key=lambda r: r.offset):
        if ranges and ranges[-1][1] is not None and r.offset == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], r.end)
        else:
            ranges.append((r.offset, r.end))
    return ranges


# ====================================================================== GRIB2 decoding
def _grib_messages(path: str, backend: str = "auto"):
    """Yield (get(key), values()) accessors for every message, using pygrib or the eccodes bindings.

    Both wrap the same ecCodes library. ``auto`` prefers pygrib: the eccodes pip wheel bundles its own
    copy of PROJ, which clashes with cartopy/pyproj's copy in the same process (heap corruption at exit).
    Never import both backends in one process (two libeccodes copies crash).
    """
    if backend == "auto":
        backend = "pygrib" if importlib.util.find_spec("pygrib") is not None else "eccodes"
    if backend == "pygrib":
        import pygrib
        with pygrib.open(path) as grbs:
            for msg in grbs:
                yield (lambda key, m=msg: m[key]), (lambda m=msg: np.ma.filled(np.ma.asarray(m.values, dtype=np.float32),
                                                                              np.nan).ravel())
        return
    if backend != "eccodes":
        raise ValueError(f"unknown GRIB backend '{backend}' (auto, pygrib, eccodes)")
    import eccodes
    with open(path, "rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                def values(g=gid):
                    v = eccodes.codes_get_values(g).astype(np.float32)
                    if eccodes.codes_get(g, "bitmapPresent"):
                        v[v == eccodes.codes_get(g, "missingValue")] = np.nan
                    return v
                yield (lambda key, g=gid: eccodes.codes_get(g, key)), values
            finally:
                eccodes.codes_release(gid)


def decode_grib(path: str, levels, variables=STATE_VARS, backend: str = "auto") -> tuple[np.ndarray, LambertGrid, bool]:
    """Decode HGT/UGRD/VGRD/TMP/RH pressure-level messages -> ([V, L, ny, nx], grid, winds_grid_relative).

    Rows are returned south -> north (flipped if the file scans north -> south). Only Lambert
    conformal grids are supported (HRRR, NAM nest, RRFS CONUS all use one).
    """
    levels = [int(p) for p in levels]
    out, grid, grid_relative = None, None, True
    filled = np.zeros((len(variables), len(levels)), bool)
    for get, values in _grib_messages(path, backend):
        var = ECCODES_NAMES.get(get("shortName"))
        if var not in variables or get("typeOfLevel") != "isobaricInhPa":
            continue
        level = int(get("level"))
        if level not in levels:
            continue
        if get("gridType") != "lambert":
            raise ValueError(f"unsupported grid type {get('gridType')}")
        nx, ny = int(get("Nx")), int(get("Ny"))
        data = values().reshape(ny, nx)
        north_first = not int(get("jScansPositively"))
        if north_first:
            data = data[::-1]
        if grid is None:
            shape = int(get("shapeOfTheEarth"))
            radius = {0: 6367470.0, 6: EARTH_RADIUS_M}.get(shape, EARTH_RADIUS_M)
            if shape == 1:
                radius = float(get("radius"))
            lov = (float(get("LoVInDegrees")) + 180.0) % 360.0 - 180.0
            dx = float(get("DxInMetres"))
            grid = LambertGrid(nx=nx, ny=ny, dx=dx, lat_first=float(get("latitudeOfFirstGridPointInDegrees")),
                               lon_first=(float(get("longitudeOfFirstGridPointInDegrees")) + 180.0) % 360.0 - 180.0,
                               lov=lov, latin1=float(get("Latin1InDegrees")), latin2=float(get("Latin2InDegrees")),
                               radius=radius, name=os.path.basename(path))
            if north_first:                      # the first point was the NW corner: find the SW corner
                x0, y0 = grid.origin_xy
                lat_sw, lon_sw = grid.unproject(x0, y0 - (ny - 1) * dx)
                grid = LambertGrid(nx=nx, ny=ny, dx=dx, lat_first=float(lat_sw), lon_first=float(lon_sw), lov=lov,
                                   latin1=grid.latin1, latin2=grid.latin2, radius=radius, name=grid.name)
            out = np.full((len(variables), len(levels), ny, nx), np.nan, np.float32)
        if var in ("U", "V"):
            grid_relative = bool(int(get("uvRelativeToGrid")))
        out[variables.index(var), levels.index(level)] = data
        filled[variables.index(var), levels.index(level)] = True
    if out is None or not filled.all():
        missing = [(variables[i], levels[j]) for i, j in zip(*np.nonzero(~filled))]
        raise DataUnavailable(f"{os.path.basename(path)} is missing {missing[:6]}")
    return out, grid, grid_relative


# ====================================================================== HTTP GRIB fetcher
class GribModelFetcher:
    """Byte-range GRIB2 downloads from NOAA open-data mirrors (AWS, Google) or NOMADS."""

    def __init__(self, model: str = "hrrr", mirrors: list[str] | None = None, cache_dir: str | None = DEFAULT_CACHE,
                 keep_grib: bool = False, timeout: float = 60.0, session: requests.Session | None = None,
                 verbose: bool = True, grib_backend: str = "auto"):
        self.grib_backend = grib_backend
        if model not in MODELS:
            raise ValueError(f"unknown model '{model}' (choose from {sorted(MODELS)})")
        self.model, self.spec = model, MODELS[model]
        self.mirrors = mirrors or list(self.spec["mirrors"])
        unknown = [m for m in self.mirrors if m not in self.spec["mirrors"] and not m.startswith("http")]
        if unknown:
            raise ValueError(f"unknown mirror(s) {unknown} for {model}; use one of {list(self.spec['mirrors'])} or a URL")
        self.cache_dir = os.path.join(cache_dir, model) if cache_dir else None
        self.keep_grib, self.timeout, self.verbose = keep_grib, timeout, verbose
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "Weather-Prediction-AI/upper-air-cam (research)")
        self.source = model
        self.last_error: str | None = None

    def url(self, init_time: datetime.datetime, fhour: int, mirror: str) -> str:
        base = self.spec["mirrors"].get(mirror, mirror).rstrip("/")
        return f"{base}/" + self.spec["path"].format(date=f"{init_time:%Y%m%d}", hh=f"{init_time:%H}", fh=int(fhour))

    def fetch_frame(self, init_time: datetime.datetime, fhour: int, levels, grid: LambertGrid | None = None) -> ModelFrame:
        path, url = self.download(init_time, fhour, levels)
        try:
            fields, src_grid, grid_relative = decode_grib(path, levels, backend=self.grib_backend)
        finally:
            if not self.keep_grib and self.cache_dir is None and os.path.exists(path):
                os.remove(path)
        return ModelFrame(fields, tuple(levels), src_grid, init_time, int(fhour), self.model, grid_relative,
                          meta={"url": url, "label": self.spec["label"]})

    def download(self, init_time: datetime.datetime, fhour: int, levels) -> tuple[str, str]:
        """Download only the needed GRIB messages; returns (local path, url used)."""
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{init_time:%Y%m%d%H}",
                                      f"f{int(fhour):02d}_{'-'.join(str(int(p)) for p in levels)}.grib2")
            if os.path.exists(cache_path):
                return cache_path, "cache"
        errors = []
        for mirror in self.mirrors:
            url = self.url(init_time, fhour, mirror)
            try:
                idx = self.session.get(url + ".idx", timeout=self.timeout)
                if idx.status_code == 404:
                    errors.append(f"{mirror}: not found")
                    continue
                idx.raise_for_status()
                records = select_records(parse_idx(idx.text), STATE_VARS, levels)
                chunks = []
                for start, end in byte_ranges(records):
                    rng = f"bytes={start}-{'' if end is None else end}"
                    resp = self.session.get(url, headers={"Range": rng}, timeout=self.timeout)
                    resp.raise_for_status()
                    if resp.status_code != 206 and (start > 0 or end is not None):
                        raise DataUnavailable(f"{mirror} ignored the Range request (HTTP {resp.status_code})")
                    chunks.append(resp.content)
                data = b"".join(chunks)
                if not data.startswith(b"GRIB"):
                    raise DataUnavailable(f"{mirror} returned something that is not GRIB2")
                target = cache_path or tempfile.NamedTemporaryFile(suffix=".grib2", delete=False).name
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(data)
                if self.verbose:
                    print(f"[{self.model}] {init_time:%Y-%m-%d %HZ} f{int(fhour):02d}: {len(records)} messages, "
                          f"{len(data) / 1e6:.1f} MB from {mirror}")
                return target, url
            except (requests.exceptions.RequestException, DataUnavailable) as exc:
                errors.append(f"{mirror}: {exc}")
        self.last_error = "; ".join(errors)
        raise DataUnavailable(f"{self.model} {init_time:%Y-%m-%d %HZ} f{int(fhour):02d} unavailable ({self.last_error})")

    def discard(self, init_time: datetime.datetime) -> None:
        """Delete the cached GRIB subsets of one run (after they were converted to training arrays)."""
        if self.keep_grib or not self.cache_dir:
            return
        folder = os.path.join(self.cache_dir, f"{init_time:%Y%m%d%H}")
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.endswith(".grib2"):
                    os.remove(os.path.join(folder, name))
            try:
                os.rmdir(folder)
            except OSError:
                pass


# ====================================================================== AWIPS / EDEX fetcher
class AwipsEdexGridFetcher:
    """Model grids from an AWIPS EDEX server through python-awips (``pip install python-awips``).

    Real-time only (EDEX keeps a short rolling window), so this is for benchmarking the model against
    the runs as they arrive, or for building a training set day by day. Winds on EDEX grids are treated
    as grid-relative when ``winds_grid_relative`` is True; verify this for your server/model before
    relying on it (set False if your EDEX rotates winds to earth-relative).
    """

    PARAMS = {"Z": "GH", "U": "uW", "V": "vW", "T": "T", "RH": "RH"}

    NATIVE_GRIDS = {"HRRR": HRRR_CONUS}

    def __init__(self, host: str = "edex-cloud.unidata.ucar.edu", model: str = "HRRR",
                 winds_grid_relative: bool = True, verbose: bool = True):
        self.host, self.model, self.winds_grid_relative, self.verbose = host, model, winds_grid_relative, verbose
        self.source = f"awips:{model}"
        self.native_grid = self.NATIVE_GRIDS.get(model.upper())

    def _dal(self):
        try:
            from awips.dataaccess import DataAccessLayer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("AWIPS access needs `pip install python-awips shapely`") from exc
        DataAccessLayer.changeEDEXHost(self.host)
        return DataAccessLayer

    def _request(self, dal, param: str, levels):
        req = dal.newDataRequest("grid")
        req.setLocationNames(self.model)
        req.setParameters(param)
        req.setLevels(*[f"{float(p):.1f}MB" for p in levels])
        return req

    def available_cycles(self) -> list[datetime.datetime]:
        dal = self._dal()
        cycles = dal.getAvailableTimes(self._request(dal, self.PARAMS["Z"], [500]), True)
        return sorted(_datatime_ref(t) for t in cycles)

    def fetch_frame(self, init_time: datetime.datetime, fhour: int, levels, grid: LambertGrid | None = None) -> ModelFrame:
        dal = self._dal()
        levels = [int(p) for p in levels]
        fields, lats, lons = None, None, None
        for var in STATE_VARS:
            req = self._request(dal, self.PARAMS[var], levels)
            times = [t for t in dal.getAvailableTimes(req)
                     if _datatime_ref(t) == init_time and int(t.getFcstTime() or 0) == int(fhour) * 3600]
            if not times:
                raise DataUnavailable(f"EDEX {self.host}: no {self.model} {var} for {init_time:%Y-%m-%d %HZ} f{fhour:02d}")
            for rec in dal.getGridData(req, times[:1]):
                level = int(round(float(re.match(r"[\d.]+", str(rec.getLevel())).group())))
                if level not in levels:
                    continue
                data = np.asarray(rec.getRawData(), dtype=np.float32)
                data = _convert_units(var, data, str(rec.getUnit() or ""))
                if fields is None:
                    a, b = (np.asarray(x, dtype=float) for x in rec.getLatLonCoords())
                    lons, lats = (a, b) if np.nanmax(np.abs(a)) > 90 else (b, a)
                    fields = np.full((len(STATE_VARS), len(levels), *data.shape), np.nan, np.float32)
                fields[STATE_VARS.index(var), levels.index(level)] = data
        if fields is None or np.isnan(fields).all(axis=(2, 3)).any():
            raise DataUnavailable(f"EDEX {self.host}: incomplete {self.model} frame")
        if lats[0, 0] > lats[-1, 0]:                    # make row 0 the southern edge
            fields, lats, lons = fields[..., ::-1, :], lats[::-1], lons[::-1]
        native = self.native_grid
        if native is not None and fields.shape[-2:] == (native.ny, native.nx):
            # the model's own Lambert grid: exact windowing and (if grid-relative) no rotation needed
            return ModelFrame(np.ascontiguousarray(fields), tuple(levels), native, init_time, int(fhour), self.source,
                              grid_relative=self.winds_grid_relative, meta={"host": self.host})
        if self.winds_grid_relative:
            raise DataUnavailable(f"{self.model} grid from EDEX has an unknown projection, so grid-relative winds "
                                  "cannot be rotated; pass winds_grid_relative=False if the server sends earth-relative winds")
        return ModelFrame(np.ascontiguousarray(fields), tuple(levels), None, init_time, int(fhour), self.source,
                          grid_relative=False, lats=lats, lons=lons, meta={"host": self.host})


def _datatime_ref(t) -> datetime.datetime:
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=int(t.getRefTime().getTime()))


def _convert_units(var: str, data: np.ndarray, unit: str) -> np.ndarray:
    u = unit.strip().lower()
    if var == "T" and u in ("c", "°c", "degc", "celsius"):
        return data + 273.15
    if var == "Z" and u in ("dam", "dam.gpm"):
        return data * 10.0
    if var in ("U", "V") and u in ("kts", "kt", "knots"):
        return data * 0.514444
    if var == "RH" and u in ("", "fraction") and np.nanmax(data) <= 1.5:
        return data * 100.0
    return data


def get_truth_fetcher(source: str, **kwargs):
    """'hrrr' / 'nam3k' (GRIB archives), 'awips' / 'awips:HRRR' (EDEX), or 'synthetic' (offline demo)."""
    source = (source or "hrrr").lower()
    if source in MODELS:
        return GribModelFetcher(source, **kwargs)
    if source.startswith("awips"):
        model = source.split(":", 1)[1].upper() if ":" in source else "HRRR"
        return AwipsEdexGridFetcher(model=model, **{k: v for k, v in kwargs.items() if k in ("host", "verbose",
                                                                                              "winds_grid_relative")})
    if source == "synthetic":
        from Database_Training.Synthetic_Atmosphere import SyntheticAtmosphere
        return SyntheticAtmosphere()
    raise ValueError(f"unknown truth source '{source}'")
