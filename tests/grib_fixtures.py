"""Helpers that write small, real GRIB2 files (Lambert conformal, HRRR-style) with ecCodes, plus the
matching wgrib2-style .idx inventory - so the GRIB download/decoding path can be tested offline."""
import numpy as np

PARAMS = {"Z": (3, 5, "HGT"), "U": (2, 2, "UGRD"), "V": (2, 3, "VGRD"), "T": (0, 0, "TMP"), "RH": (1, 1, "RH")}


def lambert_message(grid, var, level_hpa, values, init, fhour, grid_relative=True, north_first=False):
    import eccodes
    gid = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        eccodes.codes_set(gid, "gridDefinitionTemplateNumber", 30)
        lat1, lon1 = grid.lat_first, grid.lon_first
        vals = np.asarray(values, dtype=float)
        if north_first:
            x0, y0 = grid.origin_xy
            lat1, lon1 = grid.unproject(x0, y0 + (grid.ny - 1) * grid.dx)
            vals = vals[::-1]
        for key, val in [("shapeOfTheEarth", 6), ("Nx", grid.nx), ("Ny", grid.ny),
                         ("latitudeOfFirstGridPointInDegrees", float(lat1)),
                         ("longitudeOfFirstGridPointInDegrees", float(lon1) % 360.0),
                         ("LaDInDegrees", grid.latin1), ("LoVInDegrees", grid.lov % 360.0),
                         ("Latin1InDegrees", grid.latin1), ("Latin2InDegrees", grid.latin2),
                         ("DxInMetres", grid.dx), ("DyInMetres", grid.dx),
                         ("jScansPositively", 0 if north_first else 1),
                         ("uvRelativeToGrid", 1 if grid_relative else 0)]:
            eccodes.codes_set(gid, key, val)
        cat, num, _ = PARAMS[var]
        eccodes.codes_set(gid, "discipline", 0)
        eccodes.codes_set(gid, "parameterCategory", cat)
        eccodes.codes_set(gid, "parameterNumber", num)
        eccodes.codes_set(gid, "typeOfFirstFixedSurface", 100)
        eccodes.codes_set(gid, "scaleFactorOfFirstFixedSurface", 0)
        eccodes.codes_set(gid, "scaledValueOfFirstFixedSurface", int(level_hpa * 100))
        eccodes.codes_set(gid, "dataDate", int(init.strftime("%Y%m%d")))
        eccodes.codes_set(gid, "dataTime", int(init.strftime("%H00")))
        eccodes.codes_set(gid, "forecastTime", int(fhour))
        eccodes.codes_set(gid, "bitsPerValue", 16)
        eccodes.codes_set_values(gid, vals.ravel())
        return eccodes.codes_get_message(gid)
    finally:
        eccodes.codes_release(gid)


def build_grib_file(grid, fields, levels, init, fhour, extra_first=True, **kw):
    """fields: [5, L, ny, nx] in STATE_VARS order. Returns (grib bytes, idx text). An unrelated message
    (surface temperature) is put first, as in real files, so byte offsets are non-trivial."""
    order = ["Z", "U", "V", "T", "RH"]
    msgs, idx = [], []
    offset = 0
    fc = "anl" if fhour == 0 else f"{fhour} hour fcst"
    if extra_first:
        m = lambert_message(grid, "T", 1000, fields[3, 0], init, fhour, **kw)
        msgs.append(m)
        idx.append(f"1:0:d={init:%Y%m%d%H}:TMP:2 m above ground:{fc}:")
        offset = len(m)
    n = len(idx)
    for var in ["T", "Z", "U", "V", "RH"]:               # deliberately not in STATE_VARS order
        for k, p in enumerate(levels):
            m = lambert_message(grid, var, p, fields[order.index(var), k], init, fhour, **kw)
            n += 1
            idx.append(f"{n}:{offset}:d={init:%Y%m%d%H}:{PARAMS[var][2]}:{p} mb:{fc}:")
            msgs.append(m)
            offset += len(m)
    return b"".join(msgs), "\n".join(idx) + "\n"


class RangeSession:
    """Pretends to be an HTTP server holding {url: bytes}; honours 'Range: bytes=a-b'."""

    class Resp:
        def __init__(self, status, content=b""):
            self.status_code, self.content = status, content
            self.text = content.decode("latin-1") if isinstance(content, bytes) else content

        def raise_for_status(self):
            import requests
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def __init__(self, files):
        self.files, self.headers, self.calls = files, {}, []

    def get(self, url, headers=None, timeout=None, params=None):
        self.calls.append((url, dict(headers or {})))
        if url not in self.files:
            return self.Resp(404)
        data = self.files[url]
        rng = (headers or {}).get("Range")
        if rng:
            a, b = rng.split("=")[1].split("-")
            end = int(b) + 1 if b else len(data)
            return self.Resp(206, data[int(a):end])
        return self.Resp(200, data)
