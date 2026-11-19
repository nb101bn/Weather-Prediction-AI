"""Regenerates tests/fixtures/*.grib2 (+ .idx): small HRRR-grid windows of the synthetic atmosphere written
as real GRIB2 with the ecCodes Python bindings.  Run from the project root:  python tests/make_grib_fixtures.py
(Needs `pip install eccodes`. Only used to create the committed fixture files.)"""
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from grib_fixtures import build_grib_file  # noqa: E402
from Database_Training.Synthetic_Atmosphere import SyntheticAtmosphere  # noqa: E402
from Models.Model_Grid import HRRR_CONUS  # noqa: E402

LEVELS = (500, 400, 300, 250, 200, 150, 100)
INIT = datetime.datetime(2024, 5, 12, 12)
WINDOW = (600, 280, 40, 32)                 # i0, j0, nx, ny of the HRRR grid
OUT = os.path.join(ROOT, "tests", "fixtures")


def main():
    os.makedirs(OUT, exist_ok=True)
    src = HRRR_CONUS.window(*WINDOW)
    fields = SyntheticAtmosphere().fetch_frame(INIT, 1, LEVELS, src).fields
    for name, kw in {"hrrr_window_f01": {}, "hrrr_window_f01_northfirst": {"north_first": True, "grid_relative": False}}.items():
        data, idx = build_grib_file(src, fields, LEVELS, INIT, 1, **kw)
        with open(os.path.join(OUT, name + ".grib2"), "wb") as fh:
            fh.write(data)
        with open(os.path.join(OUT, name + ".grib2.idx"), "w") as fh:
            fh.write(idx)
        print(name, len(data), "bytes")


if __name__ == "__main__":
    main()
