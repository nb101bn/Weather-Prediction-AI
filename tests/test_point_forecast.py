"""Run from the project root:  python -m unittest discover tests -v"""
import datetime
import importlib.util
import os
import sys
import tempfile
import unittest

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pfnn = load("point_forecast_nn", "Models/Point-Forecast-NN.py")
fetch, clean, inj = pfnn.data_fetch, pfnn.data_cleaning, pfnn.data_injestor
api = load("point_forecast_api", "API/api.py")

# (PRES, HGHT, TEMP, DWPT, RELH, MIXR, DRCT, SKNT) - None = blank field, as on the UWyo page
LEVELS = [(1000.0, 113, None, None, None, None, None, None), (925.0, 767, None, None, None, None, None, None),
          (918.0, 790, 15.4, 10.4, 72, 8.68, 160, 12), (900.0, 955, 14.8, 9.8, 72, 8.58, 170, 20),
          (850.0, 1430, 12.6, 6.6, 67, 7.18, 185, 31), (800.0, 1928, 9.0, 1.0, 57, 5.20, 200, 30),
          (700.0, 3020, 2.2, -6.8, 52, 3.40, 220, 25), (600.0, 4250, -5.1, -18.1, 35, 1.50, 235, 33),
          (500.0, 5700, -14.9, -30.9, 25, 0.56, 240, 45), (400.0, 7390, -27.3, -43.3, 20, 0.20, 245, 58),
          (300.0, 9440, -42.1, -55.1, 22, 0.07, 250, 72), (250.0, 10680, -50.5, -62.5, 23, 0.03, 250, 88),
          (200.0, 12110, -56.9, None, None, None, 255, 76), (150.0, 13900, -58.3, None, None, None, 260, 51)]


def legacy_page(levels=LEVELS):
    fmts = ("%7.1f", "%7d", "%7.1f", "%7.1f", "%7d", "%7.2f", "%7d", "%7d")
    rows = ["".join(" " * 7 if v is None else f % v for f, v in zip(fmts, lvl)) + " " * 21 for lvl in levels]
    dash = "-" * 77
    return ("<HTML><TITLE>University of Wyoming - Radiosonde Data</TITLE><BODY BGCOLOR=\"white\">\n"
            "<H2>72451 DDC Dodge City Observations at 12Z 12 May 2024</H2>\n<PRE>\n" + dash + "\n"
            "   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV\n"
            "    hPa     m      C      C      %    g/kg    deg   knot     K      K      K \n" + dash + "\n"
            + "\n".join(rows) + "\n</PRE><H3>Station information and sounding indices</H3><PRE>\n"
            "                         Station identifier: DDC\n</PRE></BODY></HTML>")


class StubSession:
    """Pretends to be weather.uwyo.edu: wsgi endpoint 404s, legacy endpoint serves the sample page."""

    def __init__(self, fail_network=False, available=None):
        self.headers, self.calls, self.fail, self.available = {}, [], fail_network, available

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.fail:
            raise requests.exceptions.ConnectionError("no route to host")
        resp = requests.models.Response()
        when = None
        if "cgi-bin" in url:
            when = (params["YEAR"], params["MONTH"], params["FROM"])
        ok = "cgi-bin" in url and (self.available is None or when in self.available)
        resp.status_code = 200 if ok else 404
        resp._content = (legacy_page() if ok else "Not found").encode()
        resp.encoding = "utf-8"
        return resp


class RouterTests(unittest.TestCase):
    def test_keywords_and_leads(self):
        r = pfnn.ComponentRouter()
        cases = {"Dewpoint +2 HRs": ("dewpoint", 2, None), "dew point in 6 hours at 850mb": ("dewpoint", 6, "850 hPa"),
                 "wind direction +12": ("wind_direction", 12, None), "850 hPa winds +2hrs": ("wind_speed", 2, "850 hPa"),
                 "RH +1 day": ("relative_humidity", 24, None), "td at the surface +4 hr": ("dewpoint", 4, "SFC"),
                 "500mb heights +24h": ("height", 24, "500 hPa"), "temps 3h": ("temperature", 3, None)}
        for text, (component, lead, level) in cases.items():
            parsed = r.parse(text)
            self.assertEqual((parsed["component"], parsed["lead_hours"], parsed["level"]), (component, lead, level), text)
        self.assertIsNone(r.parse("banana +2")["component"])
        self.assertEqual(r.parse_lead_hours("+2 HRs"), 2.0)
        self.assertEqual(r.parse_lead_hours(" 7 "), 7.0)


class FetchAndCleanTests(unittest.TestCase):
    def test_legacy_parser(self):
        df = fetch.WyomingUpperAir.parse_legacy_text(legacy_page())
        self.assertEqual(list(df.columns), fetch.COLUMNS)
        self.assertEqual(len(df), len(LEVELS))
        self.assertTrue(np.isnan(df.loc[0, "TEMP"]))                 # below-ground 1000 hPa row is blank
        self.assertAlmostEqual(df.loc[4, "DWPT"], 6.6)
        self.assertEqual(df.loc[8, "SKNT"], 45)
        self.assertTrue(np.isnan(df.loc[12, "DWPT"]))

    def test_wsgi_csv_parser_converts_ms_to_knots(self):
        text = ("time,longitude,latitude,pressure_hPa,geopotential height_m,temperature_C,dew point temperature_C,"
                "ice point temperature_C,relative humidity_%,humidity wrt ice_%,mixing ratio_g/kg,wind direction_degree,"
                "wind speed_m/s,potential temperature_K\n"
                "2024-05-12 11:02:00,-99.97,37.76,918.0,790,15.4,10.4,10.1,72,70,8.68,160,6.17,295.6\n"
                "2024-05-12 11:05:00,-99.97,37.76,850.0,1430,12.6,6.6,6.0,67,60,7.18,185,15.95,299.9\n")
        df = fetch.WyomingUpperAir.parse_wsgi_csv(text)
        self.assertEqual(list(df["TEMP"]), [15.4, 12.6])
        self.assertEqual(list(df["DWPT"]), [10.4, 6.6])               # not confused with ice point / potential T
        self.assertAlmostEqual(df.loc[0, "SKNT"], 12.0, places=1)

    def test_cleaner(self):
        df = fetch.WyomingUpperAir.parse_legacy_text(legacy_page())
        df.loc[5, "DWPT"] = 20.0                                       # supersaturated -> capped
        df.loc[6, "TEMP"] = 99.0                                       # impossible -> level dropped
        df = __import__("pandas").concat([df, df.iloc[[4]]])           # duplicate 850 hPa
        out = clean.SoundingCleaner(required_top_hpa=200).clean(df)
        self.assertTrue((out["DWPT"].dropna() <= out["TEMP"][out["DWPT"].notna()]).all())
        self.assertNotIn(700.0, list(out["PRES"]))
        self.assertEqual(out["PRES"].is_unique, True)
        self.assertTrue(out.attrs["cleaning"]["actions"])
        self.assertIsNone(clean.SoundingCleaner(required_top_hpa=100).clean(df))  # terminated too low

    def test_fetch_uses_fallback_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = StubSession()
            w = fetch.WyomingUpperAir(cache_dir=tmp, pause_seconds=0, session=session)
            df = w.fetch("72451", datetime.datetime(2024, 5, 12, 12))
            self.assertIsNotNone(df)
            self.assertEqual(session.calls[1][1]["FROM"], "1212")
            n = len(session.calls)
            w.fetch("72451", datetime.datetime(2024, 5, 12, 12))     # second call served from disk cache
            self.assertEqual(len(session.calls), n)

    def test_network_failure_fails_fast(self):
        w = fetch.WyomingUpperAir(cache_dir=None, pause_seconds=0, session=StubSession(fail_network=True))
        with self.assertRaises(fetch.UpstreamUnavailable):
            w.fetch_latest("72451")
        self.assertEqual(len(w.session.calls), 2)                      # stopped after the first time step


class InjestorAndModelTests(unittest.TestCase):
    def test_profile_interpolation_and_wind(self):
        df = clean.SoundingCleaner(required_top_hpa=200).clean(fetch.WyomingUpperAir.parse_legacy_text(legacy_page()))
        prof = inj.SoundingInjestor().to_profile(df, datetime.datetime(2024, 5, 12, 12))
        self.assertEqual(prof.psfc, 918.0)
        self.assertEqual(list(prof.below_ground[:4]), [False, True, True, False])
        self.assertAlmostEqual(prof.var("TEMP")[3], 12.6)              # 850 hPa exact level
        self.assertAlmostEqual(float(inj.wind_direction(prof.var("U")[3], prof.var("V")[3])), 185.0, places=3)
        u, v = inj.wind_components([0, 90, 270], [10, 10, 10])
        np.testing.assert_allclose(inj.wind_direction(u, v), [0, 90, 270], atol=1e-9)

    def test_training_pairs_native_vs_interpolated(self):
        src = fetch.SyntheticUpperAir()
        data = src.fetch_range("72451", datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 10))
        ds = inj.SoundingInjestor().build_training_set(data, "dewpoint", leads=(2, 12, 24))
        self.assertGreater(ds["n_native"], 0)
        self.assertGreater(ds["n_interpolated"], 0)
        self.assertEqual(ds["x_prof"].shape[1:], (15, 10))

    def test_untrained_network_is_persistence(self):
        net = pfnn.PointForecastNN(in_channels=15, n_rows=10, n_scalars=9, out_channels=1)
        with pfnn.torch.no_grad():
            out = net(pfnn.torch.randn(4, 15, 10), pfnn.torch.randn(4, 9))
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_change_is_zero_at_zero_lead(self):
        net = pfnn.PointForecastNN(in_channels=15, n_rows=10, n_scalars=9, out_channels=1)
        pfnn.torch.nn.init.normal_(net.head[-1].weight)                # make the network non-trivial
        xs = pfnn.torch.randn(3, 9)
        xs[:, 0] = pfnn.torch.tensor([0.0, 1 / 24, 12 / 24])           # lead 0 h, 1 h, 12 h
        with pfnn.torch.no_grad():
            out = net(pfnn.torch.randn(3, 15, 10), xs)
        self.assertEqual(float(out[0].abs().max()), 0.0)
        self.assertGreater(float(out[2].abs().max()), 0.0)

    def test_forecaster_with_wyoming_stub_and_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = fetch.WyomingUpperAir(cache_dir=tmp, pause_seconds=0, session=StubSession())
            fc = pfnn.PointForecaster(checkpoint_root=tmp, fetchers={"wyoming": w})
            out = fc.forecast(query="dewpoint +2 HRs", station="72451", init_time="2024-05-12T12:00")
            self.assertEqual((out["component"], out["lead_hours"], out["source"]), ("dewpoint", 2.0, "wyoming"))
            self.assertEqual(out["valid_time"], "2024-05-12T14:00:00Z")
            self.assertEqual(out["model"]["status"], "untrained")
            self.assertEqual(out["headline"]["forecast"], out["headline"]["observed"])   # persistence
            self.assertTrue(any("PERSISTENCE" in w for w in out["warnings"]))

    def test_train_then_forecast_beats_persistence_on_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = fetch.SyntheticUpperAir()
            data = src.fetch_range("72451", datetime.datetime(2023, 1, 1), datetime.datetime(2023, 6, 30))
            res = pfnn.train_component("temperature", data, "synthetic", ["72451"], epochs=8, checkpoint_root=tmp,
                                       verbose=False)
            self.assertLess(res["metrics"]["mae_nn_all"], res["metrics"]["mae_persistence_all"])
            fc = pfnn.PointForecaster(checkpoint_root=tmp)
            out = fc.forecast(component="temp", lead_hours="+3 HRs", station="72451", source="synthetic",
                              init_time="2023-07-01T00:00")
            self.assertEqual(out["model"]["status"], "trained")
            self.assertEqual(len(out["profile"]), 10)


class ApiAndHudTests(unittest.TestCase):
    def test_fastapi_routes(self):
        from fastapi.testclient import TestClient
        client = TestClient(api.app)
        self.assertEqual(client.get("/api/health").json()["status"], "ok")
        self.assertEqual(client.post("/api/resolve", json={"query": "dew pt +2 hrs"}).json()["component"], "dewpoint")
        r = client.post("/api/forecast", json={"component": "dewpoint", "lead_hours": "+2 HRs", "source": "synthetic",
                                               "init_time": "2024-05-12T12:00"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(client.post("/api/forecast", json={"component": "dewpoint", "source": "x"}).status_code, 422)
        self.assertEqual(client.post("/api/forecast", json={"query": "banana", "lead_hours": 2,
                                                            "source": "synthetic"}).status_code, 400)

    def test_flask_hud_in_process(self):
        os.environ["POINT_FORECAST_BACKEND"] = "local"
        try:
            hud = load("hud_app", "src/app.py")
            client = hud.app.test_client()
            self.assertIn(b"SKEW-T POINT FORECAST", client.get("/").data)
            with client.get("/js/custom.js") as js:
                self.assertEqual(js.status_code, 200)
            r = client.post("/hud/forecast", json={"query": "wind direction +6 hrs", "source": "synthetic",
                                                   "init_time": "2024-05-12T12:00"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual((r.json["backend"], r.json["component"]), ("in-process", "wind_direction"))
        finally:
            os.environ.pop("POINT_FORECAST_BACKEND", None)


if __name__ == "__main__":
    unittest.main()
