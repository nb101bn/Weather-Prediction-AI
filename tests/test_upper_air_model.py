"""Tests for the 3 km upper-air CAM map model.  Run from the project root:
    python -m unittest discover tests -v
"""
import datetime
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from Database_Training.Grid_Interpolation import (grid_observations, nearest_neighbor_to_grid,  # noqa: E402
                                                  regrid_frame, ModelFrame)
from Database_Training.Map_Training_Data import (MapSampleBuilder, MapTrainingDataset, compute_normalization,  # noqa: E402
                                                 list_samples)
from Database_Training.Model_Grib_Fetch import (GribModelFetcher, byte_ranges, check_training_frame,  # noqa: E402
                                                observed_init_times, parse_idx, select_records)
from Database_Training.Synthetic_Atmosphere import SyntheticAtmosphere  # noqa: E402
from Database_Training.Upper_Air_Obs import check_observed_init, sounding_to_levels, stations_near  # noqa: E402
from Models import Atmos_Physics as ap  # noqa: E402
from Models.Model_Grid import (HRRR_CONUS, earth_to_grid_winds, get_domain, grid_to_earth_winds)  # noqa: E402
from Models.Upper_Air_CAM_Net import (DEFAULT_LEVELS, CAMNetConfig, UpperAirCAMNet, geometry_tensors,  # noqa: E402
                                      load_checkpoint, map_model_loss, save_checkpoint)

LEVELS = DEFAULT_LEVELS
INIT = datetime.datetime(2024, 5, 12, 12)
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def _has_grib_backend() -> bool:
    for name in ("pygrib", "eccodes"):
        try:
            __import__(name)
            return True
        except Exception:
            continue
    return False


def small_grid(nx=96, ny=80):
    return HRRR_CONUS.window_around(38.5, -97.5, nx, ny, name="test")


class GridTests(unittest.TestCase):
    def test_hrrr_corners(self):
        lat, lon = HRRR_CONUS.latlon()
        self.assertAlmostEqual(lat[0, 0], 21.138123, places=5)
        self.assertAlmostEqual(lon[0, 0], -122.719528, places=5)
        self.assertAlmostEqual(lat[-1, -1], 47.842195, places=4)     # official last grid point 47.842195N
        self.assertAlmostEqual(lon[-1, -1] % 360, 299.082807, places=3)  # 299.082807E

    def test_map_factor_matches_true_distance(self):
        lat, lon = HRRR_CONUS.latlon()
        m = HRRR_CONUS.map_factor(lat)
        j, i = 1000, 1700
        a1, o1, a2, o2 = map(np.radians, (lat[j, i], lon[j, i], lat[j, i + 1], lon[j, i + 1]))
        d = 2 * HRRR_CONUS.radius * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2
                                                      + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))
        self.assertAlmostEqual(HRRR_CONUS.dx / d, m[j, i], places=4)

    def test_wind_rotation(self):
        for lat, lon in [(45, -75), (30, -120)]:
            x1, y1 = HRRR_CONUS.project(lat, lon)
            x2, y2 = HRRR_CONUS.project(lat + 1e-4, lon)
            north = np.array([x2 - x1, y2 - y1]) / np.hypot(x2 - x1, y2 - y1)
            ug, vg = earth_to_grid_winds(0.0, 10.0, HRRR_CONUS.rotation_angle(lon))   # 10 m/s from the south
            np.testing.assert_allclose([ug / 10, vg / 10], north, atol=1e-4)
            ue, ve = grid_to_earth_winds(ug, vg, HRRR_CONUS.rotation_angle(lon))
            np.testing.assert_allclose([ue, ve], [0.0, 10.0], atol=1e-9)

    def test_windows(self):
        dom = get_domain("central_plains")
        self.assertEqual(dom.offset_in(HRRR_CONUS), (643, 305))
        lat_w, lon_w = dom.latlon()
        lat, lon = HRRR_CONUS.latlon()
        np.testing.assert_allclose(lat_w, lat[305:305 + dom.ny, 643:643 + dom.nx], atol=1e-6)
        self.assertEqual(dom.dx, 3000.0)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_zonal_flow_on_sphere(self):
        """Uniform 10 m/s west wind on the sphere: D = 0, zeta = u tan(lat) / a."""
        g = get_domain("demo")
        geo = {k: torch.tensor(v, dtype=torch.float64) for k, v in g.geometry().items()}
        ug, vg = earth_to_grid_winds(torch.full_like(geo["lat"], 10.0), torch.zeros_like(geo["lat"]),
                                     torch.tensor(g.rotation_angle(geo["lon"].numpy())))
        d = ap.divergence(ug, vg, geo["map_factor"], g.dx)[2:-2, 2:-2]
        z = ap.vorticity(ug, vg, geo["map_factor"], g.dx)[2:-2, 2:-2]
        expected = 10.0 * torch.tan(torch.deg2rad(geo["lat"]))[2:-2, 2:-2] / g.radius
        self.assertLess(float(d.abs().max()), 1e-9)
        self.assertLess(float((z - expected).abs().max() / expected.abs().mean()), 1e-3)

    def test_poisson_inverse(self):
        torch.manual_seed(0)
        rhs = ap.gaussian_smooth(torch.randn(2, 40, 50), 3.0) * 1e-9
        psi = ap.solve_poisson_dirichlet(rhs, 3000.0)
        lap = (psi[..., 1:-1, 2:] + psi[..., 1:-1, :-2] + psi[..., 2:, 1:-1] + psi[..., :-2, 1:-1]
               - 4 * psi[..., 1:-1, 1:-1]) / 3000.0 ** 2
        self.assertLess(float((lap - rhs[..., 1:-1, 1:-1]).abs().max() / rhs.abs().max()), 1e-10)
        self.assertEqual(float(psi[..., 0, :].abs().max()), 0.0)

    def test_kinematic_omega_sign(self):
        p = torch.tensor(LEVELS, dtype=torch.float64) * 100
        w = ap.kinematic_omega(torch.full((len(LEVELS), 1, 1), 1e-5), p)
        self.assertAlmostEqual(float(w[LEVELS.index(300)]), -0.2, places=6)   # divergence aloft -> lift
        self.assertEqual(float(w[-1]), 0.0)

    def test_laplacian_has_no_edge_artefact(self):
        z = torch.linspace(0, 300, 50)[None, :].repeat(40, 1)          # linear field: laplacian 0 everywhere
        lap = ap.laplacian(z, torch.ones(40, 50), 3000.0)
        self.assertLess(float(lap.abs().max()), 1e-12)

    def test_semi_lagrangian_translation(self):
        h, w, nl = 20, 80, 3
        ii = torch.arange(w, dtype=torch.float64).view(1, 1, 1, 1, w).expand(1, 1, nl, h, w)
        blob = torch.exp(-((ii - 20) ** 2) / 20.0)
        u = torch.full((1, nl, h, w), 10.0)
        out = ap.semi_lagrangian(blob, u, torch.zeros_like(u), torch.zeros_like(u), torch.ones(h, w), 3000.0,
                                 torch.tensor([500.0, 400.0, 300.0]) * 100, 3600.0)
        self.assertEqual(int(out[0, 0, 1, 10].argmax()), 32)                  # 36 km = 12 cells

    def test_thermodynamics_reference_values(self):
        t, p = torch.tensor([233.15]), torch.tensor([300.0])
        self.assertAlmostEqual(float(ap.potential_temperature(t, p)), 233.15 * (1000 / 300) ** (2 / 7), delta=0.05)
        # Bolton theta-e at 850 hPa, 290 K, 85 % (MetPy gives 340.15 K)
        self.assertAlmostEqual(float(ap.equivalent_potential_temperature(torch.tensor([290.0]), torch.tensor([85.0]),
                                                                          torch.tensor([850.0]))), 340.15, delta=0.2)
        self.assertAlmostEqual(float(ap.dewpoint(torch.tensor([290.0]), torch.tensor([100.0]))), 290.0, places=4)

    def test_physics_core_translates_balanced_wave(self):
        """f-plane, uniform flow + nondivergent geostrophic wave: the exact solution is a translation.
        The physics step must beat persistence by a wide margin away from the lateral boundaries."""
        H, W, dx, f0, U0, A = 100, 160, 3000.0, 1e-4, 20.0, 4.8e5
        y, x = torch.meshgrid(torch.arange(H) * dx, torch.arange(W) * dx, indexing="ij")
        k, l = 2 * math.pi / 600e3, math.pi / (H * dx)

        def state(shift):
            xs = x - shift
            psi = -U0 * y + A * torch.sin(k * xs) * torch.sin(l * y)
            u = U0 - A * l * torch.sin(k * xs) * torch.cos(l * y)
            v = A * k * torch.cos(k * xs) * torch.sin(l * y)
            s = torch.stack([9000 + f0 * psi / 9.80665, u, v, torch.full_like(u, 230.0), torch.full_like(u, 40.0)])
            return s[:, None].expand(5, 3, H, W)[None].clone()

        geo = {"map_factor": torch.ones(H, W), "coriolis": torch.full((H, W), f0)}
        core = ap.PhysicsCore((400, 300, 250), dx, substeps=2, source_smoothing_points=0.0)
        with torch.no_grad():
            out = core(state(0.0), geo)
        exact, s0 = state(U0 * 3600.0), state(0.0)
        sl = (0, 2, 1, slice(30, -30), slice(80, -20))
        err = float((out[sl] - exact[sl]).pow(2).mean().sqrt())
        pers = float((s0[sl] - exact[sl]).pow(2).mean().sqrt())
        self.assertLess(err, 0.25 * pers)


class AdaptiveEquationTests(unittest.TestCase):
    def test_coefficients_start_at_one_and_learn(self):
        eq = ap.AdaptiveEquation("demo", ["a", "b"], n_levels=2)
        self.assertTrue(torch.equal(eq.coeff.data, torch.ones(2, 2)))
        terms = {"a": torch.ones(1, 2, 3, 3), "b": 2 * torch.ones(1, 2, 3, 3)}
        target = 0.5 * terms["a"] + 2 * terms["b"]       # "truth" uses different coefficients
        opt = torch.optim.SGD(eq.parameters(), lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            ((eq(terms) - target) ** 2).mean().backward()
            opt.step()
        self.assertGreater(float(eq.coeff[1].mean()), 1.0)                       # moved toward the data
        np.testing.assert_allclose((eq(terms) - target).detach().numpy(), 0.0, atol=1e-3)

    def test_physics_core_recovers_perturbed_equations(self):
        """Perfect-model check of the 'AI may change the equations' mechanism: the truth is made by a physics
        core whose adiabatic and stretching terms are scaled by 0.4 and 0.6; a textbook core (all 1.0) trained
        on the state error must move those coefficients to (near) the true values."""
        torch.manual_seed(0)
        g = HRRR_CONUS.window_around(40.0, -97.0, 32, 32)
        geo = geometry_tensors(g.geometry())
        s0 = torch.from_numpy(SyntheticAtmosphere().fetch_frame(INIT, 0, LEVELS, g).fields)[None]
        truth_core = ap.PhysicsCore(LEVELS, g.dx, substeps=1, boundary_relax_points=4)
        with torch.no_grad():
            truth_core.thermo_eq.coeff.fill_(0.4)
            truth_core.vorticity_eq.coeff[0].fill_(0.6)
            target = truth_core(s0, geo)
        core = ap.PhysicsCore(LEVELS, g.dx, substeps=1, boundary_relax_points=4)
        for eq in (core.advection, core.divergence_eq, core.height_eq):
            eq.coeff.requires_grad_(False)
        scale = torch.tensor([5.0, 1.0, 1.0, 0.2, 2.0]).view(1, 5, 1, 1, 1)
        opt = torch.optim.Adam([core.thermo_eq.coeff, core.vorticity_eq.coeff], lr=0.05)
        first = None
        for _ in range(60):
            loss = (((core(s0, geo) - target) / scale) ** 2).mean()
            first = first if first is not None else float(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss), 0.05 * first)
        self.assertAlmostEqual(float(core.thermo_eq.coeff.mean()), 0.4, delta=0.15)
        self.assertAlmostEqual(float(core.vorticity_eq.coeff[0].mean()), 0.6, delta=0.15)

    def test_physics_core_exposes_divergence_equation(self):
        core = ap.PhysicsCore(LEVELS, 3000.0)
        table = core.coefficient_table()
        self.assertEqual(set(table["divergence"]["terms"]), set(ap.DIVERGENCE_TERMS))
        self.assertEqual(table["divergence"]["terms"]["jacobian"]["300 hPa"], 1.0)
        self.assertIn("+2J(u,v)", table["divergence"]["descriptions"].values())


class InterpolationTests(unittest.TestCase):
    def test_nearest_neighbor(self):
        # two stations at 37.5N: one near 100W, one near 90W; grid points at 35/40N x 100/90W
        grid_lat, grid_lon = np.array([[35.0, 35.0], [40.0, 40.0]]), np.array([[-100.0, -90.0], [-100.0, -90.0]])
        obs_lat, obs_lon = np.array([37.5, 37.5]), np.array([-99.9, -90.1])
        vals, dist, idx = nearest_neighbor_to_grid(obs_lat, obs_lon, np.array([1.0, 2.0]), grid_lat, grid_lon)
        np.testing.assert_array_equal(vals, [[1.0, 2.0], [1.0, 2.0]])
        np.testing.assert_array_equal(idx, [[0, 1], [0, 1]])
        self.assertAlmostEqual(dist[0, 0], 278.3, delta=1.0)                     # 2.5 deg of latitude (+0.1 deg lon)
        vals, _, _ = nearest_neighbor_to_grid(obs_lat, obs_lon, np.array([np.nan, 2.0]), grid_lat, grid_lon)
        self.assertTrue((vals == 2.0).all())                                     # missing obs are skipped
        vals, _, _ = nearest_neighbor_to_grid(obs_lat, obs_lon, np.array([1.0, 2.0]), grid_lat, grid_lon,
                                              max_distance_km=100.0)
        self.assertTrue(np.isnan(vals).all())                                    # everything is > 100 km away

    def test_grid_observations_rotates_winds(self):
        import pandas as pd
        g = small_grid()
        obs = pd.DataFrame([["A", 38.5, -97.5, 300.0, 9200.0, 0.0, 20.0, 230.0, 40.0]],
                           columns=["station", "lat", "lon", "level", "Z", "U_earth", "V_earth", "T", "RH"])
        vals, dist, counts = grid_observations(obs, g, [300])
        _, lon = g.latlon()
        ug, vg = earth_to_grid_winds(0.0, 20.0, g.rotation_angle(lon))
        np.testing.assert_allclose(vals[1, 0], ug, atol=1e-4)
        np.testing.assert_allclose(vals[2, 0], vg, atol=1e-4)
        self.assertTrue((vals[0, 0] == 9200.0).all())
        self.assertEqual(counts[("wind", 300)], 1)

    def test_regrid_window_is_exact_and_nn_otherwise(self):
        g = small_grid(60, 50)
        fields = np.random.default_rng(0).normal(size=(5, 2, g.ny, g.nx)).astype(np.float32)
        frame = ModelFrame(fields, (300, 250), g, INIT, 0, "test")
        sub = g.window(10, 5, 30, 20)
        np.testing.assert_array_equal(regrid_frame(frame, sub).fields, fields[..., 5:25, 10:40])
        lat, lon = g.latlon()                                                   # same points, "unknown" projection
        frame2 = ModelFrame(fields, (300, 250), None, INIT, 0, "test", grid_relative=False, lats=lat, lons=lon)
        out = regrid_frame(frame2, sub)
        np.testing.assert_allclose(out.fields[0], fields[0, :, 5:25, 10:40])


class TrainingRuleTests(unittest.TestCase):
    def test_only_observed_inits_and_first_12_hours(self):
        check_training_frame(datetime.datetime(2024, 5, 1, 0), 12)
        check_training_frame(datetime.datetime(2024, 5, 1, 12), 0)
        for bad_init in (datetime.datetime(2024, 5, 1, 6), datetime.datetime(2024, 5, 1, 18)):
            with self.assertRaises(ValueError):
                check_training_frame(bad_init, 1)
        with self.assertRaises(ValueError):
            check_training_frame(datetime.datetime(2024, 5, 1, 0), 13)
        with self.assertRaises(ValueError):
            check_observed_init(datetime.datetime(2024, 5, 1, 3))
        times = list(observed_init_times(datetime.datetime(2024, 5, 1, 5), datetime.datetime(2024, 5, 2, 12)))
        self.assertEqual([t.hour for t in times], [12, 0, 12])

    def test_sounding_to_levels_no_extrapolation(self):
        import pandas as pd
        df = pd.DataFrame({"PRES": [850, 700, 500, 400, 300, 250], "HGHT": [1500, 3100, 5700, 7300, 9300, 10500],
                           "TEMP": [15, 5, -12, -24, -40, -50], "DWPT": [10, -2, -25, -35, -50, -60],
                           "RELH": [70, 60, 40, 35, 30, 25], "MIXR": [1] * 6, "DRCT": [270] * 6,
                           "SKNT": [20, 30, 50, 60, 80, 90]})
        out = sounding_to_levels(df, LEVELS).set_index("level")
        self.assertAlmostEqual(out.loc[300.0, "Z"], 9300.0)
        self.assertAlmostEqual(out.loc[300.0, "U_earth"], 80 * 0.514444, places=3)     # west wind -> +u
        self.assertTrue(np.isnan(out.loc[200.0, "Z"]))                                  # above the burst
        self.assertTrue(np.isnan(out.loc[100.0, "T"]))


class GribFetchTests(unittest.TestCase):
    def test_idx_parsing_and_ranges(self):
        idx = ("1:0:d=2024051212:TMP:2 m above ground:anl:\n2:100:d=2024051212:HGT:300 mb:anl:\n"
               "3:250:d=2024051212:UGRD:300 mb:anl:\n4:400:d=2024051212:VGRD:300 mb:anl:\n"
               "5:600:d=2024051212:TMP:300 mb:anl:\n6:700:d=2024051212:RH:300 mb:anl:\n")
        recs = select_records(parse_idx(idx), ["Z", "U", "V", "T", "RH"], [300])
        self.assertEqual([r.variable for r in recs], ["HGT", "UGRD", "VGRD", "TMP", "RH"])
        self.assertEqual(byte_ranges(recs), [(100, None)])                      # contiguous -> one open range

    def test_grib_download_and_decode(self):
        """Real GRIB2 fixture files (tests/fixtures, written by tests/make_grib_fixtures.py) served through a fake
        HTTP server that honours Range requests; decoded with whichever backend is installed."""
        if not _has_grib_backend():
            self.skipTest("neither pygrib nor eccodes is installed")
        from grib_fixtures import RangeSession
        src = HRRR_CONUS.window(600, 280, 40, 32)
        fields = SyntheticAtmosphere().fetch_frame(INIT, 1, LEVELS, src).fields
        fetcher = GribModelFetcher("hrrr", cache_dir=None, verbose=False)
        url = fetcher.url(INIT, 1, "aws")
        self.assertEqual(url, "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20240512/conus/hrrr.t12z.wrfprsf01.grib2")
        for name, grid_relative in (("hrrr_window_f01", True), ("hrrr_window_f01_northfirst", False)):
            with open(os.path.join(FIXTURES, name + ".grib2"), "rb") as fh:
                data = fh.read()
            with open(os.path.join(FIXTURES, name + ".grib2.idx"), "rb") as fh:
                idx = fh.read()
            session = RangeSession({url: data, url + ".idx": idx})
            frame = GribModelFetcher("hrrr", cache_dir=None, session=session, verbose=False).fetch_frame(INIT, 1, LEVELS)
            np.testing.assert_allclose(frame.fields, fields, atol=0.05)
            self.assertEqual(frame.grid.offset_in(HRRR_CONUS), (600, 280))
            self.assertEqual(frame.grid_relative, grid_relative)
            ranged = [h for u, h in session.calls if not u.endswith(".idx")]
            self.assertTrue(ranged and all("Range" in h for h in ranged))

    def test_both_grib_backends_agree(self):
        """pygrib and the eccodes bindings must never share a process (two libeccodes copies crash), so each
        backend decodes the fixture in its own subprocess."""
        import subprocess
        code = ("import sys, numpy as np; sys.path.insert(0, {root!r}); "
                "from Database_Training.Model_Grib_Fetch import decode_grib; "
                "out, g, rel = decode_grib({path!r}, {levels!r}, backend={be!r}); "
                "print(float(np.nansum(out)), g.nx, g.ny, rel)")
        results = {}
        for be in ("pygrib", "eccodes"):
            try:
                __import__(be)
            except Exception:
                continue
            proc = subprocess.run([sys.executable, "-c", code.format(root=ROOT, be=be, levels=LEVELS,
                                   path=os.path.join(FIXTURES, "hrrr_window_f01_northfirst.grib2"))],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
            results[be] = proc.stdout.split()
        if len(results) < 2:
            self.skipTest("needs both pygrib and eccodes installed")
        self.assertEqual(results["pygrib"], results["eccodes"])

    def test_missing_on_all_mirrors(self):
        from grib_fixtures import RangeSession
        from Database_Training.Model_Grib_Fetch import DataUnavailable
        fetcher = GribModelFetcher("hrrr", cache_dir=None, session=RangeSession({}), verbose=False)
        with self.assertRaises(DataUnavailable):
            fetcher.fetch_frame(INIT, 0, LEVELS)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.grid = small_grid(64, 64)
        syn = SyntheticAtmosphere()
        cls.truth = np.stack([syn.fetch_frame(INIT, h, LEVELS, cls.grid).fields for h in range(3)])
        table = syn.fetch_obs(INIT, LEVELS, stations_near(cls.grid))
        cls.obs, cls.dist, _ = grid_observations(table, cls.grid, LEVELS)
        cls.geo = geometry_tensors(cls.grid.geometry())

    def model(self):
        return UpperAirCAMNet(CAMNetConfig(width=8, hidden=16, analysis_depth=3))

    def run_model(self, model, n_steps=2, **kw):
        return model(torch.from_numpy(self.obs)[None], torch.from_numpy(self.dist)[None], self.geo,
                     torch.tensor([12.0]), torch.tensor([133.0]), n_steps=n_steps, **kw)

    def test_untrained_analysis_is_the_smoothed_first_guess(self):
        model = self.model()
        out = self.run_model(model)
        self.assertEqual(tuple(out["states"].shape), (1, 3, 5, len(LEVELS), 64, 64))
        filled, have, smoothed = model.first_guess(torch.from_numpy(self.obs)[None])
        np.testing.assert_allclose(filled[have].numpy(), self.obs[np.isfinite(self.obs)])      # obs kept as gridded
        ref = smoothed[:, list(model.config.smoothing_scales_km).index(model.config.first_guess_km)]
        ref = torch.cat([ref[:, :4], ref[:, 4:].clamp(0, 100)], 1)
        np.testing.assert_allclose(out["analysis"].detach().numpy(), ref.numpy(), atol=1e-3)

    def test_dataset_first_guess_matches_full_domain(self):
        """Crops must see the first guess computed on the full domain (what deployment computes)."""
        syn = SyntheticAtmosphere()
        with tempfile.TemporaryDirectory() as tmp:
            MapSampleBuilder(self.grid, LEVELS, syn, syn, tmp, fhours=(0,), verbose=False).build_one(INIT)
            scales = CAMNetConfig().smoothing_scales_km
            ds = MapTrainingDataset(tmp, [INIT], n_frames=1, crop=24, seed=3, smoothing_scales_km=scales)
            item = ds[0]
            full = MapTrainingDataset(tmp, [INIT], n_frames=1, crop=None, smoothing_scales_km=scales)[0]
        obs_full, obs_crop = full["obs"], item["obs"]
        hits = [(j, i) for j in range(64 - 23) for i in range(64 - 23)
                if torch.equal(torch.nan_to_num(obs_full[..., j:j + 24, i:i + 24]), torch.nan_to_num(obs_crop))
                and torch.allclose(full["smoothed"][..., j:j + 24, i:i + 24], item["smoothed"])]
        self.assertTrue(hits, "crop's first guess differs from the full-domain first guess")

    def test_gradients_reach_network_and_equations(self):
        model = self.model()
        out = self.run_model(model, start_state=torch.from_numpy(self.truth[0])[None])
        loss, parts = map_model_loss(model, out, torch.from_numpy(self.truth)[None])
        loss.backward()
        self.assertTrue(np.isfinite(float(loss)))
        self.assertGreater(float(model.physics.divergence_eq.coeff.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.analysis_net.out.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.forecast_cell.out.weight.grad.abs().sum()), 0.0)

    def test_lead_limit(self):
        with self.assertRaises(ValueError):
            self.run_model(self.model(), n_steps=49)

    def test_checkpoint_roundtrip(self):
        model = self.model()
        with torch.no_grad():
            model.physics.divergence_eq.coeff[0, 2] = 0.7
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pt")
            save_checkpoint(model, path, {"trained_hours": 2})
            loaded, meta = load_checkpoint(path)
        self.assertEqual(meta["trained_hours"], 2)
        self.assertAlmostEqual(loaded.coefficient_table()["divergence"]["terms"]["divergence_squared"]["300 hPa"], 0.7,
                               places=5)
        with torch.no_grad():
            a = self.run_model(model, n_steps=1)["states"]
            b = self.run_model(loaded, n_steps=1)["states"]
        torch.testing.assert_close(a, b)


class PipelineTests(unittest.TestCase):
    def test_build_dataset_and_render_map(self):
        grid = small_grid(64, 56)
        syn = SyntheticAtmosphere()
        with tempfile.TemporaryDirectory() as tmp:
            builder = MapSampleBuilder(grid, LEVELS, syn, syn, tmp, fhours=(0, 1, 2), verbose=False)
            report = builder.build(datetime.datetime(2024, 5, 1, 3), datetime.datetime(2024, 5, 2, 0))
            self.assertEqual([t.hour for t in report["built"]], [12, 0])       # 03Z start -> 12Z, 00Z only
            inits = list_samples(tmp)
            ds = MapTrainingDataset(tmp, inits, n_frames=3, crop=32)
            item = ds[0]
            self.assertEqual(tuple(item["truth"].shape), (3, 5, len(LEVELS), 32, 32))
            stats = compute_normalization(tmp, inits, LEVELS, grid.dx)
            self.assertEqual(stats["state_std"].shape, (5, len(LEVELS)))
            from Maps.Upper_Air_Map import render_upper_air_map
            truth = np.load(os.path.join(tmp, f"{inits[0]:%Y%m%d%H}", "truth.npy"))
            k = LEVELS.index(300)
            path = render_upper_air_map(grid, truth[1, 0, k], truth[1, 1, k], truth[1, 2, k],
                                        os.path.join(tmp, "map.png"), init_time=inits[0], fhour=1, synthetic=True,
                                        dpi=40, barb_spacing_km=30)
            self.assertGreater(os.path.getsize(path), 10_000)


class DeploymentTests(unittest.TestCase):
    def test_run_forecast_end_to_end(self):
        """Checkpoint -> radiosondes (synthetic) -> model F00-F02 -> fields + verification (maps off for speed)."""
        from Forecast_Upper_Air_Maps import run_forecast
        grid = small_grid(48, 40)
        model = UpperAirCAMNet(CAMNetConfig(width=8, hidden=16, analysis_depth=3))
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "m.pt")
            save_checkpoint(model, ckpt, {"grid": grid.to_dict(), "obs_source": "synthetic", "synthetic": True,
                                          "trained_hours": 2, "truth_source": "synthetic"})
            summary = run_forecast(ckpt, INIT, hours=2, truth_source="synthetic", outdir=tmp, render=False)
            self.assertEqual(sorted(summary["verification"]), ["f00", "f01", "f02"])
            self.assertGreater(summary["radiosondes"], 5)
            fields = np.load(os.path.join(tmp, "cam_300mb_fields.npz"))
            for key in ("f02_divergence_s-1", "f02_omega_pa_s-1", "f02_theta_e_k", "f01_height_m"):
                self.assertEqual(fields[key].shape, (40, 48))
            with self.assertRaises(ValueError):
                run_forecast(ckpt, datetime.datetime(2024, 5, 12, 6), hours=1, outdir=tmp, render=False)


if __name__ == "__main__":
    unittest.main()
