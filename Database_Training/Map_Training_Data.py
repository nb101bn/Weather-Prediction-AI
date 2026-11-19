"""
Map_Training_Data.py
====================
Builds and serves the training set of the upper-air map model.

One sample = one OBSERVED initialisation (00Z or 12Z):
    obs.npy    [5, L, ny, nx]  radiosonde observations nearest-neighbour gridded to the 3 km grid (NaN = none)
    dist.npy   [L, ny, nx]     distance (km) to the radiosonde used at each grid point
    truth.npy  [T, 5, L, ny, nx]  model output f00 .. f{T-1} on the same grid (the "ground truth")
    meta.json  init time, forecast hours, sources, grid, station counts

stored under  Database_Training/cache/map_training/<truth source>/<domain>/<YYYYMMDDHH>/.
Arrays are plain .npy so training can memory-map them and read random crops cheaply.

``MapTrainingDataset`` (torch Dataset) returns random crops (training) or the full domain (validation).
"""
from __future__ import annotations

import datetime
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from Database_Training.Grid_Interpolation import STATE_VARS, grid_observations, regrid_frame
from Database_Training.Model_Grib_Fetch import DataUnavailable, check_training_frame, observed_init_times
from Database_Training.Upper_Air_Obs import stations_near
from Models import Atmos_Physics as ap
from Models.Model_Grid import LambertGrid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "Database_Training", "cache", "map_training")


def get_obs_source(name: str):
    """'wyoming' (real radiosondes) or 'synthetic'. Returns an object with fetch_obs(init, levels, stations)."""
    name = (name or "wyoming").lower()
    if name == "wyoming":
        from Database_Training.Upper_Air_Obs import RaobObservations
        return RaobObservations()
    if name == "synthetic":
        from Database_Training.Synthetic_Atmosphere import SyntheticAtmosphere
        return SyntheticAtmosphere()
    raise ValueError(f"unknown observation source '{name}' (expected 'wyoming' or 'synthetic')")


def sample_dir(root: str, init_time: datetime.datetime) -> str:
    return os.path.join(root, f"{init_time:%Y%m%d%H}")


class MapSampleBuilder:
    def __init__(self, grid: LambertGrid, levels, truth_fetcher, obs_source, root: str, fhours=(0, 1, 2),
                 max_obs_distance_km: float | None = None, verbose: bool = True):
        self.grid, self.levels = grid, tuple(int(p) for p in levels)
        self.truth, self.obs = truth_fetcher, obs_source
        self.root, self.fhours = root, tuple(int(h) for h in fhours)
        self.max_obs_distance_km, self.verbose = max_obs_distance_km, verbose
        if list(self.fhours) != list(range(len(self.fhours))):
            raise ValueError("fhours must be consecutive from f00 (e.g. 0, 1, 2)")

    def observations(self, init_time: datetime.datetime):
        """Gridded observations for one init (also used at forecast time)."""
        stations = stations_near(self.grid)
        table = self.obs.fetch_obs(init_time, self.levels, stations)
        if table.empty:
            raise DataUnavailable(f"no radiosonde observations for {init_time:%Y-%m-%d %HZ}")
        values, dist, counts = grid_observations(table, self.grid, self.levels, self.max_obs_distance_km)
        return values, dist, {"stations": int(table["station"].nunique()),
                               "counts": {f"{k[0]}@{k[1]}": v for k, v in counts.items()}}

    def build_one(self, init_time: datetime.datetime, overwrite: bool = False) -> dict | None:
        for h in self.fhours:
            check_training_frame(init_time, h)          # 00Z/12Z and f00-f12 only
        folder = sample_dir(self.root, init_time)
        if not overwrite and os.path.exists(os.path.join(folder, "meta.json")):
            with open(os.path.join(folder, "meta.json")) as fh:
                meta = json.load(fh)
            if meta.get("fhours") == list(self.fhours) and meta.get("levels") == list(self.levels):
                return meta
        obs, dist, obs_info = self.observations(init_time)
        frames = []
        for h in self.fhours:
            frame = self.truth.fetch_frame(init_time, h, self.levels, self.grid)
            frames.append(regrid_frame(frame, self.grid).fields)
        truth = np.stack(frames).astype(np.float32)
        if not np.isfinite(truth).all():
            raise DataUnavailable(f"{init_time:%Y-%m-%d %HZ}: truth has missing values on the domain")
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "obs.npy"), obs.astype(np.float32))
        np.save(os.path.join(folder, "dist.npy"), np.minimum(dist, 5000.0).astype(np.float32))
        np.save(os.path.join(folder, "truth.npy"), truth)
        meta = {"init_time": init_time.isoformat(), "fhours": list(self.fhours), "levels": list(self.levels),
                "variables": list(STATE_VARS), "truth_source": getattr(self.truth, "source", "unknown"),
                "obs_source": getattr(self.obs, "source", type(self.obs).__name__), "grid": self.grid.to_dict(),
                "obs": obs_info}
        with open(os.path.join(folder, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        if hasattr(self.truth, "discard"):
            self.truth.discard(init_time)
        return meta

    def build(self, start: datetime.datetime, end: datetime.datetime, overwrite: bool = False) -> dict:
        ok, failed = [], {}
        for init in observed_init_times(start, end):
            try:
                meta = self.build_one(init, overwrite)
                ok.append(init)
                if self.verbose:
                    print(f"[data] {init:%Y-%m-%d %HZ}: ok ({meta['obs']['stations']} radiosondes)")
            except (DataUnavailable, ValueError, OSError) as exc:
                # one missing model run / sounding set: skip it. (Wyoming being unreachable raises
                # UpstreamUnavailable instead, which stops the build - nothing else would work either.)
                failed[init.isoformat()] = str(exc)
                if self.verbose:
                    print(f"[data] {init:%Y-%m-%d %HZ}: skipped - {exc}")
        return {"built": ok, "failed": failed}


def list_samples(root: str, n_frames: int | None = None) -> list[datetime.datetime]:
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        meta_path = os.path.join(root, name, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                meta = json.load(fh)
            if n_frames is None or len(meta["fhours"]) >= n_frames:
                out.append(datetime.datetime.fromisoformat(meta["init_time"]))
    return out


class MapTrainingDataset(Dataset):
    """Random 3 km crops (``crop`` = size in grid points) or the full domain (``crop=None``).

    If ``smoothing_scales_km`` is given, the multi-scale first guess of the analysis is computed on the FULL domain
    (with the model's own function) before cropping, so training crops see exactly what deployment sees."""

    def __init__(self, root: str, inits, n_frames: int = 3, crop: int | None = 128, samples_per_init: int = 1,
                 seed: int | None = None, smoothing_scales_km=None):
        self.smoothing_scales_km = tuple(smoothing_scales_km) if smoothing_scales_km else None
        self.root, self.inits = root, list(inits)
        self.n_frames, self.crop, self.samples_per_init = n_frames, crop, samples_per_init
        # torch's RNG is re-seeded per DataLoader worker, so workers don't draw identical crops
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None
        if not self.inits:
            raise ValueError(f"no training samples in {root}")
        with open(os.path.join(sample_dir(root, self.inits[0]), "meta.json")) as fh:
            meta = json.load(fh)
        self.grid = LambertGrid.from_dict(meta["grid"])
        self.levels = tuple(meta["levels"])
        self.geometry = self.grid.geometry()
        if crop is not None and (crop > self.grid.nx or crop > self.grid.ny):
            raise ValueError(f"crop {crop} larger than the {self.grid.nx}x{self.grid.ny} domain")

    def __len__(self):
        return len(self.inits) * self.samples_per_init

    def __getitem__(self, idx):
        init = self.inits[idx // self.samples_per_init]
        folder = sample_dir(self.root, init)
        if self.crop is None:
            js, is_ = slice(None), slice(None)
        else:
            j0 = int(torch.randint(0, self.grid.ny - self.crop + 1, (1,), generator=self.generator))
            i0 = int(torch.randint(0, self.grid.nx - self.crop + 1, (1,), generator=self.generator))
            js, is_ = slice(j0, j0 + self.crop), slice(i0, i0 + self.crop)
        load = lambda name: np.load(os.path.join(folder, name), mmap_mode="r")
        smoothed = None
        if self.smoothing_scales_km:
            from Models.Upper_Air_CAM_Net import first_guess_stack
            full = torch.from_numpy(np.array(load("obs.npy")))[None]
            with torch.no_grad():
                _, _, sm = first_guess_stack(full, self.levels, self.grid.dx, self.smoothing_scales_km)
            smoothed = sm[0][..., js, is_].contiguous()
        obs = np.array(load("obs.npy")[..., js, is_])
        dist = np.array(load("dist.npy")[..., js, is_])
        truth = np.array(load("truth.npy")[: self.n_frames, ..., js, is_])
        geo = {k: torch.from_numpy(np.ascontiguousarray(v[js, is_])) for k, v in self.geometry.items()}
        item = {"obs": torch.from_numpy(obs), "dist": torch.from_numpy(dist), "truth": torch.from_numpy(truth),
                "geo": geo, "init_hour": torch.tensor(float(init.hour)),
                "day_of_year": torch.tensor(float(init.timetuple().tm_yday)), "init": init.isoformat()}
        if smoothed is not None:
            item["smoothed"] = smoothed
        return item


def compute_normalization(root: str, inits, levels, dx: float, max_samples: int = 60) -> dict:
    """Per (variable, level) mean/std of the truth, std of the hourly change and of the derived fields."""
    inits = list(inits)[:max_samples]
    states, deltas, diags = [], [], []
    for init in inits:
        truth = np.load(os.path.join(sample_dir(root, init), "truth.npy"), mmap_mode="r")
        with open(os.path.join(sample_dir(root, init), "meta.json")) as fh:
            grid = LambertGrid.from_dict(json.load(fh)["grid"])
        sub = (slice(None, None, 4), slice(None, None, 4))
        f0 = np.array(truth[0])
        states.append(f0[..., sub[0], sub[1]].reshape(5, len(levels), -1))
        if truth.shape[0] > 1:
            deltas.append((np.array(truth[1]) - f0)[..., sub[0], sub[1]].reshape(5, len(levels), -1))
        t = torch.from_numpy(f0)[None]
        m = torch.from_numpy(grid.map_factor().astype(np.float32))
        d = ap.divergence(t[:, 1], t[:, 2], m, dx)
        z = ap.vorticity(t[:, 1], t[:, 2], m, dx)
        w = ap.kinematic_omega(d, torch.tensor(levels, dtype=torch.float32) * 100.0)
        diags.append(torch.stack([d[0], z[0], w[0]]).numpy()[..., sub[0], sub[1]].reshape(3, len(levels), -1))
    s = np.concatenate(states, -1)
    stats = {"state_mean": s.mean(-1), "state_std": np.maximum(s.std(-1), 1e-3)}
    if deltas:
        d = np.concatenate(deltas, -1)
        stats["delta_std"] = np.maximum(np.sqrt((d ** 2).mean(-1)), 1e-3)
    dg = np.concatenate(diags, -1)
    stats["diag_std"] = np.maximum(np.sqrt((dg ** 2).mean(-1)), 1e-9)
    return {k: v.astype(np.float32) for k, v in stats.items()}
