"""
Forecast_Upper_Air_Maps.py
==========================
Runs a trained upper-air CAM model for one initialisation and draws the 300 mb maps
(heights, shaded isotachs, wind barbs, divergence) for F00 .. F{hours}.

  python Forecast_Upper_Air_Maps.py --init 2024-05-12T12                       # default checkpoint, F00-F02
  python Forecast_Upper_Air_Maps.py --init 2024-05-12T12 --truth hrrr         # + HRRR maps & verification
  python Forecast_Upper_Air_Maps.py --checkpoint Models/checkpoints/upper_air/cam_300mb_synthetic_demo.pt \
      --init 2024-05-20T00 --truth synthetic

Output: Maps/output/<YYYYMMDDHH>/  (PNG maps, an .npz with the map-level fields, summary.json)
Only 00Z / 12Z initialisations are possible: the model starts from radiosonde observations.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from Database_Training.Grid_Interpolation import grid_observations, regrid_frame
from Database_Training.Map_Training_Data import get_obs_source
from Database_Training.Model_Grib_Fetch import get_truth_fetcher
from Database_Training.Upper_Air_Obs import check_observed_init, stations_near
from Maps.Upper_Air_Map import render_upper_air_map
from Models.Model_Grid import LambertGrid, get_domain
from Models.Upper_Air_CAM_Net import geometry_tensors, load_checkpoint, verify_map_level
from Train_Upper_Air_Maps import CHECKPOINT_DIR, parse_time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "Maps", "output")


def find_checkpoint(path: str | None) -> str:
    if path:
        return path
    found = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")), key=os.path.getmtime)
    if not found:
        raise SystemExit(f"no checkpoint in {CHECKPOINT_DIR}; train one with Train_Upper_Air_Maps.py")
    return found[-1]


def run_forecast(checkpoint: str, init, hours: int = 2, obs_source: str | None = None, domain: str | None = None,
                 truth_source: str | None = None, outdir: str | None = None, level: int | None = None,
                 device: str | None = None, render: bool = True, plot_obs: bool = False) -> dict:
    check_observed_init(init)
    model, meta = load_checkpoint(checkpoint, map_location=device or "cpu")
    dev = torch.device(device or "cpu")
    model.to(dev)
    trained = int(meta.get("trained_hours", 0))
    if hours > model.config.max_lead_hours:
        raise SystemExit(f"--hours {hours} exceeds the model's {model.config.max_lead_hours} h limit")
    grid = get_domain(domain) if domain else LambertGrid.from_dict(meta["grid"])
    levels = tuple(int(p) for p in model.config.levels)
    level = level or model.config.map_level
    k = levels.index(level)
    obs_source = obs_source or meta.get("obs_source", "wyoming")
    synthetic = obs_source == "synthetic" or meta.get("synthetic", False)

    # observations -> 3 km grid (nearest neighbour)
    table = get_obs_source(obs_source).fetch_obs(init, levels, stations_near(grid))
    if table.empty:
        raise SystemExit(f"no radiosonde observations for {init:%Y-%m-%d %HZ}")
    obs, dist, _ = grid_observations(table, grid, levels)
    geo = geometry_tensors(grid.geometry(), device=dev)
    with torch.no_grad():
        out = model(torch.from_numpy(obs)[None].to(dev), torch.from_numpy(np.minimum(dist, 5000.0))[None].to(dev), geo,
                    torch.tensor([float(init.hour)], device=dev),
                    torch.tensor([float(init.timetuple().tm_yday)], device=dev), n_steps=hours)
    frames = [out["analysis"][0]] + [out["states"][0, h] for h in range(1, hours + 1)]

    outdir = outdir or os.path.join(OUTPUT_ROOT, f"{init:%Y%m%d%H}")
    os.makedirs(outdir, exist_ok=True)
    label = f"UpperAirCAMNet 3 km ({meta.get('truth_source', '?')}-trained)"
    truth_fetcher = get_truth_fetcher(truth_source) if truth_source else None
    summary = {"init": init.isoformat(), "checkpoint": checkpoint, "trained_hours": trained, "level_hpa": level,
               "obs_source": obs_source, "radiosondes": int(table["station"].nunique()), "grid": grid.name,
               "maps": [], "verification": {}}
    npz = {}
    if render and plot_obs:
        filled = np.where(np.isfinite(obs), obs, np.nanmean(obs, axis=(2, 3), keepdims=True))
        path = os.path.join(outdir, f"input_nearest_neighbour_obs_{level}mb.png")
        render_upper_air_map(grid, filled[0, k], filled[1, k], filled[2, k], path, level_hpa=level, init_time=init,
                             fhour=0, model_label="MODEL INPUT: radiosondes, nearest-neighbour gridded",
                             synthetic=synthetic, show_divergence=False,
                             notes=[f"{summary['radiosondes']} radiosondes; each grid point takes its nearest station's value"])
        summary["maps"].append(path)
    for h, state in enumerate(frames):
        z, u, v = (state[i, k].cpu().numpy() for i in range(3))
        notes = [f"{summary['radiosondes']} radiosondes, nearest-neighbour gridded"]
        if h > trained:
            notes.append(f"F{h:02d} is BEYOND THE TRAINED RANGE (trained to F{trained:02d}) - unverified")
        derived = model.derived_fields(state[None], geo)
        npz.update({f"f{h:02d}_height_m": z, f"f{h:02d}_u_grid_ms": u, f"f{h:02d}_v_grid_ms": v,
                    f"f{h:02d}_temperature_k": state[3, k].cpu().numpy(), f"f{h:02d}_rh_pct": state[4, k].cpu().numpy()})
        for name, unit in (("divergence", "s-1"), ("vorticity", "s-1"), ("absolute_vorticity", "s-1"),
                           ("omega", "pa_s-1"), ("theta", "k"), ("theta_e", "k"), ("wind_speed", "ms")):
            npz[f"f{h:02d}_{name}_{unit}"] = derived[name][0, k].cpu().numpy()
        if render:
            path = os.path.join(outdir, f"cam_{level}mb_f{h:02d}.png")
            render_upper_air_map(grid, z, u, v, path, level_hpa=level, init_time=init, fhour=h, model_label=label,
                                 synthetic=synthetic, notes=notes)
            summary["maps"].append(path)
        if truth_fetcher is not None:
            frame = regrid_frame(truth_fetcher.fetch_frame(init, h, levels, grid), grid)
            truth = torch.from_numpy(frame.fields)[None].to(dev)
            summary["verification"][f"f{h:02d}"] = verify_map_level(state[None], truth, geo, grid.dx, k)
            if render:
                path = os.path.join(outdir, f"truth_{truth_source.replace(':', '_')}_{level}mb_f{h:02d}.png")
                render_upper_air_map(grid, frame.fields[0, k], frame.fields[1, k], frame.fields[2, k], path,
                                     level_hpa=level, init_time=init, fhour=h,
                                     model_label=f"{truth_source} (truth for verification)", synthetic=synthetic)
                summary["maps"].append(path)
    np.savez_compressed(os.path.join(outdir, f"cam_{level}mb_fields.npz"), **npz, grid=json.dumps(grid.to_dict()))
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="default: newest file in Models/checkpoints/upper_air/")
    p.add_argument("--init", type=parse_time, required=True, help="00Z or 12Z initialisation, e.g. 2024-05-12T12")
    p.add_argument("--hours", type=int, default=2, help="forecast hours (goal: 2; architecture max 48)")
    p.add_argument("--obs", default=None, help="wyoming | synthetic (default: what the model was trained on)")
    p.add_argument("--truth", default=None, help="also fetch & draw this truth source and verify against it")
    p.add_argument("--domain", default=None, help="run on another domain preset (default: the training domain)")
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--plot-obs", action="store_true", help="also draw the nearest-neighbour gridded radiosonde input")
    args = p.parse_args(argv)
    summary = run_forecast(find_checkpoint(args.checkpoint), args.init, args.hours, args.obs, args.domain, args.truth,
                           args.outdir, args.level, args.device, plot_obs=args.plot_obs)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
