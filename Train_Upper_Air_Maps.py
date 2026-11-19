"""
Train_Upper_Air_Maps.py
=======================
Trains the 3 km upper-air CAM model (Models/Upper_Air_CAM_Net.py) for the 300 mb map product.

Inputs  : radiosonde observations at 00Z/12Z, nearest-neighbour gridded to the 3 km grid
Truth   : model output f00..fN from the SAME 00Z/12Z initialisations (HRRR / NAM 3 km / AWIPS EDEX)
Current goal: f00 (analysis), f01 and f02 -> ``--hours 2`` (the default). The architecture steps hourly
up to 48 h, but longer leads are not trained yet.

Examples
--------
  # real data (HRRR truth from the NOAA archive + Wyoming radiosondes), central-plains 3 km window
  python Train_Upper_Air_Maps.py --truth hrrr --obs wyoming --start 2024-04-01 --end 2024-06-30

  # NAM 3 km truth, or grids from an AWIPS EDEX server (real-time only)
  python Train_Upper_Air_Maps.py --truth nam3k --start 2024-05-01 --end 2024-05-31
  python Train_Upper_Air_Maps.py --truth awips:HRRR --start 2024-05-12T00 --end 2024-05-12T12

  # offline smoke test on synthetic data (CPU-sized model)
  python Train_Upper_Air_Maps.py --truth synthetic --obs synthetic --domain demo --start 2024-05-01 \
      --end 2024-05-20 --epochs 4 --width 16 --hidden 48 --crop 96
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from Database_Training.Map_Training_Data import (DEFAULT_ROOT, MapSampleBuilder, MapTrainingDataset,
                                                 compute_normalization, get_obs_source, list_samples)
from Database_Training.Model_Grib_Fetch import get_truth_fetcher
from Models.Model_Grid import DOMAIN_PRESETS, get_domain
from Models.Upper_Air_CAM_Net import (DEFAULT_LEVELS, CAMNetConfig, UpperAirCAMNet, map_model_loss, save_checkpoint,
                                      verify_map_level)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "Models", "checkpoints", "upper_air")


def parse_time(text: str) -> datetime.datetime:
    text = text.strip().replace("Z", "").replace("z", "")
    for fmt in ("%Y-%m-%dT%H", "%Y-%m-%d %H", "%Y%m%d%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"cannot parse time '{text}' (use YYYY-MM-DD or YYYY-MM-DDTHH)")


def default_checkpoint(truth: str, domain: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"cam_300mb_{truth.replace(':', '_')}_{domain}.pt")


def data_root(truth: str, obs: str, domain: str) -> str:
    return os.path.join(DEFAULT_ROOT, f"{truth.replace(':', '_')}__{obs}", domain)


# ---------------------------------------------------------------------------------------------- data
def build_dataset(args, grid) -> str:
    root = data_root(args.truth, args.obs, args.domain)
    if args.skip_build:
        return root
    truth_kwargs = {"keep_grib": args.keep_grib} if args.truth in ("hrrr", "nam3k") else {}
    if args.mirror and args.truth in ("hrrr", "nam3k"):
        truth_kwargs["mirrors"] = args.mirror
    builder = MapSampleBuilder(grid, args.levels, get_truth_fetcher(args.truth, **truth_kwargs),
                               get_obs_source(args.obs), root, fhours=tuple(range(args.hours + 1)))
    report = builder.build(args.start, args.end, overwrite=args.rebuild)
    print(f"[data] {len(report['built'])} initialisations ready, {len(report['failed'])} skipped -> {root}")
    return root


def to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: UpperAirCAMNet, dataset: MapTrainingDataset, n_steps: int, device) -> dict:
    """Full-domain verification at the map level for every lead, with baselines:
    model (from its own analysis, i.e. as deployed), model started from the true f00 (forecast skill
    alone), physics core only from the true f00, persistence of the true f00, the raw nearest-neighbour
    observation grid held constant, and its smoothed version (the analysis first guess) held constant."""
    model.eval()
    k = model.map_level_index
    sums, n = {}, 0
    loss_sum = 0.0
    for i in range(len(dataset)):
        b = to_device({kk: (vv.unsqueeze(0) if torch.is_tensor(vv) else ({a: c.unsqueeze(0) for a, c in vv.items()}
                                                                          if isinstance(vv, dict) else vv))
                       for kk, vv in dataset[i].items()}, device)
        truth, geo = b["truth"], b["geo"]
        out = model(b["obs"], b["dist"], geo, b["init_hour"], b["day_of_year"], n_steps=n_steps)
        loss, _ = map_model_loss(model, out, truth)
        loss_sum += float(loss)
        from_truth = model(b["obs"], b["dist"], geo, b["init_hour"], b["day_of_year"], n_steps=n_steps,
                           start_state=truth[:, 0])
        obs_filled, _, smoothed = model.first_guess(b["obs"])
        first_guess = smoothed[:, list(model.config.smoothing_scales_km).index(model.config.first_guess_km)]
        phys = [truth[:, 0]]
        for _ in range(n_steps):
            phys.append(model.physics(phys[-1], geo, 3600.0 * model.config.step_hours))
        for h in range(n_steps + 1):
            cand = {"model": out["analysis"] if h == 0 else out["states"][:, h],
                    "nn_obs_persistence": obs_filled, "smoothed_obs_persistence": first_guess}
            if h > 0:
                cand.update({"model_from_true_f00": from_truth["states"][:, h], "physics_only_from_true_f00": phys[h],
                             "true_f00_persistence": truth[:, 0]})
            for name, state in cand.items():
                m = verify_map_level(state, truth[:, h], geo, model.config.dx, k)
                acc = sums.setdefault(f"f{h:02d}", {}).setdefault(name, {})
                for key, val in m.items():
                    acc[key] = acc.get(key, 0.0) + val
        n += 1
    for lead in sums.values():
        for name in lead.values():
            for key in name:
                name[key] = round(name[key] / max(n, 1), 3)
    model.train()
    return {"loss": loss_sum / max(n, 1), "n": n, "level_hpa": model.config.levels[k], "leads": sums}


def print_metrics(metrics: dict) -> None:
    print(f"  validation ({metrics['n']} inits, {metrics['level_hpa']} mb, RMSE)  loss={metrics['loss']:.4f}")
    print(f"  {'lead':5s} {'candidate':28s} {'height m':>9s} {'speed kt':>9s} {'vec m/s':>8s} {'div 1e-5':>9s}")
    for lead, cands in metrics["leads"].items():
        for name, m in cands.items():
            print(f"  {lead:5s} {name:28s} {m['height_m']:9.2f} {m['wind_speed_kt']:9.2f} {m['vector_wind_ms']:8.2f} "
                  f"{m['divergence_1e5']:9.3f}")


# ---------------------------------------------------------------------------------------------- training
def train(args) -> str:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    grid = get_domain(args.domain)
    print(f"[grid] {args.domain}: {grid.nx} x {grid.ny} points at {grid.dx / 1000:.0f} km "
          f"({grid.extent_km[0]:.0f} x {grid.extent_km[1]:.0f} km)")
    root = build_dataset(args, grid)
    inits = [t for t in list_samples(root, n_frames=args.hours + 1) if args.start <= t <= args.end]
    if len(inits) < 2:
        raise SystemExit(f"need at least 2 initialisations with data in {root}, found {len(inits)}")
    n_val = max(1, int(round(len(inits) * args.val_fraction)))
    train_inits, val_inits = inits[:-n_val], inits[-n_val:]          # chronological split
    print(f"[data] train {len(train_inits)} inits ({train_inits[0]:%Y-%m-%d %HZ} .. {train_inits[-1]:%Y-%m-%d %HZ}), "
          f"validate {len(val_inits)} ({val_inits[0]:%Y-%m-%d %HZ} .. {val_inits[-1]:%Y-%m-%d %HZ})")

    crop = min(args.crop, grid.nx, grid.ny)
    scales = CAMNetConfig().smoothing_scales_km
    train_ds = MapTrainingDataset(root, train_inits, n_frames=args.hours + 1, crop=crop,
                                  samples_per_init=args.crops_per_init, smoothing_scales_km=scales)
    eval_inits = val_inits
    if args.val_samples and len(val_inits) > args.val_samples:      # evenly spaced subset (full-domain eval is slow)
        eval_inits = [val_inits[i] for i in np.linspace(0, len(val_inits) - 1, args.val_samples).round().astype(int)]
    val_ds = MapTrainingDataset(root, eval_inits, n_frames=args.hours + 1, crop=None)
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=False)

    config = CAMNetConfig(levels=tuple(args.levels), dx=grid.dx, width=args.width, hidden=args.hidden,
                          use_physics=not args.no_physics, physics_substeps=args.substeps, map_level=args.level)
    model = UpperAirCAMNet(config).to(device)
    stats = compute_normalization(root, train_inits, args.levels, grid.dx)
    model.set_normalization(stats)
    coeff_params = list(model.physics.parameters())
    net_params = [p for n, p in model.named_parameters() if not n.startswith("physics.")]
    opt = torch.optim.AdamW([{"params": net_params, "lr": args.lr, "weight_decay": 1e-4},
                             {"params": coeff_params, "lr": args.coeff_lr, "weight_decay": 0.0}])
    total_steps = max(1, args.epochs * len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.coeff_lr], total_steps=total_steps,
                                                pct_start=0.15)
    print(f"[model] {sum(p.numel() for p in net_params) / 1e6:.2f}M network weights, "
          f"{sum(p.numel() for p in coeff_params)} adaptive equation coefficients, device {device}")

    ckpt = args.checkpoint or default_checkpoint(args.truth, args.domain)
    baseline = evaluate(model, val_ds, args.hours, device)
    print("[eval] untrained model (analysis = smoothed first guess, forecast = physics core only):")
    print_metrics(baseline)
    best, history = float("inf"), []
    step = 0
    for epoch in range(1, args.epochs + 1):
        t0, running, parts_sum = time.time(), 0.0, {}
        for batch in loader:
            b = to_device(batch, device)
            forced = random.random() < args.teacher_forcing
            out = model(b["obs"], b["dist"], b["geo"], b["init_hour"], b["day_of_year"], n_steps=args.hours,
                        start_state=b["truth"][:, 0] if forced else None, smoothed=b.get("smoothed"))
            loss, parts = map_model_loss(model, out, b["truth"], coefficient_prior=args.coeff_prior,
                                         teacher_forced=forced)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            running += float(loss)
            for key in ("state", "divergence"):
                parts_sum[key] = parts_sum.get(key, 0.0) + parts[key]
        nb = max(1, len(loader))
        metrics = evaluate(model, val_ds, args.hours, device)
        history.append({"epoch": epoch, "train_loss": running / nb, "val_loss": metrics["loss"]})
        print(f"[epoch {epoch}/{args.epochs}] train {running / nb:.4f} (state {parts_sum['state'] / nb:.4f}, "
              f"div {parts_sum['divergence'] / nb:.4f})  val {metrics['loss']:.4f}  {time.time() - t0:.0f}s")
        if metrics["loss"] < best:
            best = metrics["loss"]
            meta = {"truth_source": args.truth, "obs_source": args.obs, "domain": args.domain, "grid": grid.to_dict(),
                    "trained_hours": args.hours, "epoch": epoch, "synthetic": "synthetic" in (args.truth, args.obs),
                    "train_inits": {"first": train_inits[0].isoformat(), "last": train_inits[-1].isoformat(),
                                    "count": len(train_inits)},
                    "val_inits": {"first": val_inits[0].isoformat(), "last": val_inits[-1].isoformat(),
                                  "count": len(val_inits)},
                    "validation": metrics, "untrained_validation": baseline, "history": history,
                    "normalization": {k: v.tolist() for k, v in stats.items()}}
            save_checkpoint(model, ckpt, meta)
            with open(os.path.splitext(ckpt)[0] + ".json", "w") as fh:
                json.dump({**meta, "coefficients": model.coefficient_table()}, fh, indent=1)
    print(f"[done] best checkpoint -> {ckpt}")
    final = json.load(open(os.path.splitext(ckpt)[0] + ".json"))
    print_metrics(final["validation"])
    print_coefficients(final["coefficients"])
    return ckpt


def print_coefficients(table: dict, level: str = "300 hPa") -> None:
    print(f"[equations] learned term coefficients at {level} (1.000 = textbook equation):")
    for eq, info in table.items():
        terms = ", ".join(f"{t}={vals.get(level, list(vals.values())[0]):.3f}" for t, vals in info["terms"].items())
        print(f"  {eq:16s} {terms}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", default="hrrr", help="hrrr | nam3k | awips[:MODEL] | synthetic")
    p.add_argument("--obs", default=None, help="wyoming (real radiosondes) | synthetic; default: synthetic only "
                                               "when --truth synthetic")
    p.add_argument("--domain", default="central_plains", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--start", type=parse_time, required=True)
    p.add_argument("--end", type=parse_time, required=True)
    p.add_argument("--hours", type=int, default=2, help="forecast hours to train (frames f00..fN); goal: 2")
    p.add_argument("--levels", type=int, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--level", type=int, default=300, help="map level (extra loss weight + verification)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--crop", type=int, default=192, help="training crop size in 3 km grid points")
    p.add_argument("--crops-per-init", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--coeff-lr", type=float, default=5e-3, help="learning rate of the equation coefficients")
    p.add_argument("--coeff-prior", type=float, default=1e-3, help="pull of the coefficients toward 1 (textbook)")
    p.add_argument("--teacher-forcing", type=float, default=0.5,
                   help="probability a batch starts its forecast from the true f00 instead of the analysis")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--substeps", type=int, default=2)
    p.add_argument("--no-physics", action="store_true", help="pure network (no physics core) for comparison")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--val-samples", type=int, default=12, help="max validation inits scored each epoch (0 = all)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--mirror", nargs="+", default=None, help="GRIB mirrors to try in order (aws google nomads)")
    p.add_argument("--keep-grib", action="store_true", help="keep the downloaded GRIB subsets")
    p.add_argument("--skip-build", action="store_true", help="use the cached training samples only")
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args(argv)
    if args.obs is None:
        args.obs = "synthetic" if args.truth == "synthetic" else "wyoming"
    if args.hours < 1 or args.hours > 12:
        p.error("--hours must be 1..12 (training frames come from f00-f12 of 00Z/12Z runs)")
    if args.hours > 2:
        print(f"[note] --hours {args.hours}: beyond the current 2-hour goal; expect slower, less tested training")
    args.levels = sorted(set(args.levels), reverse=True)
    if args.level not in args.levels:
        p.error(f"--level {args.level} must be one of --levels {args.levels}")
    if (args.truth == "synthetic") != (args.obs == "synthetic"):
        p.error("synthetic truth must be paired with synthetic observations (and real with real)")
    train(args)


if __name__ == "__main__":
    main()
