"""
Point-Forecast-NN.py
====================
Skew-T based point-forecast neural network + the service layer the API/HUD talk to.

Pipeline:  Data-Fetch (Wyoming Upper Air) -> Data-Cleaning -> Data-Injestor -> PointForecastNN

* ``PointForecastNN``   - one network ("node") per forecast component (dewpoint, temperature, ...).
  Input: the latest sounding column (surface + mandatory levels), its 12-h tendency, and time/lead
  features. Output: the change of the component at every row at ``init + lead`` hours. Predicting the
  *change* means an untrained network (zeroed output layer) equals a persistence forecast, and the change
  is multiplied by lead/24 so it is exactly 0 at lead 0 and the network learns a rate of change.
* ``ComponentRouter``   - turns free text such as "dew point +2 HRs at 850mb" into
  {component: 'dewpoint', lead_hours: 2, level: '850 hPa'} (keyword/synonym matching).
* ``PointForecaster``   - in-between class used by API/api.py and src/app.py. Resolves the component,
  fetches + cleans + ingests soundings, loads the right checkpoint, runs the NN, returns JSON-ready dicts.

CLI:
  python Models/Point-Forecast-NN.py train --source wyoming --station 72451 --start 2024-01-01 --end 2024-06-30
  python Models/Point-Forecast-NN.py train --source synthetic --components dewpoint temperature
  python Models/Point-Forecast-NN.py forecast "dewpoint +2 hrs" --station 72451 --source synthetic
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import re
import sys
import threading

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, "Models", "checkpoints")


def load_module(name: str, relative_path: str):
    """Import a module from a file path (needed because the project's file names contain hyphens)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT_ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_fetch = load_module("data_fetch", os.path.join("Database_Training", "Data-Fetch.py"))
data_cleaning = load_module("data_cleaning", os.path.join("Database_Training", "Data-Cleaning.py"))
data_injestor = load_module("data_injestor", os.path.join("Database_Training", "Data-Injestor.py"))
inj = data_injestor

COMPONENTS = {
    "temperature": {"label": "Temperature", "units": "°C", "synonyms": [
        "temperature", "temperatures", "temp", "temps", "tmp", "tmpc", "air temperature", "thermal", "heat",
        "warm", "cold", "high", "low"]},
    "dewpoint": {"label": "Dewpoint", "units": "°C", "synonyms": [
        "dewpoint", "dewpoints", "dew point", "dew points", "dew pt", "dew", "td", "dwpt", "dwpc", "moisture"]},
    "relative_humidity": {"label": "Relative Humidity", "units": "%", "synonyms": [
        "relative humidity", "humidity", "rh", "relh", "saturation"]},
    "mixing_ratio": {"label": "Mixing Ratio", "units": "g/kg", "synonyms": [
        "mixing ratio", "mixing", "mixr", "specific humidity", "water vapor", "water vapour", "pwat"]},
    "wind_speed": {"label": "Wind Speed", "units": "kt", "synonyms": [
        "wind speed", "windspeed", "wind", "winds", "speed", "sknt", "wspd", "knots", "kts", "jet"]},
    "wind_direction": {"label": "Wind Direction", "units": "°", "synonyms": [
        "wind direction", "wind dir", "direction", "drct", "wdir", "veering", "backing", "veer", "back"]},
    "height": {"label": "Geopotential Height", "units": "m", "synonyms": [
        "geopotential height", "geopotential", "height", "heights", "hght", "gph", "thickness"]},
}
MAX_LEAD_HOURS = inj.MAX_LEAD_HOURS


# ====================================================================== keyword router
class ComponentRouter:
    """Maps forecaster wording to a component, a lead time and (optionally) a level."""

    LEAD_PATTERNS = ((r"(?<![\w.])\+?\s*(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", 1.0),
                     (r"(?<![\w.])\+?\s*(\d+(?:\.\d+)?)\s*(?:d|day|days)\b", 24.0),
                     (r"\+\s*(\d+(?:\.\d+)?)\b", 1.0))
    LEVEL_PATTERN = r"\b(1000|925|850|700|500|400|300|250|200)\s*(?:hpa|mb|millibars?)?\b"

    def __init__(self, components: dict = COMPONENTS):
        self.phrases = sorted(((syn, cid) for cid, meta in components.items() for syn in meta["synonyms"] + [cid]),
                              key=lambda item: -len(item[0]))

    @staticmethod
    def _normalise(text: str) -> str:
        return " " + re.sub(r"[^a-z0-9+.]+", " ", (text or "").lower().replace("_", " ")) + " "

    def parse_lead_hours(self, text) -> float | None:
        if text is None or text == "":
            return None
        if isinstance(text, (int, float)):
            return float(text)
        cleaned = str(text).lower()
        if re.fullmatch(r"\s*\+?\s*\d+(\.\d+)?\s*", cleaned):
            return float(cleaned.replace("+", ""))
        for pattern, factor in self.LEAD_PATTERNS:
            match = re.search(pattern, cleaned)
            if match:
                return float(match.group(1)) * factor
        return None

    def parse_level(self, text: str) -> str | None:
        lowered = (text or "").lower()
        if re.search(r"\b(surface|sfc|ground|2 ?m)\b", lowered):
            return "SFC"
        match = re.search(self.LEVEL_PATTERN, re.sub(r"\+?\s*\d+(\.\d+)?\s*(h|hr|hrs|hours?)\b", " ", lowered))
        return f"{match.group(1)} hPa" if match else None

    def resolve(self, text: str) -> tuple[str | None, str | None]:
        haystack = self._normalise(re.sub(r"\+?\s*\d+(\.\d+)?\s*(h|hr|hrs|hours?|d|days?)\b", " ", (text or "").lower()))
        for phrase, cid in self.phrases:
            if f" {phrase} " in haystack:
                return cid, phrase
        return None, None

    def parse(self, text: str) -> dict:
        component, matched = self.resolve(text)
        return {"query": text, "component": component, "matched_keyword": matched,
                "label": COMPONENTS[component]["label"] if component else None,
                "lead_hours": self.parse_lead_hours(text), "level": self.parse_level(text)}


# ====================================================================== network
class PointForecastNN(nn.Module):
    """1-D CNN over the sounding column + MLP head that also sees the scalar (time/lead) features."""

    def __init__(self, in_channels: int, n_rows: int, n_scalars: int, out_channels: int,
                 width: int = 64, hidden: int = 256, dropout: float = 0.1, lead_scaled: bool = True):
        super().__init__()
        self.out_channels, self.n_rows, self.lead_scaled = out_channels, n_rows, lead_scaled
        self.column = nn.Sequential(nn.Conv1d(in_channels, width, 3, padding=1), nn.GELU(),
                                    nn.Conv1d(width, width, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width * n_rows + n_scalars, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden // 2), nn.GELU(),
                                  nn.Linear(hidden // 2, out_channels * n_rows))
        nn.init.zeros_(self.head[-1].weight)  # start as a persistence forecast
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x_prof: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        z = self.column(x_prof).flatten(1)
        rate = self.head(torch.cat([z, x_scalar], dim=1)).view(-1, self.out_channels, self.n_rows)
        if self.lead_scaled:  # x_scalar[:, 0] = lead/24: the change is forced to 0 at lead 0 and grows with lead
            return rate * x_scalar[:, :1, None]
        return rate


def _masked_mse(pred, target, mask):
    m = mask[:, None, :].expand_as(pred)
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1.0)


def _physical_errors(component, base, delta_pred, delta_true, mask) -> dict:
    """Validation MAE (physical units) of the NN vs persistence, at the surface and over all valid rows."""
    if component == "wind_direction":
        def direction(d):
            return inj.wind_direction(base[:, 0] + d[:, 0], base[:, 1] + d[:, 1])
        truth = direction(delta_true)
        err_nn = np.abs((direction(delta_pred) - truth + 180) % 360 - 180)
        err_pers = np.abs((direction(np.zeros_like(delta_true)) - truth + 180) % 360 - 180)
    else:
        err_nn, err_pers = np.abs(delta_pred - delta_true)[:, 0], np.abs(delta_true)[:, 0]
    m = mask.astype(bool)
    return {"mae_nn_all": float(err_nn[m].mean()), "mae_persistence_all": float(err_pers[m].mean()),
            "mae_nn_sfc": float(err_nn[:, 0].mean()), "mae_persistence_sfc": float(err_pers[:, 0].mean())}


def train_component(component: str, soundings, source: str, stations, epochs: int = 40, lr: float = 2e-3,
                    batch_size: int = 256, val_fraction: float = 0.2, seed: int = 0,
                    checkpoint_root: str = CHECKPOINT_ROOT, verbose: bool = True) -> dict:
    """Train one component node on cleaned (time, df) soundings and save ``<root>/<source>/<component>.pt``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    injestor = inj.SoundingInjestor()
    data = injestor.build_training_set(soundings, component)
    times = sorted(set(data["init_times"]))
    cutoff = times[int(len(times) * (1 - val_fraction))] if len(times) > 4 else times[-1] + datetime.timedelta(1)
    is_val = np.array([t >= cutoff for t in data["init_times"]])  # chronological split: no look-ahead leakage
    if is_val.all() or not is_val.any():
        is_val = np.zeros(len(is_val), dtype=bool)
        is_val[-max(1, len(is_val) // 5):] = True
    stats = inj.normalization_stats(data["x_prof"][~is_val], data["y_delta"][~is_val], data["mask"][~is_val])

    def tensors(sel):
        return (torch.from_numpy((data["x_prof"][sel] - stats["x_mean"]) / stats["x_std"]),
                torch.from_numpy(data["x_scalar"][sel]),
                torch.from_numpy(data["y_delta"][sel] / stats["y_std"]),
                torch.from_numpy(data["mask"][sel].astype(np.float32)))

    train_t, val_t = tensors(~is_val), tensors(is_val)
    config = {"in_channels": injestor.n_channels, "n_rows": injestor.n_rows, "n_scalars": inj.N_SCALARS,
              "out_channels": len(inj.TARGET_VARS[component]), "lead_scaled": True}
    model = PointForecastNN(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history, best_state, best_val, best_epoch = [], None, float("inf"), 0
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for idx in torch.randperm(len(train_t[0])).split(batch_size):  # in-memory batching (fast on CPU)
            xp, xs, y, m = (t[idx] for t in train_t)
            optimizer.zero_grad()
            loss = _masked_mse(model(xp, xs), y, m)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xp)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = _masked_mse(model(val_t[0], val_t[1]), val_t[2], val_t[3]).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_t[0]), "val_loss": val_loss})
        if val_loss < best_val:  # keep the best-validation weights (early-stopping style)
            best_val, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            print(f"  [{component}] epoch {epoch + 1}/{epochs}  train {history[-1]['train_loss']:.4f}  val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(val_t[0], val_t[1]).numpy() * stats["y_std"]
    args = (data["base"][is_val], pred, data["y_delta"][is_val], data["mask"][is_val])
    metrics = _physical_errors(component, *args)
    native = ~data["interpolated"][is_val]  # verification against real (non-interpolated) soundings only
    if native.any():
        metrics.update({f"{k}_native": v for k, v in
                        _physical_errors(component, *(a[native] for a in args)).items()})
    metrics.update({"best_epoch": best_epoch, "val_samples": int(is_val.sum()), "train_samples": int((~is_val).sum()),
                    "final_train_loss": history[-1]["train_loss"], "final_val_loss": history[-1]["val_loss"]})
    checkpoint = {"state_dict": model.state_dict(), "config": config, "stats": stats, "component": component,
                  "source": source, "stations": list(stations), "levels": list(inj.LEVELS_HPA),
                  "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  "date_range": [times[0].isoformat(), times[-1].isoformat()], "epochs": epochs,
                  "n_soundings": data["n_soundings"], "n_native_pairs": data["n_native"],
                  "n_interpolated_pairs": data["n_interpolated"], "metrics": metrics}
    path = os.path.join(checkpoint_root, source, f"{component}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)
    if verbose:
        print(f"  [{component}] saved {path}\n  [{component}] validation MAE  NN {metrics['mae_nn_all']:.3f} vs "
              f"persistence {metrics['mae_persistence_all']:.3f} {COMPONENTS[component]['units']}")
    return {"component": component, "path": path, "metrics": metrics, "history": history,
            "n_native_pairs": data["n_native"], "n_interpolated_pairs": data["n_interpolated"]}


# ====================================================================== service layer
def _num(value, digits: int = 2):
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, digits)


def parse_init_time(value) -> datetime.datetime | None:
    if value in (None, "", "latest"):
        return None
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip().upper().rstrip("Z")
    if re.fullmatch(r"\d{10}", text):
        return datetime.datetime.strptime(text, "%Y%m%d%H")
    return datetime.datetime.fromisoformat(text).replace(tzinfo=None)


class ForecastError(Exception):
    """Raised for user-facing problems (bad request, no data). ``status`` is an HTTP-style code."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class PointForecaster:
    """In-between class: the API/HUD call ``forecast()`` and get JSON-serialisable dicts back."""

    def __init__(self, checkpoint_root: str = CHECKPOINT_ROOT, fetchers: dict | None = None):
        self.checkpoint_root = checkpoint_root
        self.router = ComponentRouter()
        self.cleaner = data_cleaning.SoundingCleaner(required_top_hpa=float(min(inj.LEVELS_HPA)))
        self.injestor = inj.SoundingInjestor()
        self._fetchers = dict(fetchers or {})
        self._models: dict = {}
        self._lock = threading.Lock()

    def fetcher(self, source: str):
        if source not in self._fetchers:
            self._fetchers[source] = data_fetch.get_fetcher(source)
        return self._fetchers[source]

    def checkpoint_path(self, component: str, source: str) -> str:
        return os.path.join(self.checkpoint_root, source, f"{component}.pt")

    def load_model(self, component: str, source: str):
        path = self.checkpoint_path(component, source)
        if not os.path.exists(path):
            return None, None
        key = (component, source, os.path.getmtime(path))
        with self._lock:
            if key not in self._models:
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:  # torch < 1.13 has no weights_only argument
                    ckpt = torch.load(path, map_location="cpu")
                model = PointForecastNN(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._models[key] = (model, ckpt)
            return self._models[key]

    def model_info(self, component: str, source: str) -> dict:
        _, ckpt = self.load_model(component, source)
        if ckpt is None:
            return {"status": "untrained", "note": "No checkpoint found - showing a persistence forecast. Train with: "
                    f"python Models/Point-Forecast-NN.py train --source {source} --components {component}"}
        keys = ("source", "stations", "trained_at", "date_range", "epochs", "n_soundings", "n_native_pairs",
                "n_interpolated_pairs", "metrics")
        return {"status": "trained", **{k: ckpt.get(k) for k in keys}}

    def list_components(self, source: str = "wyoming") -> list[dict]:
        return [{"id": cid, "label": meta["label"], "units": meta["units"], "synonyms": meta["synonyms"],
                 "model": self.model_info(cid, source)["status"]} for cid, meta in COMPONENTS.items()]

    def resolve(self, query: str) -> dict:
        return self.router.parse(query)

    # ------------------------------------------------------------------ main entry point
    def forecast(self, component: str | None = None, lead_hours=None, station: str = "72451",
                 source: str = "wyoming", init_time=None, query: str | None = None, level: str | None = None) -> dict:
        parsed = self.router.parse(query) if query else {}
        if component and component not in COMPONENTS:
            component = self.router.resolve(component)[0] or component
        component = component or parsed.get("component")
        if component not in COMPONENTS:
            raise ForecastError(f"Could not determine a forecast component from {query or component!r}. "
                                f"Try one of: {', '.join(m['label'] for m in COMPONENTS.values())}.")
        lead = self.router.parse_lead_hours(lead_hours) if lead_hours not in (None, "") else parsed.get("lead_hours")
        if lead is None:
            raise ForecastError("No lead time given - use e.g. '+2 HRs' or lead_hours=2.")
        if not 1 <= lead <= MAX_LEAD_HOURS:
            raise ForecastError(f"Lead time must be between 1 and {MAX_LEAD_HOURS} hours (got {lead:g}).")
        source = (source or "wyoming").lower()
        if source not in ("wyoming", "synthetic"):
            raise ForecastError("source must be 'wyoming' or 'synthetic'.")
        station = str(station or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{3,6}", station):
            raise ForecastError("Station must be a WMO number (e.g. 72451) or identifier (e.g. DDC).")
        try:
            init = parse_init_time(init_time)
        except ValueError:
            raise ForecastError(f"Unrecognised init_time {init_time!r}; use ISO format e.g. 2024-05-12T12:00.")

        fetcher = self.fetcher(source)
        try:
            if init is None:
                raw = fetcher.fetch_latest(station, count=2)
            else:
                raw = [(t, fetcher.fetch(station, t)) for t in (init, init - datetime.timedelta(hours=12))]
                raw = [(t, df) for t, df in raw if df is not None]
        except data_fetch.UpstreamUnavailable as exc:
            raise ForecastError(f"Wyoming Upper Air is unreachable from this machine: {str(exc)[:200]}. "
                                "Check network access, or switch to the 'synthetic' source to test offline.", 503)
        soundings, rejected = self.cleaner.clean_many(raw)
        if not soundings:
            detail = getattr(fetcher, "last_error", None)
            raise ForecastError(f"No usable sounding for station {station} ({source})"
                                + (f" - {detail}" if detail else "") + ".", status=404)
        try:
            inputs = self.injestor.build_inference_input(soundings, lead)
        except ValueError as exc:
            raise ForecastError(str(exc), status=422)
        if init is not None and inputs["init_time"] != init:
            raise ForecastError(f"The {init:%Y-%m-%d %HZ} sounding for {station} was missing or failed QC.", 404)

        model, ckpt = self.load_model(component, source)
        current = inputs["current"]
        t_idx = [inj.PROFILE_VARS.index(v) for v in inj.TARGET_VARS[component]]
        base = current.values[:, t_idx].T                                   # [out_channels, rows]
        warnings = []
        if model is not None:
            stats = ckpt["stats"]
            with torch.no_grad():
                out = model(torch.from_numpy((inputs["x_prof"] - stats["x_mean"]) / stats["x_std"]),
                            torch.from_numpy(inputs["x_scalar"])).numpy()[0]
            delta = out * stats["y_std"]
            if station not in [s.upper() for s in ckpt.get("stations", [])]:
                warnings.append(f"Model was trained on {', '.join(ckpt.get('stations', [])) or 'other stations'}, "
                                f"not {station}; expect reduced skill.")
        else:
            delta = np.zeros_like(base)
            warnings.append("No trained model for this component/source yet - values are PERSISTENCE (no change).")
        if source == "synthetic":
            warnings.insert(0, "SYNTHETIC demo data - not a real observation or forecast.")
        if inputs["previous"] is None:
            warnings.append("Previous (12 h earlier) sounding unavailable; tendency features set to zero.")
        if lead % 12:
            warnings.append("Wyoming soundings are 00Z/12Z only: sub-12 h training targets were time-interpolated.")

        forecast_vals = base + delta
        if component == "wind_direction":
            observed = inj.wind_direction(base[0], base[1])
            predicted = inj.wind_direction(forecast_vals[0], forecast_vals[1])
            change = (predicted - observed + 180) % 360 - 180
        else:
            observed, predicted = base[0], forecast_vals[0]
            if component in ("relative_humidity",):
                predicted = np.clip(predicted, 0, 100)
            if component in ("mixing_ratio", "wind_speed"):
                predicted = np.clip(predicted, 0, None)
            change = predicted - observed
        pressures = [current.psfc] + list(inj.LEVELS_HPA)
        rows = [{"level": inj.ROW_LABELS[i], "pressure_hpa": _num(pressures[i], 1),
                 "observed": _num(observed[i]), "forecast": _num(predicted[i]), "change": _num(change[i]),
                 "below_ground": bool(current.below_ground[i])} for i in range(len(pressures))]
        wanted = level or parsed.get("level") or "SFC"
        headline = next((r for r in rows if r["level"] == wanted and not r["below_ground"]), rows[0])
        newest_df = max(soundings, key=lambda item: item[0])[1]
        meta = COMPONENTS[component]
        return {
            "component": component, "label": meta["label"], "units": meta["units"], "station": station,
            "source": source, "lead_hours": lead, "init_time": inputs["init_time"].isoformat() + "Z",
            "valid_time": inputs["valid_time"].isoformat() + "Z", "headline": headline, "profile": rows,
            "observed_sounding": {col: [_num(v, 1) for v in newest_df[col]] for col in
                                  ("PRES", "HGHT", "TEMP", "DWPT", "DRCT", "SKNT")},
            "model": self.model_info(component, source), "warnings": warnings,
            "soundings_used": [t.isoformat() + "Z" for t, _ in sorted(soundings, key=lambda i: i[0], reverse=True)],
            "rejected_soundings": [t.isoformat() + "Z" for t in rejected],
            "parsed_query": parsed or None,
        }


# ====================================================================== CLI
def _cli_train(args) -> None:
    components = list(COMPONENTS) if "all" in args.components else args.components
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        sys.exit(f"Unknown component(s): {unknown}. Options: {list(COMPONENTS)}")
    start, end = parse_init_time(args.start), parse_init_time(args.end)
    fetcher = data_fetch.get_fetcher(args.source)
    cleaner = data_cleaning.SoundingCleaner(required_top_hpa=float(min(inj.LEVELS_HPA)))
    soundings = []
    for station in args.station:
        print(f"Fetching {args.source} soundings for {station}: {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
        kept, rejected = cleaner.clean_many(fetcher.fetch_range(station, start, end))
        print(f"  {len(kept)} soundings passed QC, {len(rejected)} rejected")
        soundings.extend(kept)
    if len(soundings) < 10:
        sys.exit("Not enough soundings to train (need at least 10). Check the station/date range or network access.")
    for component in components:
        print(f"Training '{component}' node ...")
        train_component(component, soundings, args.source, args.station, epochs=args.epochs, lr=args.lr,
                        batch_size=args.batch_size, checkpoint_root=args.checkpoint_root)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Skew-T point-forecast neural network")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fetch soundings and train component node(s)")
    train.add_argument("--source", choices=["wyoming", "synthetic"], default="wyoming")
    train.add_argument("--station", nargs="+", default=["72451"], help="WMO number(s), e.g. 72451 (DDC)")
    train.add_argument("--start", default="2024-01-01")
    train.add_argument("--end", default="2024-06-30")
    train.add_argument("--components", nargs="+", default=["all"])
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--checkpoint-root", default=CHECKPOINT_ROOT)
    fc = sub.add_parser("forecast", help="run a forecast, print JSON")
    fc.add_argument("query", help="e.g. \"dewpoint +2 hrs\"")
    fc.add_argument("--station", default="72451")
    fc.add_argument("--source", choices=["wyoming", "synthetic"], default="wyoming")
    fc.add_argument("--init-time", default=None)
    args = parser.parse_args(argv)
    if args.command == "train":
        _cli_train(args)
    else:
        try:
            result = PointForecaster().forecast(query=args.query, station=args.station, source=args.source,
                                                init_time=args.init_time)
        except ForecastError as exc:
            sys.exit(f"Error: {exc}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
