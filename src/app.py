"""
app.py - Flask HUD for the Skew-T point-forecast network.

Start it with `flask run` (from the project root - .flaskenv points Flask at src/app.py - or from src/).

The HUD's JavaScript talks only to the /hud/* routes below. Each route forwards the request to the
FastAPI service (API/api.py, default http://127.0.0.1:8000, override with POINT_FORECAST_API_URL).
If that service isn't running, the same API code is executed in-process so `flask run` alone is
enough. Every JSON reply carries "backend": "fastapi" | "in-process" so you can see which path served it.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
API_URL = os.environ.get("POINT_FORECAST_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT = float(os.environ.get("POINT_FORECAST_API_TIMEOUT", "180"))

app = Flask(__name__, template_folder=os.path.join(SRC_DIR, "templates"), static_folder=None)
_local_api = None


def local_api():
    """Load API/api.py lazily (it imports torch) and reuse it for in-process requests."""
    global _local_api
    if _local_api is None:
        name = "point_forecast_api"
        if name in sys.modules:
            _local_api = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT_ROOT, "API", "api.py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            _local_api = module
    return _local_api


def call_backend(method: str, path: str, payload: dict | None = None, params: dict | None = None):
    mode = os.environ.get("POINT_FORECAST_BACKEND", "auto").lower()  # auto | fastapi | local
    if mode != "local":
        try:
            response = requests.request(method, f"{API_URL}{path}", json=payload, params=params,
                                        timeout=(2, API_TIMEOUT))
            try:
                body = response.json()
            except ValueError:
                body = {"detail": f"FastAPI returned non-JSON ({response.status_code})"}
            if isinstance(body, dict):
                body["backend"] = "fastapi"
            return jsonify(body), response.status_code
        except requests.exceptions.ConnectionError:
            if mode == "fastapi":
                return jsonify({"detail": f"FastAPI service not reachable at {API_URL}", "backend": "none"}), 503
        except requests.exceptions.Timeout:
            return jsonify({"detail": "FastAPI service timed out", "backend": "fastapi"}), 504
    status, body = local_api().dispatch(method, path, payload, params)
    body["backend"] = "in-process"
    return jsonify(body), status


@app.get("/")
def index():
    return render_template("index.html", api_url=API_URL)


@app.get("/css/<path:filename>")
def css(filename):
    return send_from_directory(os.path.join(SRC_DIR, "css"), filename)


@app.get("/js/<path:filename>")
def js(filename):
    return send_from_directory(os.path.join(SRC_DIR, "js"), filename)


@app.get("/hud/health")
def hud_health():
    return call_backend("GET", "/api/health")


@app.get("/hud/components")
def hud_components():
    return call_backend("GET", "/api/components", params={"source": request.args.get("source", "wyoming")})


@app.get("/hud/stations")
def hud_stations():
    return call_backend("GET", "/api/stations")


@app.post("/hud/resolve")
def hud_resolve():
    return call_backend("POST", "/api/resolve", payload=request.get_json(silent=True) or {})


@app.post("/hud/forecast")
def hud_forecast():
    return call_backend("POST", "/api/forecast", payload=request.get_json(silent=True) or {})


if __name__ == "__main__":
    app.run(debug=True)
