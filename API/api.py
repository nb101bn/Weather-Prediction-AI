"""
api.py - FastAPI endpoint between the HUD (src/app.py) and the Point-Forecast NN.

Run from the project root (either works):
    uvicorn API.api:app --port 8000
    python API/api.py

Endpoints (all JSON):
    GET  /api/health                      -> service status
    GET  /api/components?source=wyoming   -> forecast components, synonyms, model status
    GET  /api/stations                    -> common RAOB stations for the HUD picker
    POST /api/resolve   {"query": "dew point +2 HRs"}
    POST /api/forecast  {"component": "dewpoint", "lead_hours": "+2 HRs", "station": "72451",
                         "source": "wyoming", "init_time": null, "query": null, "level": null}
Interactive docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, relative_path: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(PROJECT_ROOT, relative_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pfnn = _load("point_forecast_nn", os.path.join("Models", "Point-Forecast-NN.py"))
forecaster = pfnn.PointForecaster()

# WMO number, identifier, name - common CONUS radiosonde sites (any WMO number can still be typed in the HUD).
STATIONS = [
    ("72451", "DDC", "Dodge City, KS"), ("72357", "OUN", "Norman, OK"), ("72249", "FWD", "Fort Worth, TX"),
    ("72440", "SGF", "Springfield, MO"), ("74560", "ILX", "Lincoln, IL"), ("72649", "MPX", "Chanhassen, MN"),
    ("72469", "DNR", "Denver, CO"), ("72558", "OAX", "Omaha/Valley, NE"), ("72562", "LBF", "North Platte, NE"),
    ("72456", "TOP", "Topeka, KS"), ("72363", "AMA", "Amarillo, TX"), ("72265", "MAF", "Midland, TX"),
    ("72764", "BIS", "Bismarck, ND"), ("72632", "DTX", "White Lake, MI"), ("72520", "PIT", "Pittsburgh, PA"),
    ("72501", "OKX", "Upton, NY"), ("72403", "IAD", "Sterling, VA"), ("72215", "FFC", "Peachtree City, GA"),
    ("72233", "LIX", "Slidell, LA"), ("72210", "TBW", "Tampa Bay, FL"), ("72202", "MFL", "Miami, FL"),
    ("72572", "SLC", "Salt Lake City, UT"), ("72293", "NKX", "San Diego, CA"), ("72493", "OAK", "Oakland, CA"),
    ("72797", "UIL", "Quillayute, WA"),
]


# ---------------------------------------------------------------- plain functions (shared with the Flask HUD)
def health() -> dict:
    return {"status": "ok", "service": "point-forecast-nn", "max_lead_hours": pfnn.MAX_LEAD_HOURS,
            "components": list(pfnn.COMPONENTS), "sources": ["wyoming", "synthetic"]}


def components(source: str = "wyoming") -> dict:
    return {"source": source, "components": forecaster.list_components(source),
            "max_lead_hours": pfnn.MAX_LEAD_HOURS}


def stations() -> dict:
    return {"stations": [{"wmo": w, "id": i, "name": n} for w, i, n in STATIONS]}


def resolve(query: str) -> dict:
    return forecaster.resolve(query)


def forecast(payload: dict) -> dict:
    """Raises pfnn.ForecastError (with .status) on bad input / missing data."""
    allowed = ("component", "lead_hours", "station", "source", "init_time", "query", "level")
    return forecaster.forecast(**{k: v for k, v in (payload or {}).items() if k in allowed and v is not None})


def dispatch(method: str, path: str, payload: dict | None = None, params: dict | None = None) -> tuple[int, dict]:
    """Route a request without HTTP - used by src/app.py when the FastAPI server is not running."""
    params = params or {}
    try:
        if method == "GET" and path == "/api/health":
            return 200, health()
        if method == "GET" and path == "/api/components":
            return 200, components(params.get("source", "wyoming"))
        if method == "GET" and path == "/api/stations":
            return 200, stations()
        if method == "POST" and path == "/api/resolve":
            return 200, resolve((payload or {}).get("query", ""))
        if method == "POST" and path == "/api/forecast":
            return 200, forecast(payload or {})
        return 404, {"detail": f"unknown route {method} {path}"}
    except pfnn.ForecastError as exc:
        return exc.status, {"detail": str(exc)}


# ---------------------------------------------------------------- FastAPI app
app = FastAPI(title="Skew-T Point Forecast API", version="1.0.0",
              description="JSON interface to the Wyoming-sounding point-forecast neural network.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResolveRequest(BaseModel):
    query: str = Field(..., examples=["dew point +2 HRs"])


class ForecastRequest(BaseModel):
    component: Optional[str] = Field(None, examples=["dewpoint"])
    lead_hours: Optional[Union[float, str]] = Field(None, examples=["+2 HRs"])
    station: str = Field("72451", examples=["72451"])
    source: Literal["wyoming", "synthetic"] = "wyoming"
    init_time: Optional[str] = Field(None, description="ISO time of the init sounding; omit for latest")
    query: Optional[str] = Field(None, examples=["dewpoint +2 hrs at 850mb"])
    level: Optional[str] = Field(None, examples=["SFC", "850 hPa"])


@app.get("/api/health")
def get_health():
    return health()


@app.get("/api/components")
def get_components(source: Literal["wyoming", "synthetic"] = Query("wyoming")):
    return components(source)


@app.get("/api/stations")
def get_stations():
    return stations()


@app.post("/api/resolve")
def post_resolve(request: ResolveRequest):
    return resolve(request.query)


@app.post("/api/forecast")
def post_forecast(request: ForecastRequest):  # sync def -> FastAPI runs it in a worker thread
    try:
        return forecast(request.model_dump())
    except pfnn.ForecastError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8000")))
