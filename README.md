### Weather Prediction AI

## Overview

This repository was created to hold a project focused on long-term weather prediction. 
The primary goalis to develop a fully AI based NWP model capable of forecasting up to
three months out from initialization. This project aims to create an end-to-end pipeline
for AI-driven weather forecasting. The main approach to this is to have an AI that can
gather, process, and prepare its own datasets to train itself on.

## Install
1. **Clone the repository:**
```
git clone https://github.com/nb101bn/Weather-Prediction-AI.git`
cd Weather-Prediction-AI
```

2. **Create a virtual environment:**
```
python -m venv {environment name}`
 # On Windows:`
 .\venv\Scripts\activate`
 # On macOS/Linux`
 source {environment name}/bin/activate
``` 
*Replace '{Environment name}' with your desired name for the virtual environment.*

3. **Install the required dependencies:**
```
pip install -r requirements.txt
```

## Contact
[nb101bn](https://github.com/nb101bn)

---

## Skew-T Point Forecast HUD

A Flask HUD that asks a neural network for **point forecasts derived from Skew-T log-p soundings**.
Pick a component (e.g. *Dewpoint*) and a lead time (e.g. **+2 HRs**), or type it (`dewpoint +2 HRs`),
and the network forecasts that component at the surface and every mandatory level.

```
 HUD (src/app.py + css/js)  --JSON-->  FastAPI (API/api.py)  -->  PointForecaster (Models/Point-Forecast-NN.py)
                                                                  |- ComponentRouter: keywords/synonyms -> component, "+2 HRs" -> lead
                                                                  |- Data-Fetch     : Wyoming Upper Air soundings
                                                                  |- Data-Cleaning  : QC pipeline (expandable)
                                                                  |- Data-Injestor  : soundings -> model tensors
                                                                  '- PointForecastNN: one trained "node" per component
```

### Project structure
```
Weather-Prediction-AI/
|_ Models/
|  |_ AI_Processor.py          (existing surface-map CNN, unchanged - moved from Model/ to Models/)
|  |_ Point-Forecast-NN.py     NN + keyword router + PointForecaster service + train/forecast CLI
|  '_ checkpoints/<source>/<component>.pt   (created by training; git-ignored)
|_ Database_Training/
|  |_ Data-Fetch.py            WyomingUpperAir class (+ SyntheticUpperAir for offline demos)
|  |_ Data-Cleaning.py         SoundingCleaner - ordered QC steps, add your own with add_step()
|  |_ Data-Injestor.py         SoundingInjestor - interpolation to levels, training pairs, features
|  '_ (existing CoCoRaHS / surface-map / interpolator files unchanged)
|_ API/api.py                  FastAPI JSON endpoint
|_ src/
|  |_ app.py                   Flask HUD (`flask run`)
|  |_ templates/index.html     HUD page (Flask needs an HTML template)
|  |_ css/custom.css
|  '_ js/custom.js
|_ tests/test_point_forecast.py
'_ .flaskenv                   lets `flask run` find src/app.py from the project root
```
Train.py now imports `from Models.AI_Processor import CNN`.

### Run the HUD
```
pip install -r Requirements.txt
flask run                                  # from the project root (or from src/) -> http://127.0.0.1:5000
```
Optional - run the FastAPI service in a second terminal and the HUD will route through it
(the top-bar pill shows `BACKEND: FASTAPI`; without it the HUD runs the same code `IN-PROCESS`):
```
uvicorn API.api:app --port 8000            # or: python API/api.py   (docs at http://127.0.0.1:8000/docs)
```
Settings (environment or `.flaskenv`): `POINT_FORECAST_API_URL` (default `http://127.0.0.1:8000`),
`POINT_FORECAST_BACKEND` = `auto` | `fastapi` | `local`.

### Train the component nodes on Wyoming Upper Air data
```
python Models/Point-Forecast-NN.py train --source wyoming --station 72451 --start 2023-01-01 --end 2024-12-31
python Models/Point-Forecast-NN.py train --source wyoming --station 72451 72357 72440 --components dewpoint temperature
```
* Soundings are cached in `Database_Training/cache/wyoming/<station>/` so re-runs don't re-download.
  Requests are spaced 1.5 s apart (UWyo throttles rapid requests), so one station-year (~730 soundings) takes ~20-40 min the first time.
* Each component gets its own checkpoint, `Models/checkpoints/wyoming/<component>.pt`, with metadata: stations, date range,
  number of real vs time-interpolated training pairs, and validation MAE of the NN vs a persistence forecast.
* Components: `temperature, dewpoint, relative_humidity, mixing_ratio, wind_speed, wind_direction, height`.
* Untrained components still answer, with a **persistence** forecast (no change), and the HUD says so.

Offline demo (fake data, clearly flagged in the HUD and in every JSON reply):
`python Models/Point-Forecast-NN.py train --source synthetic --start 2022-01-01 --end 2023-12-31`

Command-line forecast (prints the same JSON as the API):
`python Models/Point-Forecast-NN.py forecast "dewpoint +2 hrs" --station 72451 --source wyoming`

### API
`POST /api/forecast`
```json
{"component": "dewpoint", "lead_hours": "+2 HRs", "station": "72451", "source": "wyoming", "init_time": null, "level": "SFC"}
```
or a free-text request: `{"query": "850mb dew point +6 hrs", "station": "72451"}`.
The reply includes `headline` (value at the requested level), `profile` (observed/forecast/change for SFC and
1000-200 hPa), `observed_sounding` (for the Skew-T), `init_time`/`valid_time`, `model` (status + verification) and `warnings`.
Other routes: `GET /api/health`, `GET /api/components?source=`, `GET /api/stations`, `POST /api/resolve {"query": ...}`.
Errors come back as `{"detail": "..."}` with 400 (bad request), 404 (no sounding), 422 (validation) or 503 (Wyoming unreachable).

### How the network works
* **Input:** the latest sounding interpolated (in ln p) to the surface + 1000/925/850/700/500/400/300/250/200 hPa,
  with T, Td, RH, mixing ratio, u, v, height, plus the 12-h tendency from the previous sounding and a below-ground mask.
  Scalars: lead time, time of day/year of the valid time, init hour, surface pressure.
* **Output:** the *change* of the chosen component at every level by `init + lead`. Wind direction is learned as u/v, so 350°→10° isn't treated as a 340° error.
  The last layer starts at zero, so an untrained network equals persistence, and training learns corrections to it.
  The change is also multiplied by lead/24, so it is exactly zero at lead 0 and the network learns a *rate* of change
  (without this, +1 h forecasts jumped almost as far as +12 h ones).
* **Validation:** a chronological split (the last 20 % of dates), reported against persistence both for all pairs and for real (non-interpolated) pairs only.

### Known limitations (read before trusting a number)
* **Wyoming soundings exist only at 00Z/12Z.** A +2 h target is a linear time-interpolation between the two soundings
  that bracket it, so the network cannot learn sub-12 h events such as the afternoon temperature peak. Leads of 12/24/36/48 h use real
  soundings. The natural next step is adding hourly surface observations (e.g. the IEM ASOS fetcher already in
  `Data-Interpolator.py`) as sub-12 h targets for the surface row.
* The Wyoming fetcher targets UWyo's current `wsgi/sounding` CSV interface, with a fallback to the legacy `cgi-bin/sounding`
  TEXT:LIST page. Both parsers are unit-tested against sample pages, but they could not be exercised against the live site
  from the development sandbox, so the first real training run is also the first live test. `fetcher.last_error` shows the failure reason.
* Skill depends entirely on training data volume; a few months from one station is a smoke test, not a model.

### Tests
`python -m unittest discover tests -v`
