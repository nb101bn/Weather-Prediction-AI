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

---

## Upper-Air 3 km CAM Map Model (300 mb)

A CNN + RNN network on a **3 km convection-allowing grid** that steps the atmosphere forward **hourly**
(architecture: up to 48 h; **current goal and trained range: F00-F02**) and draws an isoplethed **300 mb map**:
heights, shaded isotachs, wind barbs and divergence.

```
 radiosondes 00Z/12Z (Wyoming archive, Data-Fetch.py + Data-Cleaning.py)
        |  nearest-neighbour interpolation to the 3 km grid  (Grid_Interpolation.py)
        v
 AnalysisNet  (CNN U-Net)        NN-gridded obs + the same smoothed at 50/100/200 km + distance-to-obs
        |                        + static fields -> correction to the 100 km first guess = 3 km analysis (F00)
        |
        v   every hour:
 PhysicsCore  (Atmos_Physics.py) adaptive primitive equations (divergence, vorticity, thermodynamic,
        |                        height tendency) - every term has a learnable coefficient
        v
 ForecastCell (CNN encoder -> ConvGRU -> CNN decoder) corrects the physics first guess  ->  F01, F02, ...
        |
        v
 Maps/Upper_Air_Map.py           300 mb: heights (dam), isotachs (kt, shaded), barbs (kt), divergence (1e-5 s-1)

 Training "ground truth": HRRR / NAM 3 km GRIB2 (NOAA archives, byte-range subsets) or AWIPS EDEX grids,
 ONLY from the 00Z/12Z (observed) initialisations and ONLY forecast hours 00-12.
```

### New files
```
Models/
|_ Upper_Air_CAM_Net.py    THE MODEL: AnalysisNet (U-Net), ForecastCell (ConvGRU), loss, checkpoints.
|                          Import it anywhere:  from Models.Upper_Air_CAM_Net import UpperAirCAMNet, load_checkpoint
|_ Atmos_Physics.py        equations: thermodynamics (theta, theta-e, RH, dewpoint, q), kinematics (divergence,
|                          vorticity, omega, Jacobian, Laplacian), Poisson solver, semi-Lagrangian advection,
|                          AdaptiveEquation + PhysicsCore
'_ Model_Grid.py           the 3 km Lambert-conformal grid (= the operational HRRR CONUS grid) + domain presets
Database_Training/
|_ Upper_Air_Obs.py        radiosonde observations (reuses WyomingUpperAir + SoundingCleaner), CONUS RAOB table
|_ Grid_Interpolation.py   nearest-neighbour obs -> grid, model grid -> our grid, wind rotation
|_ Model_Grib_Fetch.py     HRRR / NAM-3k GRIB2 byte-range downloader (AWS / Google / NOMADS), AWIPS EDEX fetcher,
|                          the 00Z/12Z + F00-F12 training rules
|_ Map_Training_Data.py    builds/caches training samples, torch Dataset with random 3 km crops, normalisation
'_ Synthetic_Atmosphere.py fake upper troposphere for offline tests/demos (always labelled SYNTHETIC)
Maps/Upper_Air_Map.py      the 300 mb chart renderer
Train_Upper_Air_Maps.py    training CLI
Forecast_Upper_Air_Maps.py forecast + map CLI (deployment)
tests/test_upper_air_model.py, tests/fixtures/*.grib2 (small real GRIB2 files for the decoder tests)
```

### Train on real data
```
pip install -r Requirements.txt      # Windows: if `pip install pygrib` fails, `conda install -c conda-forge pygrib`
# HRRR truth (NOAA open data on AWS, falls back to Google then NOMADS) + Wyoming radiosondes, central-plains window
python Train_Upper_Air_Maps.py --truth hrrr --obs wyoming --start 2024-04-01 --end 2024-06-30
# NAM CONUS nest (3 km) instead:            --truth nam3k
# grids straight from an AWIPS EDEX server:  --truth awips:HRRR   (real time only: EDEX keeps ~1 day)
```
* **Only 00Z and 12Z initialisations** are used (radiosonde times), and **only F00-F12** frames (`check_training_frame`
  raises otherwise). `--hours 2` (default) trains F00, F01, F02.
* For every init the builder grids the radiosondes near the domain (~40 for `central_plains`, all 70 CONUS sites for
  `conus`) and downloads just HGT/UGRD/VGRD/TMP/RH at 500/400/300/250/200/150/100 mb (35 GRIB messages - tens of MB
  per forecast hour instead of the few-hundred-MB `wrfprs` file) using the `.idx` inventories and HTTP Range requests.
* Archives: NOAA's open-data buckets (AWS, then Google) hold the complete HRRR archive (2014-) and NOMADS the last few
  days. Any other server with NCEP's directory layout (e.g. an NCAR/RDA or in-house mirror) can be used with
  `--mirror https://your.mirror/path` (mirrors are tried in the order given).
  Samples are cached in `Database_Training/cache/map_training/` (the raw GRIB subsets are deleted unless `--keep-grib`).
* Domains (`--domain`): `central_plains` (512 x 448 points, default), `southern_plains`, `northeast`, `demo`, `conus`
  (full 1799 x 1059 HRRR grid). Training uses random crops (`--crop`), so the fully-convolutional model runs on any domain.
* Wyoming throttles requests (1.5 s apart, cached), so the first build of an init takes ~2 min; HRRR downloads dominate after that.
* **Disk:** samples are float32 `.npy` (memory-mapped for fast random crops). `central_plains` with F00-F02 is ~135 MB per
  init, i.e. ~100 GB per year of 00Z/12Z runs (`conus` is ~8x that). float16 is not an option: its ~0.5 m resolution of
  300 mb heights would put ~16 m/s of noise into 3 km geostrophic winds.
* A GPU is strongly recommended for real training (`--device cuda` is automatic when available). Default network:
  13 M weights (`--width 32 --hidden 96`).
* Output: `Models/checkpoints/upper_air/cam_300mb_<truth>_<domain>.pt` plus a `.json` report with validation scores and the
  learned equation coefficients.

### Make the 300 mb maps
```
python Forecast_Upper_Air_Maps.py --init 2024-06-01T12                 # newest checkpoint, F00-F02
python Forecast_Upper_Air_Maps.py --init 2024-06-01T12 --truth hrrr    # + HRRR maps and RMSE verification
```
Maps, an `.npz` with the 300 mb fields (height, grid-relative u/v, temperature, RH, wind speed, divergence, vorticity,
absolute vorticity, omega, theta, theta-e) and `summary.json` go to
`Maps/output/<YYYYMMDDHH>/`. On a constant-pressure chart the "pressure lines" are **height contours** (pressure is
300 mb everywhere on that surface; isohypses play the role isobars play on a surface map). Divergence is drawn smoothed
over 20 km because raw 3 km divergence is dominated by grid-scale noise. Leads beyond the trained range are allowed
(`--hours` up to 48) but every such map is stamped *BEYOND THE TRAINED RANGE*.

### Offline demo (no internet needed)
```
python Train_Upper_Air_Maps.py --truth synthetic --domain demo --start 2024-04-01 --end 2024-05-31T12 \
    --epochs 14 --width 16 --hidden 48 --crop 128 --crops-per-init 2 --batch 2 --val-samples 6
python Forecast_Upper_Air_Maps.py --checkpoint Models/checkpoints/upper_air/cam_300mb_synthetic_demo.pt \
    --init 2024-05-25T12 --truth synthetic
```
Synthetic data is a made-up atmosphere (see `Synthetic_Atmosphere.py`); its maps carry a red *SYNTHETIC DATA* banner.

Result of that demo run (CPU, 98 training / 24 validation inits, ~1,400 training steps, 3.4 M weights; 300 mb RMSE on
6 validation inits over the 384 x 320 domain - synthetic data, so this only shows the pipeline learns, not real skill):

| lead | candidate | height (m) | speed (kt) | vector wind (m/s) | divergence (1e-5 s-1) |
|---|---|---|---|---|---|
| F00 | model analysis | **18.4** | 12.8 | 7.6 | 3.2 |
| F00 | raw nearest-neighbour obs | 42.5 | 19.6 | 12.1 | 20.9 |
| F00 | smoothed obs (first guess) | 20.7 | 12.9 | 7.6 | 3.2 |
| F01 | model (from its analysis) | **20.6** | **13.8** | **8.2** | **3.0** |
| F01 | smoothed obs persisted | 23.0 | 14.5 | 8.5 | 3.2 |
| F02 | model (from its analysis) | **23.4** | **15.7** | **9.8** | 3.9 |
| F02 | smoothed obs persisted | 27.4 | 16.9 | 10.4 | **3.3** |
| F02 | model from the *true* F00 | **11.9** | **8.3** | **5.6** | 1.4 |
| F02 | physics core only from the true F00 | **11.9** | 10.8 | 7.5 | 2.3 |
| F02 | true F00 persisted | 17.4 | 11.1 | 7.6 | **0.9** |

The learned coefficients moved exactly where this synthetic world disagrees with the textbook (it has no adiabatic
warming, no vertical advection and is only geostrophically - not nonlinearly - balanced): at 300 mb
`thermodynamic.adiabatic` 1.00 -> 0.20, `advection.vertical` -> 0.25, `divergence.jacobian` -> 0.37, while
`advection.horizontal` (0.94) and `beta` (1.01) stayed near 1. Weak spots: the analysis barely improves on its smoothed
first guess for winds, divergence forecasts are still worse than persistence, and the F02 map shows a spurious
divergence couplet along the northern boundary (no lateral boundary conditions).

### Adaptive equations ("the AI may change the equations")
Every prognostic equation is written as a sum of named terms, and **each term has a learnable coefficient per pressure
level, initialised to its textbook value 1.0** (option 1 of the design: a variable the model may change through its weights).
The divergence equation (AMS Glossary), as used:

| term | expression | coefficient |
|---|---|---|
| `divergence_squared` | -(D)^2 | c1 |
| `jacobian` | +2 J(u,v) | c2 |
| `tilting` | -grad(omega) . dV/dp | c3 |
| `beta` | -beta u  (general form v df/dx - u df/dy, since f varies in x on the map) | c4 |
| `coriolis_vorticity` | +f zeta | c5 |
| `geopotential_laplacian` | -laplacian(Phi) | c6 |
| advection (material derivative) | V.grad, omega d/dp (semi-Lagrangian) | `advection` equation |

The vorticity equation (`stretching`, `tilting`, `beta`), the thermodynamic equation (`adiabatic` kappa T omega / p) and a
linear-balance height tendency are adaptive the same way. Gradient descent moves a coefficient only if that makes the
forecasts verify better, and a small penalty (`--coeff-prior`) pulls it back toward 1 unless the data disagree. The
learned values are printed after training and stored in the checkpoint's `.json` (`"coefficients"`), so you can see
exactly how each equation was changed. Diagnostic definitions (theta, theta-e, RH, kinematic omega) stay fixed.

How the physics step works: divergence, vorticity and omega (continuity, integrated down from omega = 0 at 100 mb) are
diagnosed; vorticity, divergence, temperature and moisture are advected semi-Lagrangian (stable with 30-min steps on a
3 km grid); the equation terms are added; the vorticity/divergence increments are turned back into winds by
streamfunction / velocity-potential Poisson solves (as in spectral models); heights follow from linear balance. A
12-point lateral relaxation zone tapers the physics increments to zero at the domain edge. The ConvGRU then corrects
what the equations get wrong.

### Known limitations - read before trusting a map
* **Not run against the live servers.** The HRRR/NAM GRIB downloader, the AWIPS fetcher and the Wyoming fetcher were
  developed in a sandbox without access to NOAA/NCAR/UWyo/EDEX servers. GRIB decoding is tested on real GRIB2 files
  (`tests/fixtures`) through a fake HTTP server honouring Range requests, but the first real build is also the first
  live test. The AWS path of the NAM nest (`noaa-nam-pds`) and whether an EDEX server sends grid- or earth-relative
  winds (`AwipsEdexGridFetcher(winds_grid_relative=...)`) should be checked on first use.
* **3 km grid != 3 km information at F00.** Radiosondes are ~400 km apart; nearest-neighbour gridding gives blocky
  (Voronoi) cells. The AnalysisNet therefore starts from a Barnes-like first guess - the nearest-neighbour grid
  Gaussian-smoothed at 100 km, which alone halves the F00 error of the raw grid - and learns a correction to it. Convective-scale detail can only come from
  what the network learns from the 3 km truth. Aircraft (ACARS/AMDAR), mesonet and mesoanalysis data are the natural
  next inputs.
* **No lateral boundary conditions yet.** Fine for 1-2 h; a 48 h regional run needs boundary data from a parent model (e.g. GFS).
* **Only F00-F02 are trained/validated.** The loop runs to 48 h but longer leads are untrained.
* **Physics prior vs synthetic data:** the synthetic atmosphere is kinematic (patterns translate; it does not obey the
  primitive equations), so there the textbook physics step is *worse* than persistence for wind/temperature and the
  coefficients learn to damp the terms that world does not obey. On real HRRR data the equations should fit far better
  (an f-plane test with an exact solution shows the physics step removing ~70-90 % of the persistence error), but that is
  untested. Compare with `--no-physics` on your first real run.
* **GRIB libraries:** use `pygrib` (default). The `eccodes` pip wheel bundles its own PROJ, which clashes with
  cartopy/pyproj in the same process (crash at exit), and pygrib + eccodes in one process also crash.
* **Basemap:** cartopy downloads Natural Earth outlines on first use; offline, maps show a lat/lon grid only.
