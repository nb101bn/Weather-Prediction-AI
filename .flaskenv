# Read automatically by `flask run` (requires python-dotenv). Lets `flask run` work from the project root.
FLASK_APP=src/app.py
# FastAPI service the HUD forwards to; if it's not running the HUD calls the same code in-process.
POINT_FORECAST_API_URL=http://127.0.0.1:8000
