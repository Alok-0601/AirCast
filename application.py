"""
Urban Air Quality AQI Predictor - Flask App
Author: Alok Gupta | GGSIPU Delhi
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template

application = Flask(__name__)
app = application  # for AWS Elastic Beanstalk compatibility

# ── Load model and scaler ─────────────────────────────────────
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('lasso.pkl', 'rb') as f:
    model = pickle.load(f)

# ── All 61 features the model expects (in exact order) ────────
FEATURE_ORDER = list(scaler.feature_names_in_)

# ── Input validation ranges ───────────────────────────────────
FIELD_RANGES = {
    'temperature_C':        (5,   45,   'Temperature'),
    'humidity_pct':         (18,  98,   'Humidity'),
    'wind_speed_kmh':       (0.5, 40,   'Wind Speed'),
    'rainfall_mm':          (0,   85,   'Rainfall'),
    'traffic_density':      (5,   130,  'Traffic Density'),
    'industrial_proximity': (1,   10,   'Industrial Proximity'),
    'population_density':   (18,  165,  'Population Density'),
    'green_cover_pct':      (1,   38,   'Green Cover'),
    'PM2_5':                (5,   420,  'PM2.5'),
    'PM10':                 (10,  500,  'PM10'),
    'NO2':                  (5,   200,  'NO₂'),
    'SO2':                  (1,   80,   'SO₂'),
    'CO':                   (0.1, 15,   'CO'),
    'O3':                   (5,   120,  'O₃'),
    'Benzene':              (0.1, 18,   'Benzene'),
    'Toluene':              (0.2, 35,   'Toluene'),
}

# ── AQI category helper ───────────────────────────────────────
def get_aqi_category(aqi):
    if aqi <= 50:
        return ("Good", "#22c55e",
                "Air quality is satisfactory. Outdoor activities are safe.")
    elif aqi <= 100:
        return ("Moderate", "#facc15",
                "Acceptable air quality. Sensitive individuals should limit prolonged outdoor exertion.")
    elif aqi <= 150:
        return ("Unhealthy for Sensitive Groups", "#f97316",
                "Sensitive groups (elderly, children, asthma patients) should reduce outdoor activity.")
    elif aqi <= 200:
        return ("Unhealthy", "#ef4444",
                "Everyone may begin to experience health effects. Limit prolonged outdoor exertion.")
    elif aqi <= 300:
        return ("Very Unhealthy", "#a855f7",
                "Health alert! Everyone should avoid prolonged outdoor activity.")
    else:
        return ("Hazardous", "#e11d48",
                "Emergency conditions. Entire population is affected. Stay indoors immediately.")


def validate_inputs(form_data):
    """
    Validate user-supplied numeric fields against expected ranges.
    Returns a list of human-readable error strings (empty = all OK).
    """
    errors = []
    for field, (lo, hi, label) in FIELD_RANGES.items():
        raw = form_data.get(field, '').strip()
        if raw == '':
            errors.append(f"{label} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi} (got {val}).")
    return errors


def build_input_vector(form_data):
    """
    Build the full 61-feature input vector from user form data.
    User provides 16 pollutant/weather features + 2 dropdowns.
    Remaining features get sensible derived or neutral defaults.
    """
    now = datetime.now()

    # ── Dropdowns ────────────────────────────────────────────
    season  = form_data.get('season', 'Winter')
    station = form_data.get('station_type', 'Traffic')

    # One-hot encode season (drop_first=True dropped 'Autumn')
    season_monsoon = 1 if season == 'Monsoon' else 0
    season_spring  = 1 if season == 'Spring'  else 0
    season_winter  = 1 if season == 'Winter'  else 0

    # One-hot encode station_type (drop_first=True dropped 'Background')
    station_industrial  = 1 if station == 'Industrial'  else 0
    station_residential = 1 if station == 'Residential' else 0
    station_traffic     = 1 if station == 'Traffic'     else 0

    # ── Core user inputs ──────────────────────────────────────
    temperature_C        = float(form_data.get('temperature_C',        26))
    humidity_pct         = float(form_data.get('humidity_pct',         60))
    wind_speed_kmh       = float(form_data.get('wind_speed_kmh',       10))
    rainfall_mm          = float(form_data.get('rainfall_mm',           0))
    traffic_density      = float(form_data.get('traffic_density',      65))
    industrial_proximity = float(form_data.get('industrial_proximity',  3))
    population_density   = float(form_data.get('population_density',   88))
    green_cover_pct      = float(form_data.get('green_cover_pct',      15))
    PM2_5    = float(form_data.get('PM2_5',    60))
    PM10     = float(form_data.get('PM10',     90))
    NO2      = float(form_data.get('NO2',      40))
    SO2      = float(form_data.get('SO2',      15))
    CO       = float(form_data.get('CO',      1.5))
    O3       = float(form_data.get('O3',       70))
    Benzene  = float(form_data.get('Benzene', 2.5))
    Toluene  = float(form_data.get('Toluene', 5.0))

    # ── Derived features ──────────────────────────────────────
    vehicles_per_km2 = traffic_density * 12           # proportional to traffic
    dew_point_C      = temperature_C * 0.65           # standard approximation
    atm_pressure_hpa = 1013 - temperature_C * 0.3     # lapse-rate estimate
    visibility_km    = max(0.5, 15 - PM2_5 * 0.025)  # inversely related to PM2.5

    # ── Noise / background features → neutral defaults ────────
    noise_defaults = {
        'wind_direction_deg':        180,
        'solar_radiation_wm2':       200,
        'cloud_cover_pct':            40,
        'road_dust_index':             0,
        'construction_activity_index': 0,
        'uv_index':                    5,
        'pollen_count':                0,
        'mixing_layer_height_m':       0,
        'boundary_layer_temp_C':       temperature_C,
        'soil_moisture_index':         0,
        'aerosol_optical_depth':       0,
        'black_carbon_ugm3':           0,
        'organic_carbon_ugm3':         0,
        'secondary_organic_aerosol':   0,
        'nitrate_ugm3':                0,
        'sulfate_ugm3':                0,
        'ammonium_ugm3':               0,
        'methane_ppm':                 0,
        'formaldehyde_ugm3':           0,
        'xylene_ugm3':                 0,
        'heavy_metal_index':           0,
        'lead_ngm3':                   0,
        'arsenic_ngm3':                0,
        'noise_level_db':             55,
        'light_pollution_index':       0,
        'waste_burning_index':         0,
        'crop_burning_index':          0,
        'diesel_generator_index':      0,
        'two_wheeler_density':         0,
        'power_plant_proximity':       0,
    }

    # ── Assemble full feature dict ────────────────────────────
    feature_dict = {
        'year':                     now.year,
        'month':                    now.month,
        'day':                      now.day,
        'hour':                     now.hour,
        'is_weekend':               1 if now.weekday() >= 5 else 0,
        'temperature_C':            temperature_C,
        'humidity_pct':             humidity_pct,
        'wind_speed_kmh':           wind_speed_kmh,
        'rainfall_mm':              rainfall_mm,
        'traffic_density':          traffic_density,
        'vehicles_per_km2':         vehicles_per_km2,
        'industrial_proximity':     industrial_proximity,
        'population_density':       population_density,
        'green_cover_pct':          green_cover_pct,
        'PM2_5':                    PM2_5,
        'PM10':                     PM10,
        'NO2':                      NO2,
        'SO2':                      SO2,
        'CO':                       CO,
        'O3':                       O3,
        'Benzene':                  Benzene,
        'Toluene':                  Toluene,
        'dew_point_C':              dew_point_C,
        'atm_pressure_hpa':         atm_pressure_hpa,
        'visibility_km':            visibility_km,
        **noise_defaults,
        'season_Monsoon':           season_monsoon,
        'season_Spring':            season_spring,
        'season_Winter':            season_winter,
        'station_type_Industrial':  station_industrial,
        'station_type_Residential': station_residential,
        'station_type_Traffic':     station_traffic,
    }

    # Build DataFrame in exact feature order the scaler expects
    df_input = pd.DataFrame([feature_dict])[FEATURE_ORDER]
    return df_input


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Validate first
    errors = validate_inputs(request.form)
    if errors:
        return render_template(
            'index.html',
            error=" | ".join(errors),
            form_data=request.form
        )

    try:
        input_df   = build_input_vector(request.form)
        scaled     = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        prediction = round(float(prediction), 1)
        prediction = max(0, min(500, prediction))   # AQI valid range: 0–500

        category, color, advice = get_aqi_category(prediction)

        return render_template(
            'index.html',
            prediction=prediction,
            category=category,
            color=color,
            advice=advice,
            form_data=request.form
        )

    except Exception as e:
        return render_template('index.html', error=str(e), form_data=request.form)


if __name__ == '__main__':
    app.run()
