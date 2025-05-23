# classifier_utils.py

import pandas as pd
import numpy as np
import joblib
import requests_cache
from retry_requests import retry
import openmeteo_requests
from timezonefinder import TimezoneFinder
from skyfield.api import load, wgs84
from skyfield import almanac
from skyfield.almanac import dark_twilight_day
from skyfield.framelib import ecliptic_frame
import datetime as dt
import pytz

# Load model
model = joblib.load("xgb_model.pkl")

# Timezone finder
_tf = TimezoneFinder()

# Weather setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_and_classify(lat, lon):
    """
    Wrapper function to match expected import in app.py
    Returns hourly and daily DataFrames with predictions
    """
    return full_pipeline(lat, lon)


def get_weather_data(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m", "dew_point_2m", "wind_speed_10m",
            "relative_humidity_2m", "visibility", "cloud_cover", "weather_code",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"
        ],
        "timezone": "auto",
        "forecast_days": 7
    }
    resp = openmeteo.weather_api(url, params=params)[0]
    hourly = resp.Hourly()
    tz_str = _tf.timezone_at(lat=lat, lng=lon) or 'UTC'
    tz = pytz.timezone(tz_str)
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ).tz_convert(tz)
    data = {"date": times}
    names = [
        "temperature_2m", "dew_point_2m", "wind_speed_10m",
        "relative_humidity_2m", "visibility", "cloud_cover", "weather_code",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"
    ]
    for i, name in enumerate(names):
        arr = hourly.Variables(i).ValuesAsNumpy()
        data[name] = arr.astype(int) if name == "weather_code" else arr
    return pd.DataFrame(data)

def get_astro_hourly(lat: float, lon: float) -> pd.DataFrame:
    weather_df = get_weather_data(lat, lon)
    dates = weather_df['date']

    eph = load('de421.bsp')
    sun, moon, earth = eph['sun'], eph['moon'], eph['earth']
    ts = load.timescale()
    tz = pytz.timezone(_tf.timezone_at(lat=lat, lng=lon) or 'UTC')

    f_twilight = dark_twilight_day(eph, wgs84.latlon(lat, lon))

    phase_list, illum_list = [], []
    day_start_list, day_end_list = [], []

    for dt_utc in dates:
        t = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute)
        e = earth.at(t)
        s = e.observe(sun).apparent()
        m = e.observe(moon).apparent()
        _, slon, _ = s.frame_latlon(ecliptic_frame)
        _, mlon, _ = m.frame_latlon(ecliptic_frame)
        phase = (mlon.degrees - slon.degrees) % 360.0
        illum = (1 - np.cos(np.deg2rad(phase))) / 2 * 100
        phase_list.append(phase)
        illum_list.append(illum)

        # Fixing sunrise/sunset by finding twilight bounds
        t0 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day)
        t1 = ts.utc(dt_utc.year, dt_utc.month, dt_utc.day + 1)
        times_list, events = almanac.find_discrete(t0, t1, f_twilight)
        # Find actual sunrise and sunset for the date
        sunrise = next(
            (t.utc_datetime().astimezone(tz).time()
            for i, t in enumerate(times_list)
            if events[i] == 4),  # Day starts
            dt.time(6, 0)
        )

        sunset = next(
            (t.utc_datetime().astimezone(tz).time()
            for i, t in reversed(list(enumerate(times_list)))
            if events[i] == 0),  # Day ends
            dt.time(18, 0)
        )

        day_start_list.append(sunrise)
        day_end_list.append(sunset)

    return pd.DataFrame({
        'date': dates,
        'Phase (deg)': phase_list,
        'Illuminated (%)': illum_list,
        'Day Starts': day_start_list,
        'Day Ends': day_end_list
    })

def compute_smog_proxy(temp, wind, pm25, no2, so2):
    return ((temp < 10) & (wind < 5) & ((pm25 > 60) | (no2 > 50) | (so2 > 40))).astype(int)

def full_pipeline(lat, lon):
    weather = get_weather_data(lat, lon)
    astro = get_astro_hourly(lat, lon)

    df = weather.merge(astro, on="date", how="left")
    df["Date"] = df["date"].dt.date
    df["Time"] = df["date"].dt.strftime('%H:%M:%S')  # Fix #2
    df.drop(columns=["date"], inplace=True)

    # Dummy pollution data
    df["pm25"] = np.random.randint(30, 100, size=len(df))
    df["no2"] = np.random.randint(10, 60, size=len(df))
    df["so2"] = np.random.randint(5, 50, size=len(df))

    df["smog_proxy"] = compute_smog_proxy(
        df["temperature_2m"], df["wind_speed_10m"], df["pm25"], df["no2"], df["so2"]
    )

    # Predictions
    X_hourly = df[[
        "temperature_2m", "dew_point_2m", "wind_speed_10m",
        "relative_humidity_2m", "visibility", "cloud_cover",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "Illuminated (%)", "smog_proxy"
    ]]
    df["label"] = model.predict(X_hourly)

    # Fix #3: Update label to 'sun' if Time is between Day Starts and Day Ends
    df["Time_obj"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
    df["label"] = df.apply(
        lambda row: "sun" if row["Day Starts"] <= row["Time_obj"] <= row["Day Ends"] else row["label"],
        axis=1
    )
    df.drop(columns=["Time_obj"], inplace=True)

    # Daily Aggregation
    df2 = df.copy()
    df2["Date"] = pd.to_datetime(df2["Date"], errors="coerce")
    agg_dict = {
        "temperature_2m": "mean",
        "dew_point_2m": "mean",
        "wind_speed_10m": "mean",
        "relative_humidity_2m": "mean",
        "visibility": "mean",
        "cloud_cover": "mean",
        "cloud_cover_low": "mean",
        "cloud_cover_mid": "mean",
        "cloud_cover_high": "mean",
        "weather_code": lambda x: int(round(x.mode()[0])) if not x.mode().empty else int(round(x.mean())),
        "Phase (deg)": "mean",
        "Illuminated (%)": "mean",
        "Day Starts": "first",
        "Day Ends": "first",
        "smog_proxy": "mean",
        "label": lambda x: int(round(x.mode()[0])) if not x.mode().empty and isinstance(x.mode()[0], (int, float)) else 0
    }
    df2 = df2.groupby("Date").agg(agg_dict).reset_index()
    df2["smog_proxy"] = df2["smog_proxy"].round().astype(int)

    # Daily label prediction
    X_daily = df2[[
        "temperature_2m", "dew_point_2m", "wind_speed_10m",
        "relative_humidity_2m", "visibility", "cloud_cover",
        "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
        "Illuminated (%)", "smog_proxy"
    ]]
    df2["label"] = model.predict(X_daily)

    return df2, df

