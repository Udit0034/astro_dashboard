# moon_utils.py

import datetime as dt
import pandas as pd
import pytz
from skyfield.framelib import ecliptic_frame 
from skyfield.api import load
from skyfield import almanac
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
import requests

_tf = TimezoneFinder()
_geolocator = Nominatim(user_agent="astro-weather-app", timeout=10)

def detect_location():
    """Auto-detect via free IP API, returns lat, lon, tz_str, city."""
    try:
        r = requests.get("http://ip-api.com/json/").json()
        lat, lon, city = r["lat"], r["lon"], r["city"]
        tz_str = r["timezone"]
        return lat, lon, tz_str, city
    except:
        # last resort stub
        return 0.0, 0.0, "UTC", "Unknown"

def get_lat_lon(city_name):
    """Geocode city_name → lat, lon, city_name. Raises on failure."""
    loc = _geolocator.geocode(city_name)
    if not loc:
        raise ValueError(f"City '{city_name}' not found")
    return loc.latitude, loc.longitude, city_name

def get_timezone(lat, lon):
    tz = _tf.timezone_at(lat=lat, lng=lon)
    return tz or "UTC"

def get_moon_phases(lat, lon, tz_str):
    """Return DataFrame with columns ['phase','datetime'] in local tz."""
    ts = load.timescale()
    eph = load("de421.bsp")
    sun, moon, earth = eph["sun"], eph["moon"], eph["earth"]

    # compute for next two phases
    now_utc = dt.datetime.utcnow().replace(tzinfo=pytz.utc)
    local_tz = pytz.timezone(tz_str)
    t0 = ts.utc(now_utc.year, now_utc.month, now_utc.day)
    t1 = ts.utc(now_utc.year, now_utc.month, now_utc.day + 30)
    f = almanac.moon_phases(eph)
    times, phases = almanac.find_discrete(t0, t1, f)

    out = []
    for ti, ph in zip(times, phases):
        name = almanac.MOON_PHASES[ph]
        dt_loc = ti.utc_datetime().astimezone(local_tz)
        out.append({"phase": name, "datetime": dt_loc})
        if name == "Full Moon" and len(out) >= 2:
            break

    return pd.DataFrame(out)
