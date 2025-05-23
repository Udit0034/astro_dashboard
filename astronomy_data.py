# astronomy_data.py
import json
import pandas as pd
import numpy as np
from moon_utils import detect_location, get_lat_lon, get_timezone, get_moon_phases
from events_utils import get_astronomy_events
from eclipse_utils import get_next_eclipses_from_csv
from classifier_utils import fetch_and_classify
from sentence_transformers import SentenceTransformer


def build_astronomical_data(
    city_name: str = None,
    embeddings_path: str = "astronomy_data_embeddings.npz",
    json_path: str = "astronomical_data.json",
    chunk_size: int = 500
):
    """
    Gathers moon phases, eclipse info, astronomy events, and 7-day forecast,
    writes them to a JSON file, and builds embeddings/chunks.

    Returns:
      data (dict), embeddings (np.ndarray), chunks (List[str])
    """
    # Location
    if city_name:
        lat, lon, _ = get_lat_lon(city_name)
        tz = get_timezone(lat, lon)
    else:
        lat, lon, tz, _ = detect_location()

    # Moon phases
    moon_df = get_moon_phases(lat, lon, tz)
    moon_phases = [
        {"phase": row.phase, "datetime": row.datetime.isoformat()}
        for _, row in moon_df.iterrows()
    ]

    # Events
    events = get_astronomy_events()
    if isinstance(events, str):
        events = json.loads(events)

    # Eclipses
    eclipses = get_next_eclipses_from_csv()

    # Forecast
    daily_df, _ = fetch_and_classify(lat, lon)
    forecast_daily = [
        {"date": pd.to_datetime(r.Date).isoformat(),
         "label": r.label}
        for _, r in daily_df.iterrows()
    ]

    # Compile
    data = {
        "moon_phases": moon_phases,
        "events": events,
        "eclipses": eclipses,
        "forecast_daily": forecast_daily
    }
    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Build embeddings
    text = json.dumps(data)
    chunks = [text[i : i+chunk_size] for i in range(0, len(text), chunk_size)]
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks)
    np.savez_compressed(embeddings_path, embeddings=embeddings, chunks=chunks)

    return data, embeddings, chunks


if __name__ == "__main__":
    build_astronomical_data()
