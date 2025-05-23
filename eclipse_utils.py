import pandas as pd
from datetime import datetime, date

def parse_date(date_str):
    """
    Convert 'Calendar Date' like '2026 March 3' to datetime.date.
    """
    return datetime.strptime(date_str.strip(), "%Y %B %d").date()

def get_next_eclipses_from_csv(lunar_csv_path ="lunar.csv", solar_csv_path = "solar.csv" ):
    df_lunar = pd.read_csv(lunar_csv_path)
    df_solar = pd.read_csv(solar_csv_path)

    today = date.today()

    # Parse 'Calendar Date' and filter
    df_lunar["parsed_date"] = df_lunar["Calendar Date"].apply(parse_date)
    df_solar["parsed_date"] = df_solar["Calendar Date"].apply(parse_date)

    next_lunar_row = df_lunar[df_lunar["parsed_date"] >= today].sort_values("parsed_date").iloc[0]
    next_solar_row = df_solar[df_solar["parsed_date"] >= today].sort_values("parsed_date").iloc[0]

    return {
        "next_solar": next_solar_row["Calendar Date"],
        "next_lunar": next_lunar_row["Calendar Date"]
    }

