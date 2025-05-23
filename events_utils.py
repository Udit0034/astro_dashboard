from datetime import datetime

def get_astronomy_events(limit=6):
    """
    Returns a list of upcoming astronomical events.
    Each event is a dictionary with 'name', 'date', and 'description' keys.
    """
    events = [
        {
            "name": "Eta Aquariids Meteor Shower",
            "date": "2025-05-07",
            "description": "Peaks with around 24 meteors per hour; best viewed between 3am and 5am."
        },
        {
            "name": "Total Lunar Eclipse",
            "date": "2025-09-07",
            "description": "A total lunar eclipse visible over east Africa, Asia, and Australia."
        },
        {
            "name": "Partial Solar Eclipse",
            "date": "2025-09-21",
            "description": "A partial solar eclipse visible across much of Oceania and Antarctica."
        },
        {
            "name": "Perseid Meteor Shower",
            "date": "2025-08-12",
            "description": "One of the brightest meteor showers, producing up to 100 meteors per hour."
        },
        {
            "name": "Geminid Meteor Shower",
            "date": "2025-12-14",
            "description": "Considered the king of meteor showers, with up to 160 meteors per hour."
        },
        {
            "name": "Planetary Parade",
            "date": "2025-02-28",
            "description": "A rare alignment where all seven other planets appear in the sky."
        }
    ]

    # Convert date strings to datetime objects for sorting
    for event in events:
        event["date_obj"] = datetime.strptime(event["date"], "%Y-%m-%d")

    # Sort events by date
    events.sort(key=lambda x: x["date_obj"])

    # Remove the temporary 'date_obj' key
    for event in events:
        del event["date_obj"]

    return events[:limit]

