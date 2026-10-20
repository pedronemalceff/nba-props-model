"""Static team reference data: names, arena coordinates, altitude, time zone.

Used for travel distance, altitude, time-zone shift features and for mapping
team names coming from the Odds API / ESPN to NBA abbreviations.
"""
from __future__ import annotations

import math

# abbr: (full name, arena lat, arena lon, elevation ft, standard UTC offset)
TEAMS: dict[str, tuple[str, float, float, int, int]] = {
    "ATL": ("Atlanta Hawks", 33.757, -84.396, 1050, -5),
    "BOS": ("Boston Celtics", 42.366, -71.062, 20, -5),
    "BKN": ("Brooklyn Nets", 40.683, -73.975, 30, -5),
    "CHA": ("Charlotte Hornets", 35.225, -80.839, 750, -5),
    "CHI": ("Chicago Bulls", 41.881, -87.674, 600, -6),
    "CLE": ("Cleveland Cavaliers", 41.497, -81.688, 650, -5),
    "DAL": ("Dallas Mavericks", 32.790, -96.810, 430, -6),
    "DEN": ("Denver Nuggets", 39.749, -105.008, 5280, -7),
    "DET": ("Detroit Pistons", 42.341, -83.055, 600, -5),
    "GSW": ("Golden State Warriors", 37.768, -122.388, 10, -8),
    "HOU": ("Houston Rockets", 29.751, -95.362, 50, -6),
    "IND": ("Indiana Pacers", 39.764, -86.155, 715, -5),
    "LAC": ("Los Angeles Clippers", 33.945, -118.341, 100, -8),
    "LAL": ("Los Angeles Lakers", 34.043, -118.267, 300, -8),
    "MEM": ("Memphis Grizzlies", 35.138, -90.051, 260, -6),
    "MIA": ("Miami Heat", 25.781, -80.188, 10, -5),
    "MIL": ("Milwaukee Bucks", 43.045, -87.917, 600, -6),
    "MIN": ("Minnesota Timberwolves", 44.979, -93.276, 830, -6),
    "NOP": ("New Orleans Pelicans", 29.949, -90.082, 5, -6),
    "NYK": ("New York Knicks", 40.751, -73.993, 30, -5),
    "OKC": ("Oklahoma City Thunder", 35.463, -97.515, 1200, -6),
    "ORL": ("Orlando Magic", 28.539, -81.384, 100, -5),
    "PHI": ("Philadelphia 76ers", 39.901, -75.172, 40, -5),
    "PHX": ("Phoenix Suns", 33.446, -112.071, 1100, -7),
    "POR": ("Portland Trail Blazers", 45.532, -122.667, 50, -8),
    "SAC": ("Sacramento Kings", 38.580, -121.500, 30, -8),
    "SAS": ("San Antonio Spurs", 29.427, -98.437, 650, -6),
    "TOR": ("Toronto Raptors", 43.643, -79.379, 250, -5),
    "UTA": ("Utah Jazz", 40.768, -111.901, 4300, -7),
    "WAS": ("Washington Wizards", 38.898, -77.021, 50, -5),
}

# Alternate spellings seen in third-party feeds.
_ALIASES = {
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Philadelphia Sixers": "PHI",
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA",
    "WSH": "WAS", "PHO": "PHX", "BRK": "BKN", "CHO": "CHA", "NJN": "BKN",
}

NAME_TO_ABBR = {v[0].lower(): k for k, v in TEAMS.items()}
NAME_TO_ABBR.update({k.lower(): v for k, v in _ALIASES.items()})
NAME_TO_ABBR.update({k.lower(): k for k in TEAMS})


def to_abbr(name: str) -> str | None:
    """Map a team name/abbreviation from any feed to the NBA abbreviation."""
    if not name:
        return None
    return NAME_TO_ABBR.get(name.strip().lower())


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def arena(abbr: str) -> tuple[float, float, int, int]:
    _, lat, lon, elev, tz = TEAMS[abbr]
    return lat, lon, elev, tz
