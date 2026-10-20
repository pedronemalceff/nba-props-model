"""Central configuration. Override anything via environment variables or .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("NBA_PROPS_DATA_DIR", BACKEND_DIR / "data"))
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
PICKS_DIR = DATA_DIR / "picks"

# Market key -> (Odds API market key, stat column in our data, distribution)
MARKETS: dict[str, tuple[str, str, str]] = {
    "PTS": ("player_points", "PTS", "normal"),
    "REB": ("player_rebounds", "REB", "negbin"),
    "AST": ("player_assists", "AST", "negbin"),
    "3PM": ("player_threes", "FG3M", "negbin"),
    "PRA": ("player_points_rebounds_assists", "PRA", "normal"),
    "PR": ("player_points_rebounds", "PR", "normal"),
    "PA": ("player_points_assists", "PA", "normal"),
}
ODDS_TO_MARKET = {v[0]: k for k, v in MARKETS.items()}


def season_for(d: date) -> str:
    """NBA season string ('2026-27') that a calendar date belongs to."""
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{(start + 1) % 100:02d}"


def default_seasons(d: date, history: int = 3) -> list[str]:
    cur = int(season_for(d)[:4])
    return [f"{y}-{(y + 1) % 100:02d}" for y in range(cur - history, cur + 1)]


@dataclass
class Settings:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    odds_regions: str = os.getenv("ODDS_REGIONS", "us")
    # Comma list, e.g. "draftkings,fanduel". Empty = all books in the region.
    odds_bookmakers: str = os.getenv("ODDS_BOOKMAKERS", "")
    markets: list[str] = field(
        default_factory=lambda: os.getenv("MARKETS", ",".join(MARKETS)).split(",")
    )
    history_seasons: int = int(os.getenv("HISTORY_SEASONS", "3"))
    # Pick selection
    min_prob: float = float(os.getenv("MIN_PROB", "0.54"))
    min_ev: float = float(os.getenv("MIN_EV", "0.02"))
    model_weight: float = float(os.getenv("MODEL_WEIGHT", "0.65"))  # blend vs market
    max_per_player: int = int(os.getenv("MAX_PER_PLAYER", "1"))
    max_per_game: int = int(os.getenv("MAX_PER_GAME", "3"))
    min_recent_minutes: float = float(os.getenv("MIN_RECENT_MINUTES", "18"))
    include_questionable: bool = os.getenv("INCLUDE_QUESTIONABLE", "1") == "1"
    timezone: str = os.getenv("TZ_NAME", "America/New_York")


settings = Settings()

for _d in (CACHE_DIR, MODEL_DIR, PICKS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
