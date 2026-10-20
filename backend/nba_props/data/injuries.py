"""Current injury report from ESPN's public JSON feed."""
from __future__ import annotations

import json
import logging
from datetime import date

import httpx
import pandas as pd

from ..config import CACHE_DIR
from ..teams import to_abbr
from .names import normalize_name

log = logging.getLogger(__name__)
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Statuses that mean the player is not expected to play.
OUT_STATUSES = {"out", "doubtful", "suspension", "suspended", "inactive"}
QUESTIONABLE_STATUSES = {"questionable", "day-to-day", "game time decision", "probable"}


def parse_injuries(payload: dict) -> pd.DataFrame:
    rows = []
    for team in payload.get("injuries", []):
        team_abbr = to_abbr(team.get("displayName", ""))
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {}) or {}
            name = ath.get("displayName")
            if not name:
                continue
            abbr = team_abbr or to_abbr((ath.get("team") or {}).get("abbreviation", ""))
            status = (inj.get("status") or "").strip()
            rows.append({
                "TEAM": abbr,
                "PLAYER_NAME": name,
                "NAME_KEY": normalize_name(name),
                "STATUS": status,
                "OUT": status.lower() in OUT_STATUSES,
                "QUESTIONABLE": status.lower() in QUESTIONABLE_STATUSES,
                "DETAIL": inj.get("shortComment") or "",
            })
    return pd.DataFrame(rows, columns=[
        "TEAM", "PLAYER_NAME", "NAME_KEY", "STATUS", "OUT", "QUESTIONABLE", "DETAIL"])


def fetch_injuries(day: date, refresh: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"injuries_{day.isoformat()}.json"
    if path.exists() and not refresh:
        return parse_injuries(json.loads(path.read_text()))
    try:
        r = httpx.get(URL, timeout=30)
        r.raise_for_status()
        payload = r.json()
        path.write_text(json.dumps(payload))
        return parse_injuries(payload)
    except Exception as e:  # noqa: BLE001
        log.warning("Injury feed unavailable (%s); continuing without it", e)
        return parse_injuries({})
