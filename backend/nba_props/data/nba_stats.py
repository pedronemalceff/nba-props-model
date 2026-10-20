"""Historical box-score data from stats.nba.com via nba_api, cached as parquet.

One LeagueGameLog call per season returns every player (or team) game, so a
full refresh is only a handful of requests.

Note: stats.nba.com blocks many cloud/datacenter IPs. Run the fetch from a
home connection; the cached parquet files can then be committed or copied.
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd

from ..config import CACHE_DIR, default_seasons, season_for

log = logging.getLogger(__name__)

PLAYER_COLS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID",
    "GAME_DATE", "MATCHUP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]
TEAM_COLS = [
    "SEASON_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "MIN",
    "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST",
    "STL", "BLK", "TOV", "PF", "PTS",
]


def _call(fn, retries: int = 4, **kw):
    for attempt in range(retries):
        try:
            return fn(timeout=60, **kw).get_data_frames()[0]
        except Exception as e:  # noqa: BLE001 - nba_api raises plain exceptions
            wait = 3 * (attempt + 1)
            log.warning("nba_api call failed (%s), retrying in %ss", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"nba_api call {fn.__name__} failed after {retries} tries")


def _fetch_gamelog(season: str, kind: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    frames = []
    for stype in ("Regular Season", "Playoffs"):
        df = _call(
            leaguegamelog.LeagueGameLog,
            season=season,
            player_or_team_abbreviation=kind,
            season_type_all_star=stype,
        )
        df["SEASON_TYPE"] = stype
        frames.append(df)
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True)
    df["SEASON"] = season
    return df


def _fetch_positions(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playerindex

    df = _call(playerindex.PlayerIndex, season=season)
    id_col = "PERSON_ID" if "PERSON_ID" in df.columns else "PLAYER_ID"
    out = df[[id_col, "POSITION"]].rename(columns={id_col: "PLAYER_ID"})
    return out


def load_rosters(season: str, refresh: bool = False) -> dict[int, str]:
    """PLAYER_ID -> current team abbreviation (handles trades/signings)."""
    path = CACHE_DIR / f"rosters_{season}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
    else:
        from nba_api.stats.endpoints import commonallplayers
        try:
            df = _call(commonallplayers.CommonAllPlayers, is_only_current_season=1, season=season)
            df = df[["PERSON_ID", "TEAM_ABBREVIATION"]]
            df.to_parquet(path, index=False)
        except RuntimeError as e:
            log.warning("Rosters unavailable: %s", e)
            return {}
    df = df[df["TEAM_ABBREVIATION"].fillna("").str.len() > 0]
    return dict(zip(df["PERSON_ID"].astype(int), df["TEAM_ABBREVIATION"]))


def load_season(season: str, kind: str, refresh: bool = False) -> pd.DataFrame:
    """kind = 'P' (players) or 'T' (teams)."""
    path = CACHE_DIR / f"gamelog_{kind}_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    log.info("Fetching %s game logs for %s", "player" if kind == "P" else "team", season)
    df = _fetch_gamelog(season, kind)
    df.to_parquet(path, index=False)
    return df


def load_positions(season: str, refresh: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"positions_{season}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    try:
        df = _fetch_positions(season)
        df.to_parquet(path, index=False)
        return df
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch positions for %s: %s", season, e)
        return pd.DataFrame(columns=["PLAYER_ID", "POSITION"])


def load_all(today: date, history: int = 3, refresh_current: bool = True):
    """Return (player_logs, team_logs, positions) across history + current season.

    Past seasons are fetched once and cached; the current season is re-fetched
    when refresh_current is True so the latest games are included.
    """
    current = season_for(today)
    players, teams = [], []
    for s in default_seasons(today, history):
        refresh = refresh_current and s == current
        try:
            players.append(load_season(s, "P", refresh))
            teams.append(load_season(s, "T", refresh))
        except RuntimeError as e:
            log.warning("Skipping season %s: %s", s, e)
    if not players:
        raise RuntimeError("No game log data available. Check network access to stats.nba.com")
    pos = load_positions(current, refresh=refresh_current)
    if pos.empty:
        prev = default_seasons(today, 1)[0]
        pos = load_positions(prev)
    p = pd.concat(players, ignore_index=True)
    t = pd.concat(teams, ignore_index=True)
    keep_p = [c for c in PLAYER_COLS + ["SEASON", "SEASON_TYPE"] if c in p.columns]
    keep_t = [c for c in TEAM_COLS + ["SEASON", "SEASON_TYPE"] if c in t.columns]
    return p[keep_p], t[keep_t], pos
