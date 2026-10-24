"""Player prop lines from The Odds API (https://the-odds-api.com).

Cost: each event-odds call costs (#markets x #regions) credits. With 7 markets,
1 region and ~8 games that is ~56 credits/day, so the free tier (500/mo) covers
roughly a week of daily runs. Responses are cached per day; trim MARKETS or set
ODDS_BOOKMAKERS to save credits.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from ..config import CACHE_DIR, MARKETS, ODDS_TO_MARKET, settings
from ..teams import to_abbr

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"

LINE_COLS = ["EVENT_ID", "COMMENCE", "HOME", "AWAY", "BOOK", "BOOK_KEY", "MARKET",
             "PLAYER_NAME", "LINE", "SIDE", "PRICE"]


def american_to_decimal(price: float) -> float:
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def implied_prob(price: float) -> float:
    return 1 / american_to_decimal(price)


def _day_window_utc(day: date, tz: str) -> tuple[str, str]:
    z = ZoneInfo(tz)
    start = datetime.combine(day, time(6, 0), z)  # 6am local -> 6am next day
    end = start + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (start.astimezone(ZoneInfo("UTC")).strftime(fmt),
            end.astimezone(ZoneInfo("UTC")).strftime(fmt))


def parse_event_odds(ev: dict) -> list[dict]:
    rows = []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            market = ODDS_TO_MARKET.get(m.get("key"))
            if not market:
                continue
            for o in m.get("outcomes", []):
                if o.get("point") is None or o.get("name") not in ("Over", "Under"):
                    continue
                rows.append({
                    "EVENT_ID": ev.get("id"),
                    "COMMENCE": ev.get("commence_time"),
                    "HOME": to_abbr(ev.get("home_team", "")),
                    "AWAY": to_abbr(ev.get("away_team", "")),
                    "BOOK": bk.get("title") or bk.get("key"),
                    "BOOK_KEY": bk.get("key"),
                    "MARKET": market,
                    "PLAYER_NAME": o.get("description"),
                    "LINE": float(o["point"]),
                    "SIDE": o["name"].upper(),
                    "PRICE": float(o["price"]),
                })
    return rows


def fetch_props(day: date, refresh: bool = False) -> pd.DataFrame:
    """All Over/Under prop prices for games on `day` (local time)."""
    path = CACHE_DIR / f"odds_{day.isoformat()}.json"
    if path.exists() and not refresh:
        events = json.loads(path.read_text())
    else:
        if not settings.odds_api_key:
            raise RuntimeError("ODDS_API_KEY not set")
        frm, to = _day_window_utc(day, settings.timezone)
        key = settings.odds_api_key
        with httpx.Client(timeout=30) as c:
            r = c.get(f"{BASE}/events", params={
                "apiKey": key, "commenceTimeFrom": frm, "commenceTimeTo": to})
            r.raise_for_status()
            listing = r.json()
            markets = ",".join(MARKETS[m][0] for m in settings.markets if m in MARKETS)
            events = []
            for ev in listing:
                params = {"apiKey": key, "markets": markets, "oddsFormat": "american"}
                if settings.odds_bookmakers:
                    params["bookmakers"] = settings.odds_bookmakers
                else:
                    params["regions"] = settings.odds_regions
                rr = c.get(f"{BASE}/events/{ev['id']}/odds", params=params)
                if rr.status_code != 200:
                    log.warning("Odds for event %s failed: %s", ev["id"], rr.text[:200])
                    continue
                events.append(rr.json())
            log.info("Odds API credits remaining: %s", r.headers.get("x-requests-remaining"))
        # Don't cache an empty pull (lines often post later in the day).
        if any(ev.get("bookmakers") for ev in events):
            path.write_text(json.dumps(events))
        else:
            log.warning("Odds API returned games but no prop lines for %s", day)
    rows = [row for ev in events for row in parse_event_odds(ev)]
    return pd.DataFrame(rows, columns=LINE_COLS)


def summarize_lines(raw: pd.DataFrame, preferred: list[str] | None = None) -> pd.DataFrame:
    """One row per (player, market, line): best price per side + no-vig fair prob.

    The fair (no-vig) probability uses every book in the pull, which is a sharper
    read of the market. When `preferred` book keys are given (e.g. the only book you
    bet at), prices come from those books only and props they don't offer are
    dropped. If the preferred books have no props at all, every book is used and
    the result has attrs["preferred_missing"] = True.
    """
    if raw.empty:
        return pd.DataFrame()
    if "BOOK_KEY" not in raw:
        raw = raw.assign(BOOK_KEY=raw["BOOK"])
    keys = ["EVENT_ID", "PLAYER_NAME", "MARKET"]
    pref = [p.strip() for p in (preferred or []) if p.strip()]
    pref_raw = raw[raw["BOOK_KEY"].isin(pref)] if pref else raw.iloc[0:0]
    use_pref = bool(pref) and not pref_raw.empty
    # Main line: the preferred book's line if betting there, else the most common line.
    basis = pref_raw if use_pref else raw
    main_line = (basis.groupby(keys + ["LINE"]).size().reset_index(name="n")
                 .sort_values("n", ascending=False).drop_duplicates(keys))
    df = raw.merge(main_line[keys + ["LINE"]], on=keys + ["LINE"])
    wide = df.pivot_table(index=keys + ["LINE", "BOOK", "BOOK_KEY", "HOME", "AWAY", "COMMENCE"],
                          columns="SIDE", values="PRICE", aggfunc="first").reset_index()
    wide = wide.dropna(subset=["OVER", "UNDER"]) if {"OVER", "UNDER"} <= set(wide.columns) else wide.iloc[0:0]
    if wide.empty:
        return pd.DataFrame()
    io, iu = wide["OVER"].map(implied_prob), wide["UNDER"].map(implied_prob)
    wide["FAIR_OVER"] = io / (io + iu)
    g = wide.groupby(keys + ["LINE", "HOME", "AWAY", "COMMENCE"])
    out = g.agg(FAIR_OVER=("FAIR_OVER", "median"), N_BOOKS=("BOOK", "nunique")).reset_index()
    priced = wide[wide["BOOK_KEY"].isin(pref)] if use_pref else wide
    gp = priced.groupby(keys)
    best_o = priced.loc[gp["OVER"].idxmax(), keys + ["OVER", "BOOK"]].rename(
        columns={"OVER": "BEST_OVER", "BOOK": "BEST_OVER_BOOK"})
    best_u = priced.loc[gp["UNDER"].idxmax(), keys + ["UNDER", "BOOK"]].rename(
        columns={"UNDER": "BEST_UNDER", "BOOK": "BEST_UNDER_BOOK"})
    res = out.merge(best_o, on=keys).merge(best_u, on=keys)
    res.attrs["preferred_missing"] = bool(pref) and not use_pref
    return res
