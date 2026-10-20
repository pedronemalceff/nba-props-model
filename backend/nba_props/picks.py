"""Turn projections + sportsbook lines into a ranked list of the day's best props."""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .config import MARKETS, PICKS_DIR, settings
from .data.names import match_name, normalize_name
from .data.odds import american_to_decimal, summarize_lines
from .features import build_features

log = logging.getLogger(__name__)

# ------------------------------------------------------------ factor labelling
_GROUP = {
    "form": "Recent form", "rate": "Per-minute production", "adv": "Advanced",
    "vs": "Matchup history", "sched": "Rest/schedule", "geo": "Travel/venue",
    "tm": "Team offense", "opp": "Opponent defense", "pdef": "Positional defense",
    "inj": "Injuries", "pos": "Position",
}
_NAMES = {
    "sched_rest_days": "Days of rest", "sched_b2b": "Back-to-back",
    "sched_games_last7": "Games in last 7 days", "sched_opp_rest_days": "Opponent rest days",
    "sched_rest_adv": "Rest advantage", "sched_player_days_off": "Days since last game",
    "geo_travel_miles": "Travel miles", "geo_tz_shift": "Time-zone shift",
    "geo_altitude_ft": "Venue altitude", "geo_high_altitude": "High-altitude venue",
    "geo_home": "Home game", "geo_road_trip_len": "Road-trip length",
    "geo_opp_travel_miles": "Opponent travel miles",
    "inj_team_missing_min": "Teammate minutes out", "inj_team_missing_usg": "Teammate usage out",
    "inj_team_starters_out": "Teammate starters out", "inj_team_top_usg_out": "Top usage teammate out",
    "inj_opp_missing_min": "Opponent minutes out", "inj_opp_starters_out": "Opponent starters out",
    "inj_usg_opportunity": "Vacated usage opportunity", "vs_games": "Games vs opponent",
    "form_min_trend": "Minutes trend (L5 vs L20)",
}
_TOK = {"O": "allowed", "FG3M": "3PM", "FG3A": "3PA", "ORTG": "off rating",
        "DRTG": "def rating", "PACE": "pace", "USG": "usage", "TS": "TS%", "MIN": "minutes",
        "OREB": "OREB", "DREB": "DREB", "PCT": "%", "PER36": "per 36", "FACTOR": "vs league avg",
        "SEASON": "season avg", "EWM": "weighted avg", "STD10": "volatility (L10)",
        "AVG": "avg", "DIFF": "vs his norm",
        "HOME": "home/away", "AWAY": "", "SPLIT": "split"}
_GROUP_TEXT = {"opp": "Opp", "tm": "Team", "pdef": "Opp vs position", "vs": "Vs opp",
               "form": "", "rate": "", "adv": "", "geo": "", "sched": "", "inj": ""}


_BASELINE_PREFIX = ("form_", "rate_", "adv_")


def label(feat: str) -> str:
    """Human-readable feature name, e.g. opp_O_PTS_L10 -> 'Opp allowed PTS (L10)'."""
    if feat in _NAMES:
        return _NAMES[feat]
    group, _, rest = feat.partition("_")
    toks, window = [], ""
    for t in rest.split("_"):
        if re.fullmatch(r"L\d+", t):
            window = f" ({t})"
        else:
            tok = _TOK.get(t, t)
            if tok:
                toks.append(tok)
    head = _GROUP_TEXT.get(group, group)
    text = " ".join(toks).replace(" %", "%")
    return (f"{head} {text}".strip() + window)


def _fmt_american(p: float) -> str:
    return f"+{int(p)}" if p > 0 else str(int(p))


# ------------------------------------------------------------------ main logic
def build_upcoming(lines: pd.DataFrame, plogs: pd.DataFrame, day: date,
                   current_team: dict[int, str]) -> pd.DataFrame:
    latest = plogs.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
    name_map = {normalize_name(n): int(i) for n, i in zip(latest["PLAYER_NAME"], latest["PLAYER_ID"])}
    rows, unmatched = [], set()
    for (ev, name, home, away), _ in lines.groupby(["EVENT_ID", "PLAYER_NAME", "HOME", "AWAY"]):
        pid = match_name(name, name_map)
        team = current_team.get(pid) if pid else None
        if pid is None or team not in (home, away):
            unmatched.add(name)
            continue
        rows.append({"PLAYER_ID": pid, "PLAYER_NAME": name, "TEAM": team,
                     "OPP": away if team == home else home, "HOME": int(team == home),
                     "GAME_DATE": pd.Timestamp(day), "GAME_ID": f"UP_{ev}", "EVENT_ID": ev})
    if unmatched:
        log.info("Could not match %d prop players (rookies/new signings?): %s",
                 len(unmatched), sorted(unmatched)[:10])
    return pd.DataFrame(rows).drop_duplicates("PLAYER_ID")


def injury_sets(injuries: pd.DataFrame, plogs: pd.DataFrame):
    latest = plogs.sort_values("GAME_DATE").drop_duplicates("PLAYER_ID", keep="last")
    name_map = {normalize_name(n): int(i) for n, i in zip(latest["PLAYER_NAME"], latest["PLAYER_ID"])}
    out_ids, status = set(), {}
    for _, r in injuries.iterrows():
        pid = match_name(r["PLAYER_NAME"], name_map, cutoff=0.9)
        if pid is None:
            continue
        status[pid] = r["STATUS"]
        if str(r["STATUS"]).lower() in {"out", "doubtful", "suspension", "suspended", "inactive"}:
            out_ids.add(pid)
    return out_ids, status


def generate_picks(day: date, plogs, tlogs, positions, raw_lines: pd.DataFrame,
                   injuries: pd.DataFrame, models: dict, limit: int = 10,
                   current_team: dict | None = None, demo: bool = False) -> dict:
    s = settings
    lines = summarize_lines(raw_lines)
    if lines.empty:
        return _payload(day, [], [], demo, note="No prop lines found for this date.")
    if current_team is None:
        cur = plogs.copy()
        cur["GAME_DATE"] = pd.to_datetime(cur["GAME_DATE"])
        current_team = cur.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last().to_dict()
    upcoming = build_upcoming(lines, plogs, day, current_team)
    if upcoming.empty:
        return _payload(day, [], [], demo, note="No prop players matched to game logs.")
    out_ids, status = injury_sets(injuries, plogs)

    frame, _ = build_features(plogs, tlogs, positions, upcoming.drop(columns="EVENT_ID"),
                                  out_ids=out_ids, current_team=current_team)
    hist = frame[frame["UPCOMING"] == 0]
    up = frame[frame["UPCOMING"] == 1].merge(upcoming[["PLAYER_ID", "EVENT_ID"]], on="PLAYER_ID")
    up = up.reset_index(drop=True)

    recent = hist.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(10)
    rows = []
    for mkt, mm in models.items():
        if mkt not in s.markets:
            continue
        target = MARKETS[mkt][1]
        mu = mm.predict(up)
        ctx_idx = {j for j, f in enumerate(mm.features) if not f.startswith(_BASELINE_PREFIX)}
        contrib = mm.contributions(up)
        ml = lines[lines["MARKET"] == mkt]
        for i, r in up.iterrows():
            hit = ml[(ml["EVENT_ID"] == r["EVENT_ID"]) &
                     (ml["PLAYER_NAME"].map(normalize_name) == normalize_name(r["PLAYER_NAME"]))]
            if hit.empty:
                continue
            ln = hit.iloc[0]
            p_over, p_under = mm.prob_over_under(np.array([mu[i]]), np.array([ln["LINE"]]))
            p_over, p_under = float(p_over[0]), float(p_under[0])
            past = recent.loc[recent["PLAYER_ID"] == r["PLAYER_ID"], target].values
            c = contrib[i][:-1]
            # Show the contextual drivers (matchup, rest, travel, injuries, defense);
            # form/rate features mostly just restate the player's baseline.
            top = [j for j in np.argsort(-np.abs(c)) if j in ctx_idx and abs(c[j]) >= 0.01][:4]
            factors = [{"feature": mm.features[j], "label": label(mm.features[j]),
                        "value": None if pd.isna(r[mm.features[j]]) else round(float(r[mm.features[j]]), 2),
                        "impact": round(float(c[j]), 2)} for j in top]
            for side, p_model, fair, price, book in (
                ("OVER", p_over, ln["FAIR_OVER"], ln["BEST_OVER"], ln["BEST_OVER_BOOK"]),
                ("UNDER", p_under, 1 - ln["FAIR_OVER"], ln["BEST_UNDER"], ln["BEST_UNDER_BOOK"]),
            ):
                p = s.model_weight * p_model + (1 - s.model_weight) * fair
                dec = american_to_decimal(price)
                ev = p * (dec - 1) - (1 - p)
                hits = (past > ln["LINE"]) if side == "OVER" else (past < ln["LINE"])
                rows.append({
                    "player_id": int(r["PLAYER_ID"]), "player": r["PLAYER_NAME"],
                    "team": r["TEAM"], "opponent": r["OPP"], "home": bool(r["HOME"]),
                    "event_id": r["EVENT_ID"], "commence": ln["COMMENCE"],
                    "market": mkt, "side": side, "line": float(ln["LINE"]),
                    "projection": round(float(mu[i]), 2),
                    "model_prob": round(p_model, 4), "market_prob": round(float(fair), 4),
                    "blended_prob": round(p, 4), "edge": round(p_model - float(fair), 4),
                    "ev": round(ev, 4), "price": _fmt_american(price), "book": book,
                    "n_books": int(ln["N_BOOKS"]),
                    "l10_avg": round(float(np.mean(past)), 2) if len(past) else None,
                    "season_avg": _num(r.get(f"form_{target}_SEASON")),
                    "vs_opp_avg": _num(r.get(f"vs_{target}_AVG")) if target in
                    ("PTS", "REB", "AST", "FG3M", "PRA") else None,
                    "hit_rate_l10": round(float(hits.mean()), 2) if len(past) else None,
                    "minutes_l10": _num(r.get("form_MIN_L10")),
                    "rest_days": _num(r.get("sched_rest_days")),
                    "b2b": bool(r.get("sched_b2b", 0)),
                    "teammate_min_out": _num(r.get("inj_team_missing_min")),
                    "status": status.get(int(r["PLAYER_ID"]), "Active"),
                    "factors": factors,
                })
    board = pd.DataFrame(rows)
    if board.empty:
        return _payload(day, [], [], demo, note="No markets could be evaluated.")
    # Keep only the better side per player/market
    board = board.sort_values("ev", ascending=False).drop_duplicates(["player_id", "market"])
    q = board["status"].str.lower().isin(["questionable", "day-to-day", "game time decision"])
    board["questionable"] = q
    board.loc[q, "ev"] = board.loc[q, "ev"] * 0.8
    board = board[~board["player_id"].isin(out_ids)]
    elig = board[(board["blended_prob"] >= s.min_prob) & (board["ev"] >= s.min_ev)
                 & (board["minutes_l10"].fillna(0) >= s.min_recent_minutes)]
    if not s.include_questionable:
        elig = elig[~elig["questionable"]]
    picks, per_player, per_game = [], {}, {}
    for _, r in elig.sort_values("ev", ascending=False).iterrows():
        if per_player.get(r["player_id"], 0) >= s.max_per_player:
            continue
        if per_game.get(r["event_id"], 0) >= s.max_per_game:
            continue
        per_player[r["player_id"]] = per_player.get(r["player_id"], 0) + 1
        per_game[r["event_id"]] = per_game.get(r["event_id"], 0) + 1
        picks.append(r.to_dict())
        if len(picks) >= limit:
            break
    board_out = board.sort_values("ev", ascending=False).head(300).to_dict("records")
    return _payload(day, picks, board_out, demo)


def _num(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _payload(day, picks, board, demo, note=None):
    clean = lambda rows: _sanitize(json.loads(json.dumps(rows, default=_json_default)))
    return {"date": day.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(),
            "demo": demo, "note": note, "picks": clean(picks), "board": clean(board)}


def _sanitize(o):
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, list):
        return [_sanitize(x) for x in o]
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    return str(o)


def save_picks(payload: dict) -> str:
    path = PICKS_DIR / f"{payload['date']}.json"
    path.write_text(json.dumps(payload, indent=1))
    return str(path)


def load_picks(day: date):
    path = PICKS_DIR / f"{day.isoformat()}.json"
    return json.loads(path.read_text()) if path.exists() else None
