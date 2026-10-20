"""Feature engineering.

Every feature for a player-game is computed only from information available
*before* tip-off, so the same code path builds the training set (historical
games) and the inference rows (today's games appended as placeholder rows).

Feature groups
--------------
form_*      previous games: rolling L5/L10/L20, season-to-date, EWMA, volatility
rate_*      per-minute production
adv_*       advanced stats: usage rate, true shooting, shot volume
vs_*        matchup history vs. tonight's opponent
sched_*     rest days, back-to-backs, games in last 7 days
geo_*       travel miles, time-zone shift, altitude, home/away, road-trip length
tm_*        own team offense: ORtg, pace, scoring/assist/3PA volume
opp_*       opponent defense: DRtg, pace, stats allowed
pdef_*      opponent stats allowed to the player's position (per 36)
inj_*       teammates / opponents missing (minutes and usage vacated)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from .config import season_for
from .teams import TEAMS, haversine_miles

TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA", "PR", "PA"]
BOX = ["MIN", "FGA", "FG3A", "FTA", "TOV", "OREB", "DREB", "USG", "TS"]
POS_CODE = {"G": 0, "F": 1, "C": 2}
FAR_FUTURE = pd.Timestamp("2100-01-01")


# --------------------------------------------------------------------------- prep
def _parse_min(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        def conv(v):
            if isinstance(v, str) and ":" in v:
                m, sec = v.split(":")
                return float(m) + float(sec) / 60
            return pd.to_numeric(v, errors="coerce")
        return s.map(conv).astype(float)
    return s.astype(float)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN"] = _parse_min(df["MIN"])
    df["TEAM"] = df["TEAM_ABBREVIATION"]
    df["HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    df["OPP"] = df["MATCHUP"].str.split().str[-1]
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    if "SEASON" not in df.columns:
        df["SEASON"] = df["GAME_DATE"].map(lambda d: season_for(d.date()))
    return df


def prep_team(tlogs: pd.DataFrame) -> pd.DataFrame:
    t = _base(tlogs)
    t["POSS"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp = t[["GAME_ID", "TEAM", "PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "OREB",
             "DREB", "TOV", "POSS"]].rename(columns=lambda c: f"O_{c}" if c != "GAME_ID" else c)
    t = t.merge(opp, left_on=["GAME_ID", "OPP"], right_on=["GAME_ID", "O_TEAM"], how="left")
    t["ORTG"] = 100 * t["PTS"] / t["POSS"]
    t["DRTG"] = 100 * t["O_PTS"] / t["O_POSS"]
    t["PACE"] = (t["POSS"] + t["O_POSS"]) / 2 * 48 / (t["MIN"] / 5).clip(lower=48)
    t["OREB_PCT"] = t["OREB"] / (t["OREB"] + t["O_DREB"])
    t["DREB_PCT"] = t["DREB"] / (t["DREB"] + t["O_OREB"])
    return t.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)


def prep_player(plogs: pd.DataFrame, team: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    p = _base(plogs)
    p = p[p["MIN"] > 0].copy()
    p["PRA"] = p["PTS"] + p["REB"] + p["AST"]
    p["PR"] = p["PTS"] + p["REB"]
    p["PA"] = p["PTS"] + p["AST"]
    tm = team[["TEAM", "GAME_ID", "MIN", "FGA", "FTA", "TOV"]].rename(
        columns={"MIN": "T_MIN", "FGA": "T_FGA", "FTA": "T_FTA", "TOV": "T_TOV"})
    p = p.merge(tm, on=["TEAM", "GAME_ID"], how="left")
    p["USG"] = 100 * ((p["FGA"] + 0.44 * p["FTA"] + p["TOV"]) * (p["T_MIN"] / 5)) / (
        p["MIN"] * (p["T_FGA"] + 0.44 * p["T_FTA"] + p["T_TOV"]))
    tsa = 2 * (p["FGA"] + 0.44 * p["FTA"])
    p["TS"] = np.where(tsa > 0, p["PTS"] / tsa.replace(0, np.nan), np.nan)
    p = p.drop(columns=["T_MIN", "T_FGA", "T_FTA", "T_TOV"])
    pos = positions.drop_duplicates("PLAYER_ID").copy() if len(positions) else positions
    if len(pos):
        pos["POS"] = pos["POSITION"].fillna("").astype(str).str[:1].str.upper()
        p = p.merge(pos[["PLAYER_ID", "POS"]], on="PLAYER_ID", how="left")
    else:
        p["POS"] = np.nan
    p["POS"] = p["POS"].where(p["POS"].isin(list(POS_CODE)), "F")
    return p


# ---------------------------------------------------------------------- helpers
def _grouped_roll(df, keys, cols, window, how="mean", min_periods=1, shift=True):
    g = df.groupby(keys, sort=False)[cols]
    src = g.shift(1) if shift else df[cols]
    src = src.copy()
    for k in ([keys] if isinstance(keys, str) else keys):
        src[k] = df[k]
    r = src.groupby(keys, sort=False)[cols].rolling(window, min_periods=min_periods)
    r = getattr(r, how)()
    return r.reset_index(level=list(range(r.index.nlevels - 1)), drop=True).sort_index()


def _grouped_expanding_mean(df, keys, cols):
    src = df.groupby(keys, sort=False)[cols].shift(1)
    for k in keys:
        src[k] = df[k]
    r = src.groupby(keys, sort=False)[cols].expanding().mean()
    return r.reset_index(level=list(range(r.index.nlevels - 1)), drop=True).sort_index()


def _asof(left: pd.DataFrame, right: pd.DataFrame, left_by: str, right_by: str, cols, prefix):
    """Attach the latest right-row strictly before each left row's GAME_DATE."""
    r = right[[right_by, "GAME_DATE"] + cols].rename(columns={c: prefix + c for c in cols})
    r = r.rename(columns={right_by: "_KEY"}).sort_values("GAME_DATE")
    r["_KEY"] = r["_KEY"].astype(str).astype(object)
    l = left.assign(_KEY=left[left_by].astype(str).astype(object)).reset_index().sort_values("GAME_DATE")
    out = pd.merge_asof(l, r, on="GAME_DATE", by="_KEY", allow_exact_matches=False)
    return out.set_index("index").sort_index().drop(columns="_KEY")


# ------------------------------------------------------------------- schedule/geo
def schedule_features(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)
    s["VENUE"] = np.where(s["HOME"] == 1, s["TEAM"], s["OPP"])
    g = s.groupby("TEAM", sort=False)
    prev_date = g["GAME_DATE"].shift(1)
    prev_venue = g["VENUE"].shift(1)
    gap = (s["GAME_DATE"] - prev_date).dt.days - 1
    s["sched_rest_days"] = gap.where(gap < 30, 7).fillna(7).clip(0, 7)
    s["sched_b2b"] = (s["sched_rest_days"] == 0).astype(int)

    def last7(grp):
        d = grp["GAME_DATE"].values.astype("datetime64[D]").astype(np.int64)
        lo = np.searchsorted(d, d - 7, side="left")
        return pd.Series(np.arange(len(d)) - lo, index=grp.index)
    s["sched_games_last7"] = g[["GAME_DATE"]].apply(last7).reset_index(level=0, drop=True)

    lat = s["VENUE"].map(lambda t: TEAMS[t][1])
    lon = s["VENUE"].map(lambda t: TEAMS[t][2])
    plat = prev_venue.map(lambda t: TEAMS[t][1] if isinstance(t, str) else np.nan)
    plon = prev_venue.map(lambda t: TEAMS[t][2] if isinstance(t, str) else np.nan)
    miles = [haversine_miles(a, b, c, d) if not np.isnan(c) else 0.0
             for a, b, c, d in zip(lat, lon, plat, plon)]
    s["geo_travel_miles"] = np.where(gap < 30, miles, 0.0)
    tz = s["VENUE"].map(lambda t: TEAMS[t][4])
    ptz = prev_venue.map(lambda t: TEAMS[t][4] if isinstance(t, str) else np.nan)
    s["geo_tz_shift"] = (tz - ptz).fillna(0)
    s["geo_altitude_ft"] = s["VENUE"].map(lambda t: TEAMS[t][3])
    s["geo_high_altitude"] = (s["geo_altitude_ft"] > 4000).astype(int)
    s["geo_home"] = s["HOME"]
    away = (s["HOME"] == 0).astype(int)
    block = (away != g["HOME"].shift(1).eq(0).astype(int)).cumsum()
    s["geo_road_trip_len"] = away.groupby([s["TEAM"], block]).cumsum()
    return s


# ---------------------------------------------------------------- team strength
TEAM_ROLL = ["ORTG", "DRTG", "PACE", "PTS", "AST", "FG3A", "REB", "OREB_PCT", "DREB_PCT",
             "O_PTS", "O_REB", "O_AST", "O_FG3M", "O_FG3A", "O_TOV"]


def team_strength(team: pd.DataFrame) -> pd.DataFrame:
    t = team.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)
    out = t[["TEAM", "GAME_DATE"]].copy()
    for w in (10, 30):
        r = _grouped_roll(t, "TEAM", TEAM_ROLL, w, shift=False)
        for c in TEAM_ROLL:
            out[f"{c}_L{w}"] = r[c]
    return out


# ------------------------------------------------------------- positional defense
PDEF_STATS = ["PTS", "REB", "AST", "FG3M"]


def positional_defense(p: pd.DataFrame) -> pd.DataFrame:
    agg = (p.groupby(["OPP", "POS", "GAME_ID", "GAME_DATE"])[PDEF_STATS + ["MIN"]]
           .sum().reset_index().sort_values(["OPP", "POS", "GAME_DATE"]).reset_index(drop=True))
    sums = _grouped_roll(agg, ["OPP", "POS"], PDEF_STATS + ["MIN"], 15, how="sum", shift=False)
    # League baseline per position from all games up to that date.
    daily = agg.groupby(["POS", "GAME_DATE"])[PDEF_STATS + ["MIN"]].sum().groupby(level=0).cumsum()
    out = agg[["OPP", "POS", "GAME_DATE"]].copy()
    base = daily.reindex(pd.MultiIndex.from_frame(agg[["POS", "GAME_DATE"]])).reset_index(drop=True)
    for c in PDEF_STATS:
        rate = 36 * sums[c] / sums["MIN"]
        league = 36 * base[c] / base["MIN"]
        out[f"{c}_PER36"] = rate.values
        out[f"{c}_FACTOR"] = (rate.values / league.values)
    out["KEY"] = out["OPP"] + "_" + out["POS"]
    return out


# --------------------------------------------------------------------- absences
def absence_features(p_hist, sched, current_team: dict, out_ids: set) -> pd.DataFrame:
    """Minutes/usage vacated by rotation players who are not playing.

    Historical games: a rotation player who did not log minutes is "missing".
    Upcoming games: players flagged Out/Doubtful on the injury report.
    """
    spans = (p_hist.groupby(["PLAYER_ID", "TEAM", "SEASON"])["GAME_DATE"]
             .agg(["min", "max"]).reset_index())
    latest_season = sched["SEASON"].max()
    is_cur = spans["SEASON"] == latest_season
    cur_team = spans["PLAYER_ID"].map(current_team)
    spans.loc[is_cur & (spans["TEAM"] == cur_team), "max"] = FAR_FUTURE
    # Players whose current team differs from logs (e.g. off-season trade).
    have = set(zip(spans.loc[is_cur, "PLAYER_ID"], spans.loc[is_cur, "TEAM"]))
    extra = [{"PLAYER_ID": pid, "TEAM": tm, "SEASON": latest_season,
              "min": pd.Timestamp("1900-01-01"), "max": FAR_FUTURE}
             for pid, tm in current_team.items() if (pid, tm) not in have and tm in TEAMS]
    if extra:
        spans = pd.concat([spans, pd.DataFrame(extra)], ignore_index=True)

    tg = sched[["TEAM", "GAME_ID", "GAME_DATE", "SEASON", "UPCOMING"]]
    grid = spans.merge(tg, on=["TEAM", "SEASON"])
    grid = grid[(grid["GAME_DATE"] >= grid["min"]) & (grid["GAME_DATE"] <= grid["max"])]
    grid = grid.merge(p_hist[["PLAYER_ID", "GAME_ID", "MIN", "USG"]],
                      on=["PLAYER_ID", "GAME_ID"], how="left")
    grid.loc[grid["UPCOMING"] == 0, "MIN"] = grid.loc[grid["UPCOMING"] == 0, "MIN"].fillna(0)
    grid = grid.sort_values(["PLAYER_ID", "TEAM", "GAME_DATE"]).reset_index(drop=True)
    grid["PT"] = grid["PLAYER_ID"].astype(str) + "_" + grid["TEAM"]
    grid["EXP_MIN"] = _grouped_roll(grid, "PT", ["MIN"], 10, min_periods=3)["MIN"].fillna(0)
    grid["EXP_USG"] = _grouped_roll(grid, "PT", ["USG"], 10)["USG"].fillna(0)
    absent = np.where(grid["UPCOMING"] == 1, grid["PLAYER_ID"].isin(out_ids), grid["MIN"] == 0)
    rot = grid["EXP_MIN"] >= 12
    m = absent & rot
    grid["m_min"] = np.where(m, grid["EXP_MIN"], 0)
    grid["m_usg"] = np.where(m, grid["EXP_USG"] * grid["EXP_MIN"] / 48, 0)
    grid["m_start"] = (absent & (grid["EXP_MIN"] >= 26)).astype(int)
    grid["m_top_usg"] = np.where(m, grid["EXP_USG"], 0)
    out = grid.groupby(["TEAM", "GAME_ID"]).agg(
        missing_min=("m_min", "sum"), missing_usg=("m_usg", "sum"),
        starters_out=("m_start", "sum"), top_usg_out=("m_top_usg", "max")).reset_index()
    return out


# ------------------------------------------------------------------------ build
def build_features(plogs, tlogs, positions, upcoming: pd.DataFrame | None = None,
                   out_ids: set | None = None, current_team: dict | None = None):
    """Return (frame, feature_columns).

    upcoming: rows [PLAYER_ID, PLAYER_NAME, TEAM, OPP, HOME, GAME_DATE, GAME_ID]
    for games to predict; they come back with UPCOMING == 1.
    """
    out_ids = out_ids or set()
    team = prep_team(tlogs)
    p = prep_player(plogs, team, positions)
    if current_team is None:
        current_team = p.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM"].last().to_dict()

    p["UPCOMING"] = 0
    if upcoming is not None and len(upcoming):
        up = upcoming.copy()
        up["GAME_DATE"] = pd.to_datetime(up["GAME_DATE"])
        up["GAME_ID"] = up["GAME_ID"].astype(str)
        up["SEASON"] = up["GAME_DATE"].map(lambda d: season_for(d.date()))
        up["UPCOMING"] = 1
        pos_map = p.drop_duplicates("PLAYER_ID", keep="last").set_index("PLAYER_ID")["POS"]
        up["POS"] = up["PLAYER_ID"].map(pos_map).fillna("F")
        p = pd.concat([p, up], ignore_index=True)

    # ---- schedule (historical team games + upcoming games for both teams)
    sched = team[["TEAM", "OPP", "HOME", "GAME_DATE", "GAME_ID", "SEASON"]].assign(UPCOMING=0)
    if upcoming is not None and len(upcoming):
        ug = up[["TEAM", "OPP", "HOME", "GAME_DATE", "GAME_ID", "SEASON"]].drop_duplicates(
            ["TEAM", "GAME_ID"])
        mirror = ug.rename(columns={"TEAM": "OPP", "OPP": "TEAM"}).assign(HOME=lambda d: 1 - d["HOME"])
        ug = pd.concat([ug, mirror]).drop_duplicates(["TEAM", "GAME_ID"]).assign(UPCOMING=1)
        sched = pd.concat([sched, ug], ignore_index=True)
    sched = schedule_features(sched)

    # ---- player rolling form
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    feats: list[str] = []
    stat_cols = TARGETS + BOX
    for w in (5, 10, 20):
        r = _grouped_roll(p, "PLAYER_ID", stat_cols, w)
        for c in stat_cols:
            name = f"form_{c}_L{w}" if c in TARGETS or c == "MIN" else f"adv_{c}_L{w}"
            p[name] = r[c]
            feats.append(name)
    r = _grouped_expanding_mean(p, ["PLAYER_ID", "SEASON"], stat_cols)
    for c in stat_cols:
        p[f"form_{c}_SEASON"] = r[c]
        feats.append(f"form_{c}_SEASON")
    sd = _grouped_roll(p, "PLAYER_ID", TARGETS + ["MIN"], 10, how="std", min_periods=3)
    for c in TARGETS + ["MIN"]:
        p[f"form_{c}_STD10"] = sd[c]
        feats.append(f"form_{c}_STD10")
    shifted = p.groupby("PLAYER_ID")[TARGETS + ["MIN"]].shift(1)
    shifted["PLAYER_ID"] = p["PLAYER_ID"]
    ew = shifted.groupby("PLAYER_ID")[TARGETS + ["MIN"]].transform(
        lambda x: x.ewm(halflife=5, ignore_na=True).mean())
    for c in TARGETS + ["MIN"]:
        p[f"form_{c}_EWM"] = ew[c]
        feats.append(f"form_{c}_EWM")
    sums = _grouped_roll(p, "PLAYER_ID", TARGETS + ["MIN"], 10, how="sum")
    for c in TARGETS:
        p[f"rate_{c}_PER36_L10"] = 36 * sums[c] / sums["MIN"].replace(0, np.nan)
        feats.append(f"rate_{c}_PER36_L10")
    p["form_games_played"] = p.groupby("PLAYER_ID").cumcount()
    p["form_games_season"] = p.groupby(["PLAYER_ID", "SEASON"]).cumcount()
    last = p.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    p["sched_player_days_off"] = ((p["GAME_DATE"] - last).dt.days - 1).clip(0, 30).fillna(30)
    p["form_min_trend"] = p["form_MIN_L5"] - p["form_MIN_L20"]
    feats += ["form_games_played", "form_games_season", "sched_player_days_off", "form_min_trend"]

    # ---- matchup history vs this opponent
    vs_cols = ["PTS", "REB", "AST", "FG3M", "PRA", "MIN"]
    p = p.sort_values(["PLAYER_ID", "OPP", "GAME_DATE"]).reset_index(drop=True)
    r = _grouped_roll(p, ["PLAYER_ID", "OPP"], vs_cols, 6)
    for c in vs_cols:
        p[f"vs_{c}_AVG"] = r[c]
        feats.append(f"vs_{c}_AVG")
    p["vs_games"] = p.groupby(["PLAYER_ID", "OPP"]).cumcount()
    feats.append("vs_games")
    for c in ["PTS", "REB", "AST", "PRA"]:
        p[f"vs_{c}_DIFF"] = p[f"vs_{c}_AVG"] - p[f"form_{c}_L20"]
        feats.append(f"vs_{c}_DIFF")
    # ---- home/away split (player's recent production in the same venue type)
    split_cols = ["PTS", "REB", "AST", "FG3M", "PRA"]
    p = p.sort_values(["PLAYER_ID", "HOME", "GAME_DATE"]).reset_index(drop=True)
    r = _grouped_roll(p, ["PLAYER_ID", "HOME"], split_cols, 15)
    for c in split_cols:
        p[f"geo_{c}_HOME_AWAY_SPLIT"] = r[c] - p[f"form_{c}_L20"]
        feats.append(f"geo_{c}_HOME_AWAY_SPLIT")
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # ---- schedule / geography (own + opponent rest)
    sc = [c for c in sched.columns if c.startswith(("sched_", "geo_"))]
    p = p.merge(sched[["TEAM", "GAME_ID"] + sc], on=["TEAM", "GAME_ID"], how="left")
    opp_rest = sched[["TEAM", "GAME_ID", "sched_rest_days", "sched_b2b", "geo_travel_miles"]].rename(
        columns={"TEAM": "OPP", "sched_rest_days": "sched_opp_rest_days",
                 "sched_b2b": "sched_opp_b2b", "geo_travel_miles": "geo_opp_travel_miles"})
    p = p.merge(opp_rest, on=["OPP", "GAME_ID"], how="left")
    p["sched_rest_adv"] = p["sched_rest_days"] - p["sched_opp_rest_days"]
    feats += sc + ["sched_opp_rest_days", "sched_opp_b2b", "geo_opp_travel_miles", "sched_rest_adv"]

    # ---- team offense (own) and defense (opponent), as of the prior game
    ts = team_strength(team)
    tcols = [c for c in ts.columns if c not in ("TEAM", "GAME_DATE")]
    own = [c for c in tcols if not c.startswith("O_") and not c.startswith("DRTG")
           and not c.startswith("DREB")]
    dfn = [c for c in tcols if c.startswith(("O_", "DRTG", "PACE", "DREB_PCT"))]
    p = _asof(p, ts, "TEAM", "TEAM", own, "tm_")
    p = _asof(p, ts, "OPP", "TEAM", dfn, "opp_")
    feats += [f"tm_{c}" for c in own] + [f"opp_{c}" for c in dfn]

    # ---- opponent defense vs the player's position
    pdef = positional_defense(p[p["UPCOMING"] == 0])
    p["_PK"] = p["OPP"] + "_" + p["POS"]
    pcols = [c for c in pdef.columns if c.endswith(("_PER36", "_FACTOR"))]
    p = _asof(p, pdef, "_PK", "KEY", pcols, "pdef_")
    feats += [f"pdef_{c}" for c in pcols]

    # ---- injuries / absences
    hist = p[p["UPCOMING"] == 0]
    ab = absence_features(hist, sched, current_team, out_ids)
    acols = ["missing_min", "missing_usg", "starters_out", "top_usg_out"]
    p = p.merge(ab.rename(columns={c: f"inj_team_{c}" for c in acols}), on=["TEAM", "GAME_ID"], how="left")
    p = p.merge(ab.rename(columns={"TEAM": "OPP", **{c: f"inj_opp_{c}" for c in acols}}),
                on=["OPP", "GAME_ID"], how="left")
    feats += [f"inj_team_{c}" for c in acols] + [f"inj_opp_{c}" for c in acols]
    # Player's share of vacated usage scales with his own role.
    p["inj_usg_opportunity"] = p["inj_team_missing_usg"] * p["adv_USG_L10"] / 100
    feats.append("inj_usg_opportunity")

    p["pos_code"] = p["POS"].map(POS_CODE).fillna(1)
    feats.append("pos_code")
    p = p.drop(columns=["_PK"])
    feats = list(dict.fromkeys(feats))
    return p, feats
