"""Synthetic league generator for tests and offline demo mode.

Produces data in exactly the same shape as nba_api's LeagueGameLog output, with
real effects baked in (rest, defense, home court, injuries shifting minutes)
so the full pipeline can be exercised and sanity-checked without network.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .teams import TEAMS

ABBRS = list(TEAMS)


def generate(seasons=("2024-25", "2025-26"), seed: int = 7, days_per_season: int = 160,
             current_season_days: int = 0):
    rng = np.random.default_rng(seed)
    team_def = {t: rng.normal(1.0, 0.05) for t in ABBRS}  # >1 = bad defense
    team_pace = {t: rng.normal(1.0, 0.03) for t in ABBRS}
    roster, pid = {}, 1000
    positions = []
    for t in ABBRS:
        players = []
        for i in range(12):
            pos = ["G", "G", "F", "F", "C"][i % 5]
            star = max(0.25, rng.lognormal(0, 0.45) * (1.6 if i < 2 else 1.0 if i < 5 else 0.6))
            players.append({
                "id": pid, "name": f"Player {pid}", "pos": pos,
                "min": float(np.clip(36 - i * 2.6 + rng.normal(0, 2), 8, 38)),
                "pts36": 14 * star + rng.normal(0, 2),
                "reb36": (11 if pos == "C" else 7 if pos == "F" else 4.5) * rng.lognormal(0, 0.2),
                "ast36": (6.5 if pos == "G" else 3 if pos == "F" else 2.2) * rng.lognormal(0, 0.3),
                "fg3r": (0.42 if pos == "G" else 0.33 if pos == "F" else 0.12) * rng.lognormal(0, 0.3),
                "inj": 0,
            })
            positions.append({"PLAYER_ID": pid, "POSITION": pos})
            pid += 1
        roster[t] = players

    all_seasons = list(seasons)
    if current_season_days:
        y = int(seasons[-1][:4]) + 1
        all_seasons.append(f"{y}-{(y + 1) % 100:02d}")
    prows, trows, gid = [], [], 22000000
    for si, season in enumerate(all_seasons):
        start = date(int(season[:4]), 10, 22)
        ndays = current_season_days if (current_season_days and si == len(all_seasons) - 1) else days_per_season
        last_played: dict[str, date] = {}
        for d in range(ndays):
            day = start + timedelta(days=d)
            teams = list(rng.permutation(ABBRS))[:14]
            for k in range(0, len(teams), 2):
                home, away = teams[k], teams[k + 1]
                gid += 1
                for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                    rest = (day - last_played.get(team, day - timedelta(days=5))).days - 1
                    fatigue = 0.94 if rest == 0 else 1.0
                    players = roster[team]
                    for pl in players:  # injury process
                        if pl["inj"] > 0:
                            pl["inj"] -= 1
                        elif rng.random() < 0.04:
                            pl["inj"] = int(rng.integers(1, 8))
                    active = [pl for pl in players if pl["inj"] == 0]
                    missing = sum(pl["min"] for pl in players if pl["inj"] > 0)
                    boost = 1 + 0.6 * missing / 240
                    tot = {k2: 0.0 for k2 in ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                                              "OREB", "DREB", "REB", "AST", "STL", "BLK",
                                              "TOV", "PF", "PTS"]}
                    for pl in active:
                        mins = float(np.clip(pl["min"] * boost * fatigue + rng.normal(0, 3), 2, 44))
                        env = team_def[opp] * team_pace[team] * team_pace[opp] * (1.02 if is_home else 0.98)
                        usage = 1 + 0.5 * (boost - 1)
                        pts = rng.poisson(max(0.1, pl["pts36"] * mins / 36 * env * usage))
                        reb = rng.poisson(max(0.1, pl["reb36"] * mins / 36 * team_pace[opp]))
                        ast = rng.poisson(max(0.1, pl["ast36"] * mins / 36 * usage))
                        fga = max(int(round(pts / 1.15 + rng.normal(0, 1.5))), 0)
                        fg3a = rng.binomial(fga, min(0.9, pl["fg3r"])) if fga else 0
                        fg3m = rng.binomial(fg3a, 0.36) if fg3a else 0
                        fta = rng.poisson(max(0.05, pts * 0.22))
                        ftm = rng.binomial(fta, 0.78) if fta else 0
                        fgm = max(fg3m, int(round((pts - ftm - fg3m) / 2)))
                        pts = 2 * (fgm - fg3m) + 3 * fg3m + ftm
                        fga = max(fga, fgm)
                        oreb = rng.binomial(reb, 0.22) if reb else 0
                        row = {"FGM": fgm, "FGA": fga, "FG3M": fg3m, "FG3A": max(fg3a, fg3m),
                               "FTM": ftm, "FTA": fta, "OREB": oreb, "DREB": reb - oreb,
                               "REB": reb, "AST": ast, "STL": rng.poisson(0.9 * mins / 36),
                               "BLK": rng.poisson(0.6 * mins / 36), "TOV": rng.poisson(1.8 * mins / 36),
                               "PF": rng.poisson(2.5 * mins / 36), "PTS": pts}
                        for k2 in tot:
                            tot[k2] += row[k2]
                        prows.append({
                            "SEASON_ID": f"2{season[:4]}", "PLAYER_ID": pl["id"],
                            "PLAYER_NAME": pl["name"], "TEAM_ABBREVIATION": team,
                            "GAME_ID": f"00{gid}", "GAME_DATE": day.isoformat(),
                            "MATCHUP": f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                            "MIN": round(mins, 1), "PLUS_MINUS": 0, "SEASON": season,
                            "SEASON_TYPE": "Regular Season", **row})
                    trows.append({
                        "SEASON_ID": f"2{season[:4]}", "TEAM_ABBREVIATION": team,
                        "GAME_ID": f"00{gid}", "GAME_DATE": day.isoformat(),
                        "MATCHUP": f"{team} vs. {opp}" if is_home else f"{team} @ {opp}",
                        "MIN": 240, "SEASON": season, "SEASON_TYPE": "Regular Season", **tot})
                    last_played[team] = day
    p = pd.DataFrame(prows)
    t = pd.DataFrame(trows)
    return p, t, pd.DataFrame(positions), roster


def demo_slate(roster, plogs: pd.DataFrame, day: date, n_games: int = 6, seed: int = 11):
    """Fake injury report + sportsbook lines for a demo slate on `day`.

    Lines are set from each player's recent average (like a lazy book would),
    with standard -110/-110 juice and a little noise.
    """
    rng = np.random.default_rng(seed)
    teams = list(rng.permutation(ABBRS))[: n_games * 2]
    recent = plogs.copy()
    recent["GAME_DATE"] = pd.to_datetime(recent["GAME_DATE"])
    recent = recent.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(15)
    recent["PRA"] = recent["PTS"] + recent["REB"] + recent["AST"]
    recent["PR"] = recent["PTS"] + recent["REB"]
    recent["PA"] = recent["PTS"] + recent["AST"]
    avg = recent.groupby("PLAYER_ID")[["PTS", "REB", "AST", "FG3M", "PRA", "PR", "PA", "MIN"]].mean()
    stat_for = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "FG3M",
                "PRA": "PRA", "PR": "PR", "PA": "PA"}
    lines, injuries = [], []
    for k in range(n_games):
        home, away = teams[2 * k], teams[2 * k + 1]
        ev = f"demo{k}"
        for team in (home, away):
            for pl in roster[team]:
                if pl["inj"] > 0:
                    injuries.append({"TEAM": team, "PLAYER_NAME": pl["name"], "STATUS": "Out"})
                    continue
                if pl["id"] not in avg.index or avg.loc[pl["id"], "MIN"] < 20:
                    continue
                for mkt, col in stat_for.items():
                    v = avg.loc[pl["id"], col] * rng.lognormal(0, 0.06)
                    if v < 0.8:
                        continue
                    line = np.floor(v) + 0.5
                    for side in ("OVER", "UNDER"):
                        lines.append({"EVENT_ID": ev, "COMMENCE": f"{day}T23:00:00Z",
                                      "HOME": home, "AWAY": away, "BOOK": "DemoBook",
                                      "MARKET": mkt, "PLAYER_NAME": pl["name"], "LINE": line,
                                      "SIDE": side, "PRICE": -110.0})
    inj = pd.DataFrame(injuries, columns=["TEAM", "PLAYER_NAME", "STATUS"])
    return pd.DataFrame(lines), inj
