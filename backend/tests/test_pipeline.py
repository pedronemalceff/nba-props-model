import numpy as np
import pandas as pd
import pytest

from nba_props.data.injuries import parse_injuries
from nba_props.data.names import match_name, normalize_name
from nba_props.data.odds import american_to_decimal, parse_event_odds, summarize_lines
from nba_props.features import build_features
from nba_props.model import MarketModel
from nba_props.synthetic import generate


@pytest.fixture(scope="module")
def league():
    return generate(seasons=("2025-26",), days_per_season=70, seed=3)


def test_no_leakage(league):
    """Features for a game must not change when later games are removed."""
    p, t, pos, _ = league
    full, feats = build_features(p, t, pos)
    cutoff = sorted(p["GAME_DATE"].unique())[50]
    p2, t2 = p[p["GAME_DATE"] <= cutoff], t[t["GAME_DATE"] <= cutoff]
    trunc, _ = build_features(p2, t2, pos)
    key = ["PLAYER_ID", "GAME_ID"]
    a = full[full["GAME_DATE"] == pd.Timestamp(cutoff)].set_index(key)[feats].sort_index()
    b = trunc[trunc["GAME_DATE"] == pd.Timestamp(cutoff)].set_index(key)[feats].sort_index()
    # Positional-defense league baselines and schedule look-backs are past-only too.
    pd.testing.assert_frame_equal(a, b.loc[a.index], check_dtype=False, atol=1e-9)


def test_upcoming_rows(league):
    p, t, pos, _ = league
    last = pd.to_datetime(p["GAME_DATE"]).max()
    row = p.iloc[-1]
    up = pd.DataFrame([{"PLAYER_ID": row["PLAYER_ID"], "PLAYER_NAME": row["PLAYER_NAME"],
                        "TEAM": row["TEAM_ABBREVIATION"], "OPP": "DEN" if row["TEAM_ABBREVIATION"] != "DEN" else "BOS",
                        "HOME": 0, "GAME_DATE": last + pd.Timedelta(days=1), "GAME_ID": "UP_x"}])
    f, _ = build_features(p, t, pos, up)
    u = f[f["UPCOMING"] == 1]
    assert len(u) == 1
    assert u["geo_altitude_ft"].iloc[0] in (5280, 20)
    assert u["sched_rest_days"].iloc[0] >= 0
    assert u["form_PTS_L10"].notna().all()


def test_absences_raise_missing_minutes(league):
    p, t, pos, _ = league
    f, _ = build_features(p, t, pos)
    assert (f["inj_team_missing_min"] > 0).mean() > 0.2


def test_probabilities_sum_and_monotone():
    mm = MarketModel("PTS", "PTS", "normal", [], 1, sigma_a=2.0, sigma_b=0.2)
    po, pu = mm.prob_over_under(np.array([20.0, 20.0]), np.array([19.5, 24.5]))
    assert np.allclose(po + pu, 1)
    assert po[0] > 0.5 > po[1]
    nb = MarketModel("REB", "REB", "negbin", [], 1, nb_alpha=0.1)
    po, pu = nb.prob_over_under(np.array([8.0]), np.array([8.0]))  # whole line -> push
    assert po[0] + pu[0] < 1


def test_odds_parsing():
    ev = {"id": "e1", "commence_time": "2026-10-21T23:00:00Z", "home_team": "Boston Celtics",
          "away_team": "New York Knicks", "bookmakers": [
              {"key": "dk", "title": "DraftKings", "markets": [{"key": "player_points", "outcomes": [
                  {"name": "Over", "description": "Jalen Brunson", "price": -115, "point": 26.5},
                  {"name": "Under", "description": "Jalen Brunson", "price": -105, "point": 26.5}]}]},
              {"key": "fd", "title": "FanDuel", "markets": [{"key": "player_points", "outcomes": [
                  {"name": "Over", "description": "Jalen Brunson", "price": -110, "point": 26.5},
                  {"name": "Under", "description": "Jalen Brunson", "price": -110, "point": 26.5}]}]}]}
    raw = pd.DataFrame(parse_event_odds(ev))
    assert set(raw["HOME"]) == {"BOS"} and set(raw["MARKET"]) == {"PTS"}
    s = summarize_lines(raw)
    assert len(s) == 1
    r = s.iloc[0]
    assert r["BEST_OVER"] == -110 and r["BEST_OVER_BOOK"] == "FanDuel"
    assert r["BEST_UNDER"] == -105 and 0.45 < r["FAIR_OVER"] < 0.55
    assert american_to_decimal(150) == 2.5


def test_names_and_injuries():
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("Jaren Jackson Jr.") == "jaren jackson"
    cands = {"nikola jokic": 1, "jaren jackson": 2}
    assert match_name("Nikola Jokić", cands) == 1
    assert match_name("Jaren Jackson Jr", cands) == 2
    inj = parse_injuries({"injuries": [{"displayName": "Memphis Grizzlies", "injuries": [
        {"athlete": {"displayName": "Ja Morant"}, "status": "Out", "shortComment": "knee"}]}]})
    assert inj.iloc[0]["TEAM"] == "MEM" and bool(inj.iloc[0]["OUT"])


def test_end_to_end_demo(tmp_path, monkeypatch):
    from datetime import timedelta

    from nba_props import model as M
    from nba_props.picks import generate_picks
    from nba_props.synthetic import demo_slate

    p, t, pos, roster = generate(seasons=("2025-26",), days_per_season=80, seed=5)
    frame, feats = build_features(p, t, pos)
    models = {m: M.fit_market(frame, feats, m, num_rounds=150) for m in ("PTS", "REB")}
    M.save(models, path=tmp_path)
    models = M.load(tmp_path)
    day = (pd.to_datetime(p["GAME_DATE"]).max() + timedelta(days=1)).date()
    lines, inj = demo_slate(roster, p, day)
    out = generate_picks(day, p, t, pos, lines, inj, models, limit=8, demo=True)
    assert out["board"], "board should not be empty"
    assert len(out["picks"]) <= 8
    for pk in out["picks"]:
        assert pk["market"] in ("PTS", "REB") and pk["side"] in ("OVER", "UNDER")
        assert pk["factors"] and 0 < pk["blended_prob"] < 1
    players = [pk["player_id"] for pk in out["picks"]]
    assert len(players) == len(set(players))
