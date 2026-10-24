"""High-level jobs: update data, train, generate picks, backtest, demo."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import model as M
from .config import MARKETS, MODEL_DIR, season_for, settings
from .features import build_features
from .picks import generate_picks, save_picks

log = logging.getLogger(__name__)
DEMO_MODEL_DIR = MODEL_DIR / "demo"


def today_local() -> date:
    return datetime.now(ZoneInfo(settings.timezone)).date()


def load_data(day: date, refresh: bool = True):
    from .data.nba_stats import load_all, load_rosters

    plogs, tlogs, pos = load_all(day, settings.history_seasons, refresh_current=refresh)
    p = plogs.copy()
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])
    current_team = p.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last().to_dict()
    current_team.update(load_rosters(season_for(day), refresh=refresh))
    return plogs, tlogs, pos, current_team


def train(day: date | None = None, refresh: bool = True, model_dir=MODEL_DIR, data=None):
    day = day or today_local()
    plogs, tlogs, pos, _ = data or load_data(day, refresh)
    frame, feats = build_features(plogs, tlogs, pos)
    models = M.train_all(frame, feats, settings.markets)
    M.save(models, {"trained_on": day.isoformat(), "rows": int(len(frame)),
                    "n_features": len(feats)}, model_dir)
    return models


def run_picks(day: date | None = None, limit: int = 10, refresh_odds: bool = False,
              refresh_data: bool = True) -> dict:
    from .data.injuries import fetch_injuries
    from .data.odds import fetch_props

    day = day or today_local()
    plogs, tlogs, pos, current_team = load_data(day, refresh_data)
    models = M.load()
    lines = fetch_props(day, refresh=refresh_odds)
    inj = fetch_injuries(day, refresh=refresh_odds)
    payload = generate_picks(day, plogs, tlogs, pos, lines, inj, models, limit,
                             current_team=current_team)
    save_picks(payload)
    return payload


# ------------------------------------------------------------------------- demo
def run_demo(limit: int = 10, retrain: bool = False) -> dict:
    """Full pipeline on a synthetic league: no network or API key needed."""
    from .synthetic import demo_slate, generate

    plogs, tlogs, pos, roster = generate(current_season_days=12)
    day = (pd.to_datetime(plogs["GAME_DATE"]).max() + timedelta(days=1)).date()
    if retrain or not (DEMO_MODEL_DIR / "summary.json").exists():
        train(day, data=(plogs, tlogs, pos, None), model_dir=DEMO_MODEL_DIR)
    models = M.load(DEMO_MODEL_DIR)
    lines, inj = demo_slate(roster, plogs, day)
    payload = generate_picks(day, plogs, tlogs, pos, lines, inj, models, limit, demo=True)
    save_picks(payload)
    return payload


# --------------------------------------------------------------------- backtest
def backtest(frame: pd.DataFrame, feats: list[str], test_days: int = 45, markets=None) -> pd.DataFrame:
    """Train on everything before the last `test_days`, evaluate after.

    Without historical sportsbook lines we use a proxy line (player's L10 average
    rounded to x.5), which is roughly where books open. Reports projection MAE vs
    naive baselines, and hit rate / Brier score of the model's over/under calls.
    """
    cut = frame.loc[frame["UPCOMING"] == 0, "GAME_DATE"].max() - pd.Timedelta(days=test_days)
    train_df = frame[frame["GAME_DATE"] < cut]
    test = M.training_rows(frame[frame["GAME_DATE"] >= cut])
    out = []
    for mkt in markets or list(MARKETS):
        target = MARKETS[mkt][1]
        mm = M.fit_market(train_df, feats, mkt)
        t = test.dropna(subset=[target, f"form_{target}_L10"])
        mu = mm.predict(t)
        y = t[target].values
        line = np.floor(t[f"form_{target}_L10"].values) + 0.5
        p_over, p_under = mm.prob_over_under(mu, line)
        went_over = (y > line).astype(float)
        conf = np.maximum(p_over, p_under)
        call_over = p_over >= p_under
        correct = np.where(call_over, y > line, y < line)
        strong = conf >= 0.58
        out.append({
            "market": mkt, "n": len(t),
            "mae_model": np.mean(np.abs(y - mu)),
            "mae_l10": np.mean(np.abs(y - t[f"form_{target}_L10"].values)),
            "mae_season": np.nanmean(np.abs(y - t[f"form_{target}_SEASON"].values)),
            "brier": np.mean((p_over - went_over) ** 2),
            "hit_rate_all": correct.mean(),
            "hit_rate_conf58": correct[strong].mean() if strong.any() else np.nan,
            "n_conf58": int(strong.sum()),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------- review
def run_review(day: date, refresh: bool = False) -> dict:
    """Projections vs actual results for a past date (see review.py)."""
    from .review import load_review, review_day, save_review

    if not refresh:
        cached = load_review(day)
        if cached:
            return cached
    plogs, tlogs, pos, _ = load_data(today_local(), refresh=True)
    payload = review_day(day, plogs, tlogs, pos)
    if day < today_local():
        save_review(payload)
    return payload
