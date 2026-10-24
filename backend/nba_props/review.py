"""Look back at a past date: what would the model have projected, and what happened?

The model used for a review is retrained on games *before* that date only, so
nothing from the day being reviewed (or later) leaks into its projections.
Historical sportsbook lines aren't free, so each projection is graded against a
proxy line (player's last-10 average rounded to x.5). If the app saved real
picks for that date, those are graded too, against the real lines.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import model as M
from .config import MARKETS, MODEL_DIR, PICKS_DIR, settings
from .features import build_features
from .picks import _sanitize

log = logging.getLogger(__name__)
ASOF_DIR = MODEL_DIR / "asof"
CONFIDENT = 0.58


def _asof_models(frame, feats, cutoff: pd.Timestamp, base: dict, model_dir) -> dict:
    """Retrain each market on games before `cutoff`, reusing the main model's settings."""
    if (model_dir / "summary.json").exists():
        return M.load(model_dir)
    train = M.training_rows(frame[frame["GAME_DATE"] < cutoff])
    models = {}
    for mkt, mm in base.items():
        d = train.dropna(subset=[mm.target])
        booster = lgb.train(M.PARAMS, lgb.Dataset(d[feats], d[mm.target]), mm.best_iter)
        models[mkt] = replace(mm, features=list(feats), booster=booster)
        log.info("as-of %s model trained for %s", cutoff.date(), mkt)
    M.save(models, {"cutoff": str(cutoff.date())}, model_dir)
    return models


def _grade(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "P"
    return "W" if (actual > line) == (side == "OVER") else "L"


def review_day(day: date, plogs, tlogs, positions, base_models: dict | None = None,
               model_root=ASOF_DIR, picks_dir=PICKS_DIR) -> dict:
    cutoff = pd.Timestamp(day)
    frame, feats = build_features(plogs, tlogs, positions)
    games = frame[(frame["GAME_DATE"] == cutoff) & (frame["UPCOMING"] == 0)]
    # Rotation players only: books don't post props for deep bench guys.
    games = M.training_rows(games)
    games = games[games["form_MIN_L10"] >= settings.min_recent_minutes]
    if games.empty:
        raise RuntimeError(f"No completed games found on {day}.")
    base = base_models or M.load()
    models = _asof_models(frame, feats, cutoff, base, model_root / day.isoformat())

    rows, summary = [], []
    for mkt, mm in models.items():
        target = MARKETS[mkt][1]
        g = games.dropna(subset=[target, f"form_{target}_L10"])
        if g.empty:
            continue
        mu = mm.predict(g)
        y = g[target].to_numpy(float)
        l10 = g[f"form_{target}_L10"].to_numpy(float)
        line = np.floor(l10) + 0.5
        p_over, p_under = mm.prob_over_under(mu, line)
        side = np.where(p_over >= p_under, "OVER", "UNDER")
        prob = np.maximum(p_over, p_under)
        won = np.where(side == "OVER", y > line, y < line)
        conf = prob >= CONFIDENT
        summary.append({
            "market": mkt, "n": int(len(g)),
            "mae_model": round(float(np.mean(np.abs(y - mu))), 2),
            "mae_l10": round(float(np.mean(np.abs(y - l10))), 2),
            "hit_rate": round(float(won.mean()), 3),
            "n_confident": int(conf.sum()),
            "hit_rate_confident": round(float(won[conf].mean()), 3) if conf.any() else None,
        })
        for i, (_, r) in enumerate(g.iterrows()):
            rows.append({
                "player_id": int(r["PLAYER_ID"]), "player": r["PLAYER_NAME"],
                "team": r["TEAM"], "opponent": r["OPP"], "home": bool(r["HOME"]),
                "market": mkt, "projection": round(float(mu[i]), 1), "actual": float(y[i]),
                "l10_avg": round(float(l10[i]), 1), "line": float(line[i]),
                "side": str(side[i]), "prob": round(float(prob[i]), 3),
                "result": "W" if won[i] else "L", "minutes": round(float(r["MIN"]), 1),
            })

    graded = grade_saved_picks(day, plogs, picks_dir)
    out = {"date": day.isoformat(), "line_type": "proxy (L10 avg rounded to .5)",
           "summary": summary, "rows": rows, "graded_picks": graded}
    return _sanitize(json.loads(json.dumps(out, default=str)))


def grade_saved_picks(day: date, plogs, picks_dir=PICKS_DIR) -> dict | None:
    """Grade picks the app actually made for `day` against real lines."""
    f = picks_dir / f"{day.isoformat()}.json"
    if not f.exists():
        return None
    saved = json.loads(f.read_text())
    if saved.get("demo") or not saved.get("picks"):
        return None
    logs = plogs.copy()
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.date
    logs = logs[logs["GAME_DATE"] == day].set_index("PLAYER_ID")
    for c in ("PTS", "REB", "AST"):
        logs[c] = pd.to_numeric(logs[c], errors="coerce")
    logs["PRA"] = logs["PTS"] + logs["REB"] + logs["AST"]
    logs["PR"] = logs["PTS"] + logs["REB"]
    logs["PA"] = logs["PTS"] + logs["AST"]
    out, units = [], 0.0
    for p in saved["picks"]:
        target = MARKETS[p["market"]][1]
        pid = p["player_id"]
        if pid not in logs.index:
            res, actual = "DNP", None
        else:
            actual = float(pd.to_numeric(logs.loc[pid, target], errors="coerce"))
            res = _grade(actual, p["line"], p["side"])
        price = float(p["price"])
        dec = 1 + (price / 100 if price > 0 else 100 / abs(price))
        units += {"W": dec - 1, "L": -1.0}.get(res, 0.0)
        out.append({**{k: p[k] for k in ("player", "market", "side", "line", "projection", "price", "book")},
                    "actual": actual, "result": res})
    rec = {r: sum(1 for x in out if x["result"] == r) for r in ("W", "L", "P", "DNP")}
    return {"picks": out, "record": rec, "units": round(units, 2)}


def save_review(payload: dict):
    (PICKS_DIR / f"review_{payload['date']}.json").write_text(json.dumps(payload))


def load_review(day: date):
    f = PICKS_DIR / f"review_{day.isoformat()}.json"
    return json.loads(f.read_text()) if f.exists() else None
