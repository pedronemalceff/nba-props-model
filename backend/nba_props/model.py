"""Per-market LightGBM projections + a calibrated probability layer.

For each market the regressor predicts the expected stat. The prediction is
turned into P(over line) with a distribution fitted on held-out games:
  - normal markets (PTS, PRA, PR, PA): sigma = a + b * mean, fitted on residuals
  - count markets (REB, AST, 3PM): negative binomial, var = mu + alpha * mu^2
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from .config import MARKETS, MODEL_DIR

log = logging.getLogger(__name__)

PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1)


@dataclass
class MarketModel:
    market: str
    target: str
    dist: str
    features: list[str]
    best_iter: int
    sigma_a: float = 0.0
    sigma_b: float = 0.0
    nb_alpha: float = 0.1
    val_mae: float = 0.0
    baseline_mae: float = 0.0
    booster: lgb.Booster | None = None

    # ------------------------------------------------------------- probability
    def prob_over_under(self, mu: np.ndarray, line: np.ndarray):
        """Returns (P(over), P(under)); a whole-number line leaves room for a push."""
        mu = np.maximum(np.asarray(mu, float), 0.05)
        line = np.asarray(line, float)
        over_k = np.floor(line) + 1          # need X >= over_k to go over
        under_k = np.ceil(line) - 1          # need X <= under_k to go under
        if self.dist == "negbin":
            n = 1.0 / self.nb_alpha
            p = n / (n + mu)
            p_over = stats.nbinom.sf(over_k - 1, n, p)
            p_under = stats.nbinom.cdf(under_k, n, p)
        else:
            sigma = np.maximum(self.sigma_a + self.sigma_b * mu, 1.0)
            p_over = 1 - stats.norm.cdf((over_k - 0.5 - mu) / sigma)
            p_under = stats.norm.cdf((under_k + 0.5 - mu) / sigma)
        return p_over, p_under

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.booster.predict(X[self.features], num_iteration=self.best_iter), 0)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X[self.features], num_iteration=self.best_iter, pred_contrib=True)


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training: real games with enough history to have form."""
    return df[(df["UPCOMING"] == 0) & (df["form_games_played"] >= 5)
              & (df["form_MIN_L10"] >= 10) & (df["MIN"] > 0)]


def fit_market(df: pd.DataFrame, features: list[str], market: str, val_frac: float = 0.15,
               num_rounds: int = 2000) -> MarketModel:
    _, target, dist = MARKETS[market]
    d = training_rows(df).dropna(subset=[target]).sort_values("GAME_DATE")
    cut = d["GAME_DATE"].quantile(1 - val_frac)
    tr, va = d[d["GAME_DATE"] < cut], d[d["GAME_DATE"] >= cut]
    dtr = lgb.Dataset(tr[features], tr[target])
    dva = lgb.Dataset(va[features], va[target], reference=dtr)
    booster = lgb.train(PARAMS, dtr, num_rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    best = booster.best_iteration or num_rounds
    pred = np.maximum(booster.predict(va[features], num_iteration=best), 0.05)
    y = va[target].values
    resid = y - pred
    mm = MarketModel(market, target, dist, features, best)
    mm.val_mae = float(np.mean(np.abs(resid)))
    mm.baseline_mae = float(np.nanmean(np.abs(y - va[f"form_{target}_L10"].values)))
    # Heteroscedastic sigma: E|resid| = sigma * sqrt(2/pi)
    scaled = np.abs(resid) * math.sqrt(math.pi / 2)
    b, a = np.polyfit(pred, scaled, 1)
    mm.sigma_a, mm.sigma_b = float(max(a, 0.5)), float(max(b, 0.0))
    alpha = np.mean(resid ** 2 - pred) / np.mean(pred ** 2)
    mm.nb_alpha = float(np.clip(alpha, 0.01, 3.0))
    # Refit on all data with the tuned number of rounds.
    mm.booster = lgb.train(PARAMS, lgb.Dataset(d[features], d[target]), best)
    log.info("%s: val MAE %.3f vs L10-avg baseline %.3f (iters=%d)",
             market, mm.val_mae, mm.baseline_mae, best)
    return mm


def train_all(df: pd.DataFrame, features: list[str], markets=None) -> dict[str, MarketModel]:
    return {m: fit_market(df, features, m) for m in (markets or list(MARKETS))}


def save(models: dict[str, MarketModel], meta: dict | None = None, path=MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)
    summary = {}
    for m, mm in models.items():
        mm.booster.save_model(str(path / f"{m}.txt"))
        info = {k: v for k, v in asdict(mm).items() if k != "booster"}
        joblib.dump(info, path / f"{m}.meta.joblib")
        summary[m] = {k: info[k] for k in ("val_mae", "baseline_mae", "best_iter", "sigma_a",
                                           "sigma_b", "nb_alpha")}
    (path / "summary.json").write_text(json.dumps({"markets": summary, **(meta or {})}, indent=2))


def load(path=MODEL_DIR) -> dict[str, MarketModel]:
    models = {}
    for f in path.glob("*.meta.joblib"):
        info = joblib.load(f)
        info.pop("booster", None)
        mm = MarketModel(**info)
        mm.booster = lgb.Booster(model_file=str(path / f"{mm.market}.txt"))
        models[mm.market] = mm
    if not models:
        raise FileNotFoundError(f"No trained models in {path}. Run `python -m nba_props.cli train`.")
    return models


def summary(path=MODEL_DIR) -> dict:
    f = path / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}
