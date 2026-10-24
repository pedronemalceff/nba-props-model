"""FastAPI service consumed by the React UI.

    uvicorn nba_props.api:app --reload --port 8000
"""
from __future__ import annotations

import logging
import threading
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import model as M
from . import pipeline
from .config import MARKETS, settings
from .picks import load_picks

log = logging.getLogger(__name__)
app = FastAPI(title="NBA Props Model", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_lock = threading.Lock()


def _filter(payload: dict, limit: int, markets: list[str] | None) -> dict:
    out = dict(payload)
    if markets:
        out["picks"] = [p for p in payload["picks"] if p["market"] in markets]
        out["board"] = [p for p in payload["board"] if p["market"] in markets]
    out["picks"] = out["picks"][:limit]
    return out


@app.get("/api/health")
def health():
    return {"ok": True, "odds_api_key": bool(settings.odds_api_key)}


@app.get("/api/meta")
def meta():
    return {"markets": list(MARKETS), "enabled_markets": settings.markets,
            "model": M.summary(), "demo_model": M.summary(pipeline.DEMO_MODEL_DIR),
            "odds_api_key": bool(settings.odds_api_key),
            "settings": {"min_prob": settings.min_prob, "min_ev": settings.min_ev,
                         "model_weight": settings.model_weight}}


@app.get("/api/picks")
def picks(day: date | None = Query(None, alias="date"), limit: int = Query(10, ge=1, le=25),
          markets: str | None = None, demo: bool = False):
    """Cached picks for a date; computes them on first request."""
    mk = markets.split(",") if markets else None
    if demo:
        return _filter(_run(lambda: pipeline.run_demo(25)), limit, mk)
    day = day or pipeline.today_local()
    cached = load_picks(day)
    if cached and not cached.get("demo"):
        return _filter(cached, limit, mk)
    return _filter(_run(lambda: pipeline.run_picks(day, 25, refresh_data=True)), limit, mk)


@app.post("/api/refresh")
def refresh(day: date | None = Query(None, alias="date"), odds: bool = True):
    """Recompute picks; odds=true re-pulls lines and injuries (uses API credits)."""
    day = day or pipeline.today_local()
    return _run(lambda: pipeline.run_picks(day, 25, refresh_odds=odds))


@app.get("/api/review")
def review(day: date = Query(..., alias="date"), refresh: bool = False):
    """What the model would have projected for a past date vs what happened."""
    if day >= pipeline.today_local():
        raise HTTPException(400, "Pick a date before today; games need to be finished.")
    return _run(lambda: pipeline.run_review(day, refresh))


def _run(fn):
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, "A refresh is already running, try again shortly.")
    try:
        return fn()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    finally:
        _lock.release()


# Serve the pre-built React UI at http://localhost:8000 so Node isn't needed to use it.
# Rebuild with `npm run build` in frontend/ and copy dist/ here after UI changes.
_STATIC = Path(__file__).parent / "static"
if _STATIC.exists():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="ui")
