"""Command line entry point.

    python -m nba_props.cli update            # fetch/refresh game logs
    python -m nba_props.cli train             # train all market models
    python -m nba_props.cli picks --limit 10  # today's best props -> data/picks/<date>.json
    python -m nba_props.cli backtest          # hold-out evaluation
    python -m nba_props.cli demo              # synthetic end-to-end run, no network
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from . import pipeline


def _date(s: str) -> date:
    return date.fromisoformat(s)


def _print_picks(payload: dict):
    tag = " (DEMO DATA)" if payload.get("demo") else ""
    print(f"\nTop props for {payload['date']}{tag}")
    if payload.get("note"):
        print(payload["note"])
    for i, p in enumerate(payload["picks"], 1):
        print(f"{i:>2}. {p['player']:<24} {p['market']:<4} {p['side']:<5} {p['line']:>5}  "
              f"proj {p['projection']:>5}  p={p['blended_prob']:.0%}  EV {p['ev']:+.1%}  "
              f"{p['price']} @ {p['book']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nba_props")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update"); u.add_argument("--date", type=_date)
    t = sub.add_parser("train"); t.add_argument("--date", type=_date)
    t.add_argument("--no-refresh", action="store_true")
    pk = sub.add_parser("picks"); pk.add_argument("--date", type=_date)
    pk.add_argument("--limit", type=int, default=10)
    pk.add_argument("--refresh-odds", action="store_true", help="re-pull lines (costs credits)")
    b = sub.add_parser("backtest"); b.add_argument("--days", type=int, default=45)
    b.add_argument("--demo", action="store_true")
    d = sub.add_parser("demo"); d.add_argument("--limit", type=int, default=10)
    d.add_argument("--retrain", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if a.cmd == "update":
        day = a.date or pipeline.today_local()
        p, t_, _, _ = pipeline.load_data(day, refresh=True)
        print(f"Player rows: {len(p):,}  Team rows: {len(t_):,}  "
              f"Latest game: {pd.to_datetime(p['GAME_DATE']).max().date()}")
    elif a.cmd == "train":
        models = pipeline.train(a.date, refresh=not a.no_refresh)
        for m, mm in models.items():
            print(f"{m:<4} val MAE {mm.val_mae:.2f}  (L10 baseline {mm.baseline_mae:.2f})")
    elif a.cmd == "picks":
        _print_picks(pipeline.run_picks(a.date, a.limit, a.refresh_odds))
    elif a.cmd == "backtest":
        from .features import build_features
        if a.demo:
            from .synthetic import generate
            p, t_, pos, _ = generate()
        else:
            p, t_, pos, _ = pipeline.load_data(pipeline.today_local(), refresh=False)
        frame, feats = build_features(p, t_, pos)
        res = pipeline.backtest(frame, feats, a.days)
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(res.to_string(index=False))
    elif a.cmd == "demo":
        _print_picks(pipeline.run_demo(a.limit, a.retrain))


if __name__ == "__main__":
    main()
