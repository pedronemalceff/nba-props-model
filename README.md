# NBA Props Model

Projects NBA player stat lines, compares them to sportsbook prop lines, and surfaces the day's **5–10 best player props** ranked by expected value. Python model (LightGBM) behind a FastAPI service, with a React UI.

Markets: Points, Rebounds, Assists, 3PM, PRA, P+R, P+A.

## How it works

```
nba_api game logs ─┐
ESPN injuries ─────┼─> features.py ─> LightGBM per market ─> P(over/under) ─┐
                   │                                                         ├─> picks.py ─> FastAPI ─> React UI
The Odds API ──────┴──────────────── best price + no-vig market prob ───────┘
```

1. **Data**: one `LeagueGameLog` call per season pulls every player and team box score (last 3 seasons + current, regular season + playoffs). Cached as parquet.
2. **Features** (all computed from games *before* tip-off; a unit test enforces no leakage):

   | Factor | Features |
   |---|---|
   | Previous games | L5/L10/L20, season-to-date, EWMA, volatility, minutes trend, per-36 rates |
   | Matchup stats | player's avg vs this opponent (last 6 meetings), diff vs his norm, games vs opp |
   | Geography | travel miles since last game, time-zone shift, venue altitude (DEN/UTA), home/away, road-trip length, player's home/away split |
   | Rest | days of rest, back-to-back, games in last 7 days, opponent rest, rest advantage, player days off |
   | Advanced stats | usage rate, true shooting %, FGA/3PA/FTA volume |
   | Opponent defense | DRtg, pace, points/rebounds/assists/3s allowed (L10/L30), stats allowed **to the player's position** per 36 and vs league avg |
   | Own team offense | ORtg, pace, scoring, assists, 3PA volume, OREB% |
   | Injuries | teammate minutes & usage vacated, starters out, top-usage teammate out, same for opponent |

   The injury features are learned historically from who actually sat (rotation players with 0 minutes). On game day, players listed Out/Doubtful on ESPN's injury report fill the same features, so the model knows e.g. that a star being out bumps a teammate's usage.
3. **Model**: one LightGBM regressor per market predicts the expected stat. Held-out residuals calibrate a distribution, normal with heteroscedastic sigma for PTS/combos and negative binomial for REB/AST/3PM. That gives P(over) and P(under), with push handling on whole-number lines.
4. **Ranking**: model probability is blended with the market's no-vig probability (`MODEL_WEIGHT`, default 0.65), then EV is computed against the best available price across books. Filters: min probability, min EV, min recent minutes, no Out/Doubtful players, Questionable players penalized. Diversified to max 1 prop per player and 3 per game.
5. **Explanations**: each pick shows its top contextual drivers (SHAP contributions from LightGBM), plus L10, season, vs-opponent averages and L10 hit rate.

## Use the app

After setup (below), double-click **start.bat** in the project folder. It starts the model and opens the UI at http://localhost:8000. Keep the black window open while you use it. Use the **Refresh lines** button to pull today's lines and injuries.

The built UI lives in `backend/nba_props/static`, so Node is only needed if you change the frontend (`npm run build`, then copy `frontend/dist` into that folder).

## Quick start

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Try it with no network / API key (synthetic league)
python -m nba_props.cli demo

# frontend (second terminal)
cd frontend
npm install
npm run dev            # http://localhost:5173 (proxies /api to :8000)

# API (third terminal)
cd backend
uvicorn nba_props.api:app --reload --port 8000
```

With no `ODDS_API_KEY` the UI opens in demo mode automatically.

## Real data

```bash
cp .env.example .env        # add ODDS_API_KEY from https://the-odds-api.com
cd backend
python -m nba_props.cli -v update     # pull game logs (first run takes a few minutes)
python -m nba_props.cli -v train      # train all 7 market models
python -m nba_props.cli -v picks      # today's top 10 -> data/picks/<date>.json
python -m nba_props.cli backtest      # hold-out MAE vs baselines, hit rate, Brier
```

Then run the API + UI and hit **Refresh lines**. The `make` targets wrap all of these (`make demo`, `make train`, `make picks`, ...).

Daily routine: `update` → `picks` (retrain weekly with `train`).

### Notes and limits

- **stats.nba.com blocks most cloud IPs** (GitHub Actions, AWS, etc.). Run `update` from a home connection. CI only runs the offline tests.
- **Odds API credits**: each game costs (markets × regions) credits, so ~56/day with all 7 markets. The free 500/month tier covers about a week of daily runs. Odds are cached per day; set `MARKETS` or `ODDS_BOOKMAKERS` to cut usage. `picks` only re-pulls with `--refresh-odds`.
- **Backtest lines are a proxy** (player's L10 average rounded to x.5) because historical prop lines aren't free. Start logging the daily `data/picks/*.json` files and you'll build a real closing-line history to grade against.
- Rookies and players with fewer than 5 games are skipped until they have history.
- For research and entertainment. No model guarantees profit.

## Project layout

```
backend/
  nba_props/
    data/nba_stats.py   game logs, positions, rosters (nba_api)
    data/injuries.py    ESPN injury report
    data/odds.py        The Odds API props + no-vig / best-price summary
    data/names.py       name normalization / fuzzy matching
    teams.py            arena coords, altitude, time zones
    features.py         all feature engineering
    model.py            LightGBM per market + probability calibration
    picks.py            EV ranking, filters, explanations
    pipeline.py         update / train / picks / backtest / demo jobs
    api.py              FastAPI (/api/picks, /api/refresh, /api/meta)
    cli.py              command line
    synthetic.py        fake league for tests + demo
  tests/
frontend/               Vite + React + TypeScript UI
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/picks?date=YYYY-MM-DD&limit=10&markets=PTS,REB` | cached picks (computes on first call) |
| GET | `/api/picks?demo=true` | synthetic demo slate |
| POST | `/api/refresh?date=` | re-pull odds + injuries and recompute |
| GET | `/api/meta` | model validation metrics and settings |
| GET | `/api/health` | status |

## Ideas for next steps

- Separate minutes model feeding the stat models (minutes is most of the variance)
- Play-by-play matchup data (primary defender), referee crew pace tendencies
- Log closing lines daily and recalibrate `MODEL_WEIGHT` against real results
- Correlated parlays from the same game
