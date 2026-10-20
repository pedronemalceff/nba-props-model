.PHONY: install api ui demo update train picks backtest test

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

api:
	cd backend && uvicorn nba_props.api:app --reload --port 8000

ui:
	cd frontend && npm run dev

demo:
	cd backend && python -m nba_props.cli demo

update:
	cd backend && python -m nba_props.cli -v update

train:
	cd backend && python -m nba_props.cli -v train

picks:
	cd backend && python -m nba_props.cli -v picks --limit 10

backtest:
	cd backend && python -m nba_props.cli backtest

test:
	cd backend && python -m pytest -q
