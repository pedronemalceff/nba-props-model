import { useCallback, useEffect, useState } from "react";
import { getHealth, getPicks, refreshPicks, type PicksResponse } from "./api";
import PickCard from "./components/PickCard";
import Board from "./components/Board";
import Review from "./components/Review";

const MARKETS = ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA"];

function todayLocal() {
  const d = new Date();
  const off = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - off).toISOString().slice(0, 10);
}

export default function App() {
  const [date, setDate] = useState(todayLocal());
  const [limit, setLimit] = useState(10);
  const [markets, setMarkets] = useState<string[]>([]);
  const [demo, setDemo] = useState(false);
  const [data, setData] = useState<PicksResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"picks" | "board" | "review">("picks");
  const [ready, setReady] = useState(false);

  // Default to demo mode when the backend has no Odds API key configured.
  useEffect(() => {
    getHealth()
      .then((h) => {
        if (!h.odds_api_key) setDemo(true);
        setReady(true);
      })
      .catch(() => setError("Can't reach the model. Double-click start.bat in the project folder to start it."));
  }, []);

  const load = useCallback(async () => {
    if (!ready) return;
    setLoading(true);
    setError(null);
    try {
      setData(await getPicks({ date: demo ? undefined : date, limit, markets, demo }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [date, limit, markets, demo, ready]);

  useEffect(() => {
    load();
  }, [load]);

  const onRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      await refreshPicks(date);
      await load();
    } catch (e) {
      setError((e as Error).message);
      setLoading(false);
    }
  };

  const toggleMarket = (m: string) =>
    setMarkets((cur) => (cur.includes(m) ? cur.filter((x) => x !== m) : [...cur, m]));

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>NBA Props Model</h1>
          <p className="sub">
            Best player props of the day, ranked by expected value
            {data && (
              <>
                {" "}· {data.date}
                {data.demo && <span className="badge demo">Demo data</span>}
              </>
            )}
          </p>
        </div>
        <div className="controls">
          <input type="date" value={date} onChange={(e) => setDate(e.target.value)} disabled={demo} />
          <label className="slider">
            Top {limit}
            <input type="range" min={5} max={10} value={limit} onChange={(e) => setLimit(+e.target.value)} />
          </label>
          <label className="check">
            <input type="checkbox" checked={demo} onChange={(e) => setDemo(e.target.checked)} /> Demo
          </label>
          <button onClick={onRefresh} disabled={loading || demo}>
            {loading ? "Working…" : "Refresh lines"}
          </button>
        </div>
      </header>

      <div className="chips">
        {MARKETS.map((m) => (
          <button key={m} className={`chip ${markets.includes(m) ? "on" : ""}`} onClick={() => toggleMarket(m)}>
            {m}
          </button>
        ))}
        {markets.length > 0 && (
          <button className="chip clear" onClick={() => setMarkets([])}>
            All markets
          </button>
        )}
      </div>

      <nav className="tabs">
        <button className={tab === "picks" ? "on" : ""} onClick={() => setTab("picks")}>
          Top picks
        </button>
        <button className={tab === "board" ? "on" : ""} onClick={() => setTab("board")}>
          Full board {data ? `(${data.board.length})` : ""}
        </button>
        <button className={tab === "review" ? "on" : ""} onClick={() => setTab("review")}>
          Results
        </button>
      </nav>

      {tab !== "review" && error && <div className="error">{error}</div>}
      {tab !== "review" && data?.note && <div className="note">{data.note}</div>}
      {tab !== "review" && loading && !data && <div className="note">Building projections… first run can take a minute.</div>}

      {data && tab === "picks" && (
        <section className="grid">
          {data.picks.length === 0 && <div className="note">No props cleared the edge threshold today.</div>}
          {data.picks.map((p, i) => (
            <PickCard key={`${p.player_id}-${p.market}`} pick={p} rank={i + 1} />
          ))}
        </section>
      )}
      {data && tab === "board" && <Board rows={data.board} />}
      {tab === "review" && <Review />}

      <footer>
        Projections are estimates, not guarantees. For research and entertainment; bet responsibly.
      </footer>
    </div>
  );
}
