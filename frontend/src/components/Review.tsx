import { useMemo, useState } from "react";
import { getReview, type ReviewResponse, type ReviewRow } from "../api";

function yesterday() {
  const d = new Date(Date.now() - 86400000);
  const off = d.getTimezoneOffset() * 60000;
  return new Date(d.getTime() - off).toISOString().slice(0, 10);
}

const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(0)}%`);

type Key = "player" | "market" | "projection" | "actual" | "line" | "prob" | "miss";

export default function Review() {
  const [date, setDate] = useState(yesterday());
  const [data, setData] = useState<ReviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [market, setMarket] = useState("ALL");
  const [onlyConfident, setOnlyConfident] = useState(false);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<Key>("prob");
  const [asc, setAsc] = useState(false);

  const run = async (refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      setData(await getReview(date, refresh));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const rows = useMemo(() => {
    if (!data) return [];
    const val = (r: ReviewRow, k: Key) => (k === "miss" ? Math.abs(r.projection - r.actual) : r[k]);
    return data.rows
      .filter((r) => market === "ALL" || r.market === market)
      .filter((r) => !onlyConfident || r.prob >= 0.58)
      .filter((r) => r.player.toLowerCase().includes(q.toLowerCase()) || r.team.includes(q.toUpperCase()))
      .sort((a, b) => {
        const x = val(a, sort);
        const y = val(b, sort);
        const c = x < y ? -1 : x > y ? 1 : 0;
        return asc ? c : -c;
      });
  }, [data, market, onlyConfident, q, sort, asc]);

  const wl = rows.reduce((acc, r) => acc + (r.result === "W" ? 1 : 0), 0);
  const click = (k: Key) => (k === sort ? setAsc(!asc) : (setSort(k), setAsc(false)));
  const th = (k: Key, label: string) => (
    <th onClick={() => click(k)}>
      {label}
      {sort === k ? (asc ? " ▲" : " ▼") : ""}
    </th>
  );

  return (
    <section className="review">
      <div className="review-bar">
        <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
        <button onClick={() => run(false)} disabled={loading}>
          {loading ? "Running…" : "Run review"}
        </button>
        {data && (
          <button className="ghost" onClick={() => run(true)} disabled={loading}>
            Re-run
          </button>
        )}
      </div>
      <p className="hint">
        Retrains the model on games before this date only, projects every player who played, and compares to what
        actually happened. Lines are a stand-in (each player's last-10 average rounded to .5) because past sportsbook
        lines aren't free. First run for a date takes a few minutes.
      </p>

      {error && <div className="error">{error}</div>}

      {data && (
        <>
          {data.graded_picks && (
            <div className="graded">
              <h3>
                Your saved picks for {data.date}: {data.graded_picks.record.W}-{data.graded_picks.record.L}
                {data.graded_picks.record.P ? `-${data.graded_picks.record.P}` : ""}{" "}
                <span className={data.graded_picks.units >= 0 ? "pos" : "neg"}>
                  ({data.graded_picks.units >= 0 ? "+" : ""}
                  {data.graded_picks.units}u)
                </span>
              </h3>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Bet</th>
                      <th>Proj</th>
                      <th>Actual</th>
                      <th>Price</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.graded_picks.picks.map((p, i) => (
                      <tr key={i}>
                        <td>{p.player}</td>
                        <td>
                          {p.side} {p.line} {p.market}
                        </td>
                        <td>{p.projection}</td>
                        <td>{p.actual ?? "–"}</td>
                        <td>
                          {p.price} <small>{p.book}</small>
                        </td>
                        <td className={`res ${p.result}`}>{p.result}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="summary-grid">
            {data.summary.map((s) => (
              <button
                key={s.market}
                className={`sum-card ${market === s.market ? "on" : ""}`}
                onClick={() => setMarket(market === s.market ? "ALL" : s.market)}
              >
                <div className="sum-mkt">{s.market}</div>
                <div className="sum-main">{pct(s.hit_rate_confident)}</div>
                <div className="sum-sub">
                  confident calls ({s.n_confident}) · all {pct(s.hit_rate)}
                </div>
                <div className="sum-sub">
                  avg miss {s.mae_model} vs L10 {s.mae_l10}
                </div>
              </button>
            ))}
          </div>

          <div className="review-filters">
            <input className="search" placeholder="Filter by player or team" value={q} onChange={(e) => setQ(e.target.value)} />
            <label className="check">
              <input type="checkbox" checked={onlyConfident} onChange={(e) => setOnlyConfident(e.target.checked)} />
              Confident calls only (58%+)
            </label>
            <span className="hint">
              {rows.length} rows · {rows.length ? pct(wl / rows.length) : "–"} correct
            </span>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {th("player", "Player")}
                  {th("market", "Mkt")}
                  {th("line", "Line*")}
                  <th>Call</th>
                  {th("prob", "Prob")}
                  {th("projection", "Proj")}
                  {th("actual", "Actual")}
                  {th("miss", "Miss")}
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 400).map((r) => (
                  <tr key={`${r.player_id}-${r.market}`}>
                    <td>
                      {r.player}{" "}
                      <small>
                        {r.team} {r.home ? "vs" : "@"} {r.opponent} · {r.minutes} min
                      </small>
                    </td>
                    <td>{r.market}</td>
                    <td>{r.line}</td>
                    <td className={`side-cell ${r.side.toLowerCase()}`}>{r.side}</td>
                    <td>{pct(r.prob)}</td>
                    <td>{r.projection}</td>
                    <td>{r.actual}</td>
                    <td>{Math.abs(r.projection - r.actual).toFixed(1)}</td>
                    <td className={`res ${r.result}`}>{r.result}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">* Stand-in line: player's last-10 average rounded to .5.</p>
        </>
      )}
    </section>
  );
}
