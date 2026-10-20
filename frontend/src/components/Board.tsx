import { useMemo, useState } from "react";
import type { Pick } from "../api";

type Key = "player" | "market" | "line" | "projection" | "blended_prob" | "ev" | "hit_rate_l10";

const COLS: { key: Key; label: string }[] = [
  { key: "player", label: "Player" },
  { key: "market", label: "Mkt" },
  { key: "line", label: "Line" },
  { key: "projection", label: "Proj" },
  { key: "blended_prob", label: "Prob" },
  { key: "ev", label: "EV" },
  { key: "hit_rate_l10", label: "Hit L10" },
];

export default function Board({ rows }: { rows: Pick[] }) {
  const [sort, setSort] = useState<Key>("ev");
  const [asc, setAsc] = useState(false);
  const [q, setQ] = useState("");

  const sorted = useMemo(() => {
    const f = rows.filter((r) => r.player.toLowerCase().includes(q.toLowerCase()) || r.team.includes(q.toUpperCase()));
    return [...f].sort((a, b) => {
      const x = a[sort] ?? -Infinity;
      const y = b[sort] ?? -Infinity;
      const c = x < y ? -1 : x > y ? 1 : 0;
      return asc ? c : -c;
    });
  }, [rows, sort, asc, q]);

  const click = (k: Key) => {
    if (k === sort) setAsc(!asc);
    else {
      setSort(k);
      setAsc(false);
    }
  };

  return (
    <section className="board">
      <input className="search" placeholder="Filter by player or team" value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLS.map((c) => (
                <th key={c.key} onClick={() => click(c.key)}>
                  {c.label}
                  {sort === c.key ? (asc ? " ▲" : " ▼") : ""}
                </th>
              ))}
              <th>Side</th>
              <th>Price</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={`${r.player_id}-${r.market}`}>
                <td>
                  {r.player} <small>{r.team} {r.home ? "vs" : "@"} {r.opponent}</small>
                </td>
                <td>{r.market}</td>
                <td>{r.line}</td>
                <td>{r.projection.toFixed(1)}</td>
                <td>{(r.blended_prob * 100).toFixed(0)}%</td>
                <td className={r.ev >= 0 ? "pos" : "neg"}>{(r.ev * 100).toFixed(1)}%</td>
                <td>{r.hit_rate_l10 == null ? "–" : `${(r.hit_rate_l10 * 100).toFixed(0)}%`}</td>
                <td className={`side-cell ${r.side.toLowerCase()}`}>{r.side}</td>
                <td>
                  {r.price} <small>{r.book}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
