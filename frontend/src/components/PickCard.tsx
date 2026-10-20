import { useState } from "react";
import type { Pick } from "../api";

const pct = (v: number | null | undefined, d = 0) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
const num = (v: number | null | undefined, d = 1) => (v == null ? "–" : v.toFixed(d));

export default function PickCard({ pick: p, rank }: { pick: Pick; rank: number }) {
  const [imgOk, setImgOk] = useState(true);
  const diff = p.projection - p.line;
  const maxImpact = Math.max(...p.factors.map((f) => Math.abs(f.impact)), 0.01);

  return (
    <article className="card">
      <div className="card-head">
        <span className="rank">{rank}</span>
        {imgOk ? (
          <img
            className="headshot"
            src={`https://cdn.nba.com/headshots/nba/latest/260x190/${p.player_id}.png`}
            alt=""
            onError={() => setImgOk(false)}
          />
        ) : (
          <div className="headshot placeholder">{p.player.split(" ").map((s) => s[0]).join("")}</div>
        )}
        <div className="who">
          <div className="name">
            {p.player}
            {p.questionable && <span className="badge warn">{p.status}</span>}
            {p.b2b && <span className="badge">B2B</span>}
          </div>
          <div className="matchup">
            {p.team} {p.home ? "vs" : "@"} {p.opponent}
          </div>
        </div>
      </div>

      <div className="bet">
        <span className={`side ${p.side.toLowerCase()}`}>{p.side}</span>
        <span className="line">{p.line}</span>
        <span className="mkt">{p.market}</span>
        <span className="price">
          {p.price} <small>{p.book}</small>
        </span>
      </div>

      <div className="kpis">
        <div>
          <label>Projection</label>
          <b>{num(p.projection)}</b>
          <small className={diff >= 0 ? "pos" : "neg"}>
            {diff >= 0 ? "+" : ""}
            {diff.toFixed(1)}
          </small>
        </div>
        <div>
          <label>Win prob</label>
          <b>{pct(p.blended_prob)}</b>
          <small>mkt {pct(p.market_prob)}</small>
        </div>
        <div>
          <label>EV / $1</label>
          <b className="pos">{p.ev >= 0 ? "+" : ""}{(p.ev * 100).toFixed(1)}%</b>
          <small>edge {pct(p.edge, 1)}</small>
        </div>
      </div>

      <div className="probbar" title="Model probability vs market no-vig probability">
        <div className="fill" style={{ width: pct(p.model_prob, 1) }} />
        <div className="mark" style={{ left: pct(p.market_prob, 1) }} />
      </div>

      <table className="mini">
        <tbody>
          <tr>
            <td>L10 avg</td>
            <td>{num(p.l10_avg)}</td>
            <td>Season</td>
            <td>{num(p.season_avg)}</td>
          </tr>
          <tr>
            <td>Vs {p.opponent}</td>
            <td>{num(p.vs_opp_avg)}</td>
            <td>Hit L10</td>
            <td>{pct(p.hit_rate_l10)}</td>
          </tr>
          <tr>
            <td>Min L10</td>
            <td>{num(p.minutes_l10)}</td>
            <td>Rest</td>
            <td>{p.rest_days == null ? "–" : `${p.rest_days}d`}</td>
          </tr>
        </tbody>
      </table>

      <div className="factors">
        <label>Key drivers</label>
        {p.factors.map((f) => (
          <div key={f.feature} className="factor">
            <span className="flabel" title={f.feature}>
              {f.label}
              {f.value != null && <small> {f.value}</small>}
            </span>
            <span className="fbar">
              <span
                className={f.impact >= 0 ? "up" : "down"}
                style={{ width: `${(Math.abs(f.impact) / maxImpact) * 100}%` }}
              />
            </span>
            <span className={`fval ${f.impact >= 0 ? "pos" : "neg"}`}>
              {f.impact >= 0 ? "+" : ""}
              {f.impact.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </article>
  );
}
