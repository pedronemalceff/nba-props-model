export type Factor = { feature: string; label: string; value: number | null; impact: number };

export type Pick = {
  player_id: number;
  player: string;
  team: string;
  opponent: string;
  home: boolean;
  event_id: string;
  commence: string;
  market: string;
  side: "OVER" | "UNDER";
  line: number;
  projection: number;
  model_prob: number;
  market_prob: number;
  blended_prob: number;
  edge: number;
  ev: number;
  price: string;
  book: string;
  n_books: number;
  l10_avg: number | null;
  season_avg: number | null;
  vs_opp_avg: number | null;
  hit_rate_l10: number | null;
  minutes_l10: number | null;
  rest_days: number | null;
  b2b: boolean;
  teammate_min_out: number | null;
  status: string;
  questionable: boolean;
  factors: Factor[];
};

export type PicksResponse = {
  date: string;
  generated_at: string;
  demo: boolean;
  note: string | null;
  picks: Pick[];
  board: Pick[];
};

const BASE = import.meta.env.VITE_API_URL ?? "";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j.detail) msg = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const getHealth = () => req<{ ok: boolean; odds_api_key: boolean }>("/api/health");

export function getPicks(opts: { date?: string; limit: number; markets: string[]; demo: boolean }) {
  const q = new URLSearchParams({ limit: String(opts.limit) });
  if (opts.date) q.set("date", opts.date);
  if (opts.markets.length) q.set("markets", opts.markets.join(","));
  if (opts.demo) q.set("demo", "true");
  return req<PicksResponse>(`/api/picks?${q}`);
}

export function refreshPicks(date?: string) {
  const q = new URLSearchParams();
  if (date) q.set("date", date);
  return req<PicksResponse>(`/api/refresh?${q}`, { method: "POST" });
}

export type ReviewSummary = {
  market: string;
  n: number;
  mae_model: number;
  mae_l10: number;
  hit_rate: number;
  n_confident: number;
  hit_rate_confident: number | null;
};

export type ReviewRow = {
  player_id: number;
  player: string;
  team: string;
  opponent: string;
  home: boolean;
  market: string;
  projection: number;
  actual: number;
  l10_avg: number;
  line: number;
  side: "OVER" | "UNDER";
  prob: number;
  result: "W" | "L";
  minutes: number;
};

export type GradedPick = {
  player: string;
  market: string;
  side: string;
  line: number;
  projection: number;
  price: string;
  book: string;
  actual: number | null;
  result: "W" | "L" | "P" | "DNP";
};

export type ReviewResponse = {
  date: string;
  line_type: string;
  summary: ReviewSummary[];
  rows: ReviewRow[];
  graded_picks: { picks: GradedPick[]; record: Record<string, number>; units: number } | null;
};

export function getReview(date: string, refresh = false) {
  const q = new URLSearchParams({ date });
  if (refresh) q.set("refresh", "true");
  return req<ReviewResponse>(`/api/review?${q}`);
}
