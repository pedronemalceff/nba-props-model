"""Player-name normalization so sportsbook / ESPN names match nba_api names."""
from __future__ import annotations

import difflib
import re
import unicodedata

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def match_name(name: str, candidates: dict[str, int], cutoff: float = 0.85) -> int | None:
    """candidates: normalized name -> PLAYER_ID. Exact match first, then fuzzy."""
    key = normalize_name(name)
    if key in candidates:
        return candidates[key]
    hit = difflib.get_close_matches(key, list(candidates), n=1, cutoff=cutoff)
    return candidates[hit[0]] if hit else None
