"""Hybrid recall and the session brief.

recall: three ranked lists over the active memories of the scope (+ global), fused by
reciprocal-rank fusion (k=60), multiplied by the decayed memory score:

  bm25    BM25(query, name + description + body). A memory must share >= 1 distinct query
          term (15 % of them for long queries, capped at 4). With vectors available, a
          keyword hit whose cosine is below BM25_VEC_FLOOR is dropped as noise.
  vector  cosine(query, memory) >= VEC_MIN (only with the [vector] extra)
  graph   memories naming an entity that appears in the query (1.0 each) or a non-hub
          1-hop neighbor of one (0.5); hubs (named in > 3 memories) are not expanded

Hits below REL_CUTOFF × the best are dropped. Recall is read-only: it never changes scores
(reinforcing whatever was returned would make frequently-returned memories win forever);
use `reinforce` when a memory actually helped. Thresholds were calibrated on real memories
with BAAI/bge-small-en-v1.5: unrelated queries score 0.40-0.51 cosine, the right memory
0.57-0.80. They are looser than an automatic per-prompt injector would use, because here the
model asks on purpose and judges the hits.
"""

from __future__ import annotations

import math
import re

from .store import KINDS, Store, decayed, scope_dir_name, scope_key

MIN_TERMS = 2
MIN_MATCHED = 1        # recall is an explicit tool call: the model judges the hits (the
                       # 2-term gate of automatic per-prompt injection loses obvious matches)
MATCH_FRACTION = 0.15
MAX_MATCHED = 4
VEC_MIN = 0.55         # explicit recall; unrelated pairs measured 0.40-0.51 with bge-small
BM25_VEC_FLOOR = 0.50
REL_CUTOFF = 0.5
RRF_K = 60
K1, B = 1.5, 0.75
STOPWORDS = set("""
the and for are but not you your yours with this that these those from have has had was were
will would can could should what when where which who whom why how all any each some into onto
about after before again also just like make made more most much only other our out over same
such than then them they their there here its it's let lets please now new use used using via
want need get got does did done doing yes okay sure thanks thank check look see show tell give
""".split())


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def terms(text: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in STOPWORDS]


def _scopes(scope: str | None) -> list[str]:
    sd = scope_dir_name(scope_key(scope))
    return ["global"] if sd == "global" else [sd, "global"]


def _bm25(query: list[str], docs: list[tuple[dict, list[str]]]) -> list[tuple[float, dict]]:
    if not docs:
        return []
    n = len(docs)
    avgdl = sum(len(d) for _, d in docs) / n or 1.0
    df: dict[str, int] = {}
    for _, d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    need = min(MAX_MATCHED, max(MIN_MATCHED, round(MATCH_FRACTION * len(query))))
    out = []
    for row, d in docs:
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        matched = [t for t in query if t in tf]
        if len(matched) < need:
            continue
        s = 0.0
        for t in matched:
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            f = tf[t]
            s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * len(d) / avgdl))
        out.append((s, row))
    return sorted(out, key=lambda x: -x[0])


def _hit(row, score: float, signals: list[str], body_chars: int) -> dict:
    body = row["body"] or ""
    return {"id": row["id"], "kind": row["kind"], "scope": row["scope"],
            "status": row["status"], "description": row["description"],
            "content": body if len(body) <= body_chars else body[:body_chars] + "…",
            "memory_score": round(decayed(row["kind"], row["confidence"],
                                          row["last_reinforced"]), 3),
            "relevance": round(score, 5), "signals": signals}


def recall(store: Store, query: str, scope: str | None = None, k: int = 5,
           kinds: list[str] | None = None, include_provisional: bool = False,
           body_chars: int = 600) -> list[dict]:
    query = query or ""
    q = list(dict.fromkeys(terms(query)))
    if len(q) < MIN_TERMS:
        return []
    store.sync()
    kinds = [x for x in (kinds or [x for x in KINDS if x != "trace"]) if x in KINDS]
    scopes = _scopes(scope)
    statuses = ("active", "provisional") if include_provisional else ("active",)
    rows = store.rows(scopes, kinds, statuses)
    if not rows:
        return []
    by_id = {r["id"]: r for r in rows}
    docs = [(r, terms(f"{r['name'].replace('-', ' ')} {r['description']} {r['body']}"))
            for r in rows]
    cos = store.vector_scores(query, list(by_id))
    graph = store.graph_scores(query, scopes)
    lists = (
        ("bm25", [r["id"] for _, r in _bm25(q, docs)
                  if not cos or cos.get(r["id"], 1.0) >= BM25_VEC_FLOOR]),
        ("vector", [i for i, c in sorted(cos.items(), key=lambda x: -x[1]) if c >= VEC_MIN]),
        ("graph", [i for i, _ in sorted(((i, v[0]) for i, v in graph.items() if i in by_id),
                                        key=lambda x: -x[1])]),
    )
    fused: dict[str, float] = {}
    signals: dict[str, list[str]] = {}
    for name, ids in lists:
        for rank, i in enumerate(ids):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            via = graph.get(i, (0, []))[1] if name == "graph" else []
            signals.setdefault(i, []).append(f"graph:{','.join(via[:2])}" if via else name)
    ranked = sorted(((s * decayed(by_id[i]["kind"], by_id[i]["confidence"],
                                  by_id[i]["last_reinforced"]), i) for i, s in fused.items()),
                    reverse=True)
    if ranked:
        ranked = [x for x in ranked if x[0] >= REL_CUTOFF * ranked[0][0]]
    return [_hit(by_id[i], s, signals[i], body_chars) for s, i in ranked[:max(1, min(k, 20))]]


BRIEF_LIMITS = {"preference": 8, "procedural": 6, "fact": 5, "episodic": 3}


def brief(store: Store, scope: str | None = None) -> dict:
    """What to know before starting: preferences and procedures always (they apply whatever
    the task), plus the strongest facts and the most recent episodes. Grouped, best first."""
    store.sync()
    scopes = _scopes(scope)
    out: dict[str, list[dict]] = {}
    for kind, limit in BRIEF_LIMITS.items():
        rows = store.rows(scopes, [kind])
        key = (lambda r: r["last_reinforced"] or "") if kind == "episodic" else \
            (lambda r: decayed(r["kind"], r["confidence"], r["last_reinforced"]))
        top = sorted(rows, key=key, reverse=True)[:limit]
        if top:
            out[kind] = [_hit(r, 0.0, [], 300) for r in top]
            for h in out[kind]:
                h.pop("relevance")
                h.pop("signals")
    provisional = len(store.rows(scopes, None, ("provisional",)))
    return {"scope": scope_key(scope), "memories": out, "provisional_pending": provisional}
