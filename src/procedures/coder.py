# src/procedures/coder.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import re, difflib

from src.procedures.vector_store import ensure_ingested, multi_query_note

@dataclass
class PcsCandidate:
    code: str
    title: str
    confidence: float
    rationale: List[str]

# --------- generic token helpers ----------
_STOP = {
    "a","an","and","the","of","in","on","with","without","w","for","to","by",
    "at","from","or","via","into","than","as","per","using","use","none"
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _norm_tokens(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if len(t) > 1 and t not in _STOP]

def _f1_overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    if inter == 0: return 0.0
    prec = inter / len(b)
    rec  = inter / len(a)
    return 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)

def _split_core_modifiers(title: str) -> Tuple[Set[str], Set[str]]:
    t = title.lower()
    parts = t.split(" with ", 1)
    core = _norm_tokens(parts[0])
    mods = _norm_tokens(parts[1]) if len(parts) > 1 else []
    return set(core), set(mods)

def _score_to_conf(score: float) -> float:
    score = max(0.0, min(1.0, score))
    return round(0.5 + 0.48 * score, 3)

# ----- anchor (generic) to avoid ultrasonic≠ultrasonography confusions -----
def _anchors(tokens: List[str], k: int = 2, min_len: int = 7) -> List[str]:
    long = [t for t in tokens if len(t) >= min_len]
    long.sort(key=len, reverse=True)
    seen, out = set(), []
    for t in long:
        if t not in seen:
            out.append(t); seen.add(t)
        if len(out) >= k: break
    return out

def _has_anchor(title_tokens: List[str], anchor: str, thresh: float = 0.85) -> bool:
    for tt in title_tokens:
        if tt == anchor: return True
        if difflib.SequenceMatcher(None, tt, anchor).ratio() >= thresh:
            return True
    return False

def _anchor_factor(note: str, title: str) -> float:
    note_toks = _norm_tokens(note)
    anchors = _anchors(note_toks, k=2, min_len=7)
    if not anchors: return 1.0
    t_toks = _norm_tokens(title)
    miss = [a for a in anchors if not _has_anchor(t_toks, a)]
    if not miss: return 1.0
    return 0.6 if len(miss) == 1 else 0.45

def _make_context(hits: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, str]]:
    """Small context payload for supervisor LLM (RAG)."""
    ctx = []
    for h in hits[:k]:
        ctx.append({
            "code": h.get("code", ""),
            "title": h.get("title", ""),
            "table": str(h.get("table", "")),
            "section": str(h.get("section", "")),
        })
    return ctx

# --------- main ----------
def code_note(note: str, top_k: int = 8, min_conf: float = 0.6) -> Dict[str, Any]:
    total = ensure_ingested()
    # RAG: multiple small queries merged by code
    hits = multi_query_note(note, top_k).get("results", [])

    if not hits:
        return {
            "procedures": [{
                "kind": "rag_retriever+lexical",
                "hint": "no semantically similar PCS titles found",
                "evidence_span": None,
                "candidates": [],
                "needs_clarification": "No candidates returned by the vector store.",
            }],
            "notes": [f"Vector store contained {total} PCS titles (2026).", "No results returned for this note."],
            "context": [],
        }

    note_core, _ = _split_core_modifiers(note)
    note_toks = set(_norm_tokens(note))

    scored = []
    for h in hits:
        title = h.get("title") or ""
        sim = float(h.get("score") or 0.0)          # merged max score
        t_core, t_mods = _split_core_modifiers(title)

        f1 = _f1_overlap(note_toks, set(_norm_tokens(title)))
        unmatched = t_mods - note_toks
        unmatched_frac = (len(unmatched) / max(1, len(t_mods))) if t_mods else 0.0
        len_pen = min(1.0, len(title) / 120.0)
        anchor = _anchor_factor(note, title)        # multiplicative

        base = 0.60*sim + 0.30*f1 - 0.08*unmatched_frac - 0.02*len_pen
        base = max(0.0, min(1.0, base))
        final = base * anchor

        scored.append((
            final, h, {
                "sim": round(sim, 4),
                "f1": round(f1, 4),
                "unmatched_mod_frac": round(unmatched_frac, 4),
                "len_penalty": round(len_pen, 4),
                "anchor_factor": round(anchor, 3),
                "hits": h.get("_hits", 1),
            }
        ))

    scored.sort(key=lambda x: x[0], reverse=True)

    cands: List[PcsCandidate] = []
    for final, h, parts in scored:
        conf = _score_to_conf(final)
        if conf < min_conf:
            continue
        cands.append(PcsCandidate(
            code=h.get("code"),
            title=h.get("title"),
            confidence=conf,
            rationale=[
                "rag_retriever+lexical",
                f"score_sim={parts['sim']}",
                f"f1={parts['f1']}",
                f"unmatched_mod_frac={parts['unmatched_mod_frac']}",
                f"len_penalty={parts['len_penalty']}",
                f"anchor_factor={parts['anchor_factor']}",
                f"rewrites_hits={parts['hits']}",
                f"table={h.get('table')}",
                f"section={h.get('section')}",
            ],
        ))
        if len(cands) >= top_k:
            break

    return {
        "procedures": [{
            "kind": "rag_retriever+lexical",
            "hint": "Merged multi-query retrieval + generic lexical re-rank (no domain rules).",
            "evidence_span": None,
            "candidates": [asdict(c) for c in cands],
            "needs_clarification": None,
        }],
        "notes": [
            f"Vector store contained {total} PCS titles (2026).",
            "Retrieval = small set of query rewrites merged by code.",
            "Ranking = 60% dense sim + 30% unigram F1 − 8% unmatched modifiers − 2% length; multiplied by anchor factor.",
        ],
        # RAG context for the supervisor LLM to craft a final sentence if desired
        "context": _make_context(hits),
    }
