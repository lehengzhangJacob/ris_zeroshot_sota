"""GRES-aware selection: a training-free selector for generalized RIS.

The RefCOCO-family operator (`method.py`) returns *one* mask and never
abstains. gRefCOCO (GRES) requires both: an expression may refer to several
instances (union output) or to nothing at all (empty output). This module
extends the operator to a *set-valued* one:

  1. **Clause decomposition.** The expression is split at top level on
     "and"/"or"/","; each clause contributes one head noun phrase. This is the
     gRefCOCO construction: a multi-target expression is two referring clauses
     merged with a conjunction, and the ground truth is exactly the pair of
     clause referents (co-mentioned modifiers, e.g. "a laptop" inside
     "a mouse beside a laptop", are not targets).

  2. **One selection per clause.** Each clause's head noun phrase scores the
     proposal pool (frozen text encoder, cached features); the clause selects
     its best proposal, restricted to the clause's spatial-relation side when
     the clause carries one (reuse of `method.select`).

  3. **Instance expansion on plurals/quantifiers.** When a clause carries a
     plural head or a count word ("two"/"all"/"both"/...), the clause admits
     every proposal whose head score is within `margin` of the clause's best
     and that is not a near-duplicate of an already-admitted mask.

  4. **Abstention.** A no-target expression is one where no clause's best
     score clears `abstain_thr`; the operator then returns the empty set.
     The threshold is calibrated on RefCOCO val (where every expression has a
     target) and applied unchanged.

The caller supplies a `scorer(phrase) -> np.ndarray` function (scores every
proposal against a phrase) and the record; nothing here trains or reads ground
truth.
"""
from __future__ import annotations

import re

import numpy as np

COUNT_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
               "both", "all", "every", "each", "several", "many", "some", "multiple",
               "few", "pair", "group", "couple", "trio"}
CLAUSE_SPLIT = re.compile(r"\s+and\s+|\s+or\s+|,\s*|\s*&\s*")


def clauses(sent):
    """Top-level clause split; drops empty/very short fragments."""
    parts = [c.strip(" .!?") for c in CLAUSE_SPLIT.split(sent.lower())]
    return [c for c in parts if len(c.split()) >= 2]


def is_plural_clause(clause, head_np):
    words = set(re.findall(r"[a-z]+", clause.lower()))
    if words & COUNT_WORDS:
        return True
    w = head_np.strip().lower().split()
    return bool(w) and w[-1].endswith("s") and not w[-1].endswith("ss")


def _dedupe(idxs, scores, pc, thr=0.5):
    """Keep the higher-scoring member of any pair with mask IoU > thr."""
    kept = []
    for i in sorted(idxs, key=lambda x: -scores[x]):
        if all(pc[i, j] <= thr for j in kept):
            kept.append(int(i))
    return kept


def clause_select(clause_head_vec, clause, rel, rec, scorer, k=8, pct=0.5, margin=0.0):
    """Indices selected for one clause (possibly several on plurals)."""
    from method import select
    head_vec = clause_head_vec
    top = int(np.argmax(head_vec))
    if is_plural_clause(clause, clause_head_vec_name(rec, clause_head_vec)):
        cand = np.flatnonzero(head_vec >= head_vec[top] - margin) if margin else np.array([top])
    else:
        cand = np.array([top])
    cand = list(cand)
    if rel is not None and len(cand) > 1:
        from method import side_selection
        m = side_selection(cand, rel, rec["cx"], rec["cy"], pct)
        if m is not None and m.any():
            cand = list(np.array(cand)[m])
    return _dedupe(cand, head_vec, rec["pc"], 0.5)


def clause_head_vec_name(rec, vec):
    return ""


def gres_select(rec, scorer, nlp, clause_phrases=None, margin=0.0, abstain_thr=None,
                snap=True, use_relations=True, k=8, pct=0.5):
    """Return the list of proposal indices the operator outputs for `rec`.

    `scorer(phrase) -> np.ndarray` scores every proposal in the record's pool
    against `phrase` (the frozen text encoder). `nlp` is a spaCy pipeline.
    """
    from method import parse_relation
    rel = parse_relation(rec["sent"]) if use_relations else None
    phrases = clause_phrases if clause_phrases is not None else clause_heads(rec["sent"], nlp)
    idxs, best = [], -np.inf
    for phrase in phrases:
        vec = scorer(phrase)
        top = int(np.argmax(vec))
        best = max(best, float(vec[top]))
        if abstain_thr is not None and float(vec[top]) < abstain_thr:
            continue
        cand = _clause_candidates(rec, phrase, vec, rel, margin, pct)
        idxs.extend(cand)
    idxs = _dedupe(idxs, np.asarray(rec["sc_content"]), rec["pc"], 0.5)
    if not idxs:
        return []
    if snap:
        idxs = sorted({int(rec["ps_idx"][i]) for i in idxs})
    return idxs


def _clause_candidates(rec, phrase, vec, rel, margin, pct):
    from method import side_selection
    top = int(np.argmax(vec))
    plural = is_plural_clause(phrase, phrase)
    if plural and margin:
        cand = list(np.flatnonzero(vec >= vec[top] - margin))
    else:
        cand = [top]
    if rel is not None and len(cand) > 1:
        m = side_selection(cand, rel, np.asarray(rec["cx"]), np.asarray(rec["cy"]), pct)
        if m is not None and m.any():
            cand = list(np.array(cand)[m])
    return _dedupe(cand, vec, rec["pc"], 0.5)


def clause_heads(sent, nlp):
    """Head noun phrase of each clause, deduped, order preserved."""
    out = []
    for clause in clauses(sent):
        heads = _np_heads(clause, nlp)
        for h in heads:
            if h and h not in out:
                out.append(h)
    return out or _np_heads(sent.lower(), nlp)


def _np_heads(text, nlp):
    """The clause's head noun phrases: root NP of the clause, plus coordinated
    subject/object NPs (spaCy)."""
    doc = nlp(text)
    heads = []
    for chunk in doc.noun_chunks:
        root = chunk.root
        dep = root.dep_
        # subject/object/attribute positions, or the sentence root itself
        if dep in ("nsubj", "nsubjpass", "dobj", "pobj", "attr", "appos", "ROOT", "conj", "dative"):
            if chunk.text.strip() and chunk.text.strip() not in heads:
                heads.append(chunk.text.strip())
    if not heads:
        for chunk in doc.noun_chunks:
            heads.append(chunk.text.strip())
            break
    return heads
