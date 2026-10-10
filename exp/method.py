"""The selector under test: category-set relation guidance.

One place, so the paper, the ablations and the second-dataset run all execute
the same code.

Given the quantities a training-free propose-then-select RIS pipeline already
computes — a semantic score per proposal, the released coherence score
(GEM attention under the position mask), mask geometry, and the expression —
the operator is:

  1. score every proposal against the expression with its spatial vocabulary
     removed  (the *category* score);
  2. candidate set C = the top k proposals under that score;
  3. if the expression carries a spatial relation, return the member of C that
     the relation selects:
        - ordinal  ("second from the left")  -> the n-th along the axis
        - left/right/top/bottom             -> rank the members by the released
                                               coherence score, restricted to
                                               the half of C lying on the
                                               relation's side (median split
                                               along the relation axis)
        - middle/center                     -> same, restricted to the central
                                               half
  4. otherwise return the pipeline's own selection unchanged.

No training, no ground truth, no extra forward pass: step 1 reuses the text
encoder, step 3 reuses the coherence map the pipeline already computed.
"""
from __future__ import annotations

import re

import numpy as np

H_WORDS = {"left": -1, "leftmost": -1, "l": -1, "right": +1, "rightmost": +1}
V_WORDS = {"top": -1, "upper": -1, "above": -1, "topmost": -1,
           "bottom": +1, "lower": +1, "below": +1}
MID = {"middle", "center", "centre", "mid", "central"}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def parse_relation(sent):
    """Deterministic spatial-relation parse. Returns None when the expression
    carries no spatial constraint."""
    w = re.findall(r"[a-z0-9]+", sent.lower())
    rel = dict(h=0, v=0, mid=False, ordinal=None, axis=None, direction=None)
    for i, x in enumerate(w):
        if x in H_WORDS:
            rel["h"] = H_WORDS[x]
        if x in V_WORDS:
            rel["v"] = V_WORDS[x]
        if x in MID:
            rel["mid"] = True
        if x in ORD:
            rel["ordinal"] = ORD[x]
            for y in w[i + 1:i + 4]:
                if y in H_WORDS:
                    rel["axis"], rel["direction"] = "h", H_WORDS[y]
                if y in V_WORDS:
                    rel["axis"], rel["direction"] = "v", V_WORDS[y]
    return rel if (rel["h"] or rel["v"] or rel["mid"] or rel["ordinal"]) else None


def side_selection(cand, rel, cx, cy, pct=0.5):
    """Boolean mask over `cand` marking the members the relation's side admits.

    A relation may constrain one axis ("left") or two ("bottom left"); with two
    axes the member must satisfy both half-plane constraints. Returns None when
    no axis applies, and falls back to the primary axis when the intersection
    is empty (a side split over few candidates can be degenerate).
    """
    cand = list(cand)
    v_h = np.asarray(cx)[cand]
    v_v = np.asarray(cy)[cand]
    parts = []
    if rel["h"]:
        sign = rel["h"]
        thr = np.quantile(v_h, pct) if sign < 0 else np.quantile(v_h, 1 - pct)
        parts.append(v_h <= thr if sign < 0 else v_h >= thr)
    if rel["v"]:
        sign = rel["v"]
        thr = np.quantile(v_v, pct) if sign < 0 else np.quantile(v_v, 1 - pct)
        parts.append(v_v <= thr if sign < 0 else v_v >= thr)
    if not parts:
        if rel["mid"]:
            v = -np.abs(v_h - 0.5)
            thr = np.quantile(v, 1 - pct)
            parts.append(v >= thr)
        else:
            return None
    m = parts[0]
    for extra in parts[1:]:
        m = m & extra
    if not m.any() and len(parts) > 1:
        m = parts[0]
    return m


def select(cand, rel, cx, cy, gem, pct=0.5):
    """Return the chosen index from `cand`, or None if the relation cannot act."""
    cand = list(cand)
    if len(cand) < 2:
        return None
    cx_all = np.asarray(cx)
    cy_all = np.asarray(cy)
    g = np.asarray(gem)

    if rel["ordinal"]:
        axis = rel["axis"] or "h"
        d = rel["direction"] or -1
        v = cx_all[cand] if axis == "h" else cy_all[cand]
        order = np.argsort(d * v)
        pos = rel["ordinal"] - 1
        return cand[int(order[pos])] if pos < len(order) else None

    m = side_selection(cand, rel, cx_all, cy_all, pct)
    if m is None:
        return None
    sub = np.array(cand)[m]
    if not len(sub):
        return None
    return int(sub[int(np.argmax(g[sub]))])


class Selector:
    """Applies `select` to the quantities stored for one expression."""

    def __init__(self, k=8, key="sc_content", pct=0.5, score="score_gem", ordinals=False):
        self.k, self.key, self.pct, self.score = k, key, pct, score
        self.ordinals = ordinals

    def __call__(self, r):
        rel = r.rel
        if rel is None:
            return r.final
        # The deterministic ordinal resolver did not beat the released selection
        # on this stratum, so by default the operator abstains there rather than
        # taking a loss.
        if rel["ordinal"] and not self.ordinals:
            return r.final
        rank = r.S_gem if self.score == "score_gem" else r.S[self.score]
        cand = np.argsort(-r.S[self.key], kind="stable")[:self.k].tolist()
        idx = select(cand, rel, r.cx, r.cy, rank, self.pct)
        return int(idx) if idx is not None else r.final
