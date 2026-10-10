#!/usr/bin/env python
"""EXP-1e — Offline search over wrong-instance repair rules (true mask IoU).

Reads the EXP-1d record file. Confusion sets use the stored TRUE pairwise mask
IoU, never a bbox proxy. Reports mIoU, oIoU, delta against argmax, and how many
expressions the rule changed, so a rule cannot win by firing everywhere.

Controls included so the gain cannot be attributed to something trivial:
  geomall_*      resolve by geometry over the WHOLE score-ordered pool, with no
                 confusion set -> isolates the contribution of the category set
  geom_swapped   apply the resolver with the mirror-image relation (text says
                 left, resolver takes the rightmost) -> isolates whether the
                 relation semantics matter or any extreme pick helps
  argmax_<prompt> prompt decomposition alone, no geometry
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

H_WORDS = {"left": -1, "leftmost": -1, "l": -1, "right": +1, "rightmost": +1}
V_WORDS = {"top": -1, "upper": -1, "above": -1, "topmost": -1, "bottom": +1, "lower": +1, "below": +1}
MID = {"middle", "center", "centre", "mid", "central"}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def parse(sent):
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


class Pool:
    def __init__(self, rec):
        self.r = rec
        self.n = rec["n"]
        self.cx = np.array(rec["cx"])
        self.cy = np.array(rec["cy"])
        self.area = np.array(rec["area"])
        self.gt = np.array(rec["gt_iou"])
        self.S = {k: np.array(v["fused"]) for k, v in rec["scores"].items()}
        self.rel = parse(rec["sent"])
        self.orc = rec["orc"]
        self.tops = {k: np.array(v) for k, v in rec["tops"].items()}
        self.pairs = {k: np.array(v) for k, v in rec["pairs"].items()}
        self.base = int(np.argmax(self.S["full"]))


def confusion(p, key, k, delta, split_iou=0.30):
    """Top-k of the `key` prompt ranking, greedily keeping mutually disjoint
    candidates using TRUE mask IoU."""
    S = p.S[key]
    top = p.tops[key]
    pair = p.pairs[key]
    order = np.argsort(-S[top], kind="stable")
    rng = float(S.max() - S.min())
    thr = S.max() - delta * rng if rng > 0 else S.max()
    loc = {int(m): j for j, m in enumerate(top)}
    kept = []
    for j in order:
        i = int(top[j])
        if len(kept) >= k:
            break
        if kept and S[i] < thr:
            break
        if all(pair[j, loc[m]] < split_iou for m in kept):
            kept.append(i)
    return kept, S


def resolve(p, cand, rel, swap=False):
    cx, cy = p.cx[cand], p.cy[cand]
    h = rel["h"] * (-1 if swap else 1)
    v = rel["v"] * (-1 if swap else 1)
    if rel["ordinal"]:
        axis = rel["axis"] or "h"
        d = (rel["direction"] or -1) * (-1 if swap else 1)
        order = np.argsort(d * (cx if axis == "h" else cy))
        pos = rel["ordinal"] - 1
        return cand[int(order[pos])] if pos < len(order) else None
    if h:
        return cand[int(np.argmin(cx))] if h < 0 else cand[int(np.argmax(cx))]
    if v:
        return cand[int(np.argmin(cy))] if v < 0 else cand[int(np.argmax(cy))]
    if rel["mid"]:
        return cand[int(np.argmin(np.abs(cx - 0.5)))]
    return None


def resolve_soft(p, cand, rel, S, lam):
    if len(cand) < 2:
        return None
    if rel["h"]:
        v = p.cx[cand]
    elif rel["v"]:
        v = p.cy[cand]
    elif rel["mid"]:
        v = -np.abs(p.cx[cand] - 0.5)
        return cand[int(np.argmax(S[cand] + lam * v))]
    else:
        return None
    r = np.argsort(np.argsort(v)).astype(float) / max(len(cand) - 1, 1)
    prior = r if (rel["h"] > 0 or rel["v"] > 0) else 1.0 - r
    return cand[int(np.argmax(S[cand] + lam * prior))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp1d/records_collect_s101.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp1e"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pools = [Pool(json.loads(l)) for l in args.records.open()]
    n = len(pools)
    base_ious = np.array([p.gt[p.base] for p in pools])
    base = float(base_ious.mean())
    base_o = (sum(p.gt[p.base] for p in pools), 0)  # placeholder, oIoU needs areas

    results = []

    def add(name, fn):
        ious = np.zeros(n)
        fired = 0
        for t, p in enumerate(pools):
            idx = fn(p)
            ious[t] = p.gt[idx]
            fired += int(idx != p.base)
        results.append(dict(rule=name, mIoU=float(ious.mean()), fired=fired,
                            fired_frac=fired / n, delta=float(ious.mean() - base)))
        return ious

    add("argmax", lambda p: p.base)
    for key in ("nospace", "content", "noun"):
        add(f"argmax_{key}", lambda p, key=key: int(np.argmax(p.S[key])))

    # --- controls: geometry over the whole pool, no category set
    for key, q in itertools.product(("full", "nospace", "content"), (0.25, 0.5, 1.0)):
        def f_all(p, key=key, q=q):
            if p.rel is None:
                return p.base
            S = p.S[key]
            thr = np.quantile(S, 1 - q)
            cand = [int(i) for i in np.flatnonzero(S >= thr)]
            c = resolve(p, cand, p.rel)
            return int(c) if c is not None else p.base
        add(f"geomall_{key}_q{q}", f_all)

    # --- main family: true-IoU confusion set + geometry
    for key, k, delta in itertools.product(("full", "nospace", "content"), (2, 3, 5, 8, 12), (0.10, 0.20, 0.35, 0.50, 1.01)):
        def f(p, key=key, k=k, delta=delta):
            cand, S = confusion(p, key, k, delta)
            if p.rel is None or len(cand) < 2:
                return p.base
            c = resolve(p, cand, p.rel)
            return int(c) if c is not None else p.base
        add(f"geom_{key}_k{k}_d{delta}", f)

    # --- swapped-relation control (same grid, best few)
    for key, k, delta in itertools.product(("nospace", "content"), (3, 5, 8), (0.35, 1.01)):
        def f(p, key=key, k=k, delta=delta):
            cand, S = confusion(p, key, k, delta)
            if p.rel is None or len(cand) < 2:
                return p.base
            c = resolve(p, cand, p.rel, swap=True)
            return int(c) if c is not None else p.base
        add(f"geomSWAP_{key}_k{k}_d{delta}", f)

    # --- soft spatial prior
    for key, lam, k, delta in itertools.product(("nospace", "content"), (0.05, 0.15, 0.30), (5, 8), (0.35, 1.01)):
        def f(p, key=key, lam=lam, k=k, delta=delta):
            cand, S = confusion(p, key, k, delta)
            if p.rel is None or len(cand) < 2:
                return p.base
            c = resolve_soft(p, cand, p.rel, p.S[key], lam)
            return int(c) if c is not None else p.base
        add(f"soft_{key}_k{k}_d{delta}_l{lam}", f)

    results.sort(key=lambda r: -r["mIoU"])
    with (args.output / "rule_search_trueiou.json").open("w") as fh:
        json.dump(dict(n=n, baseline=base, results=results), fh, indent=2)
    print(f"n={n} baseline(full/fused argmax)={base:.4f}")
    print("  top 20:")
    for r in results[:20]:
        print(f"  {r['rule']:30s} mIoU={r['mIoU']:.4f} d={r['delta']:+.4f} fired={r['fired']:4d} ({r['fired_frac']:.2f})")
    print("  controls / bottom:")
    for r in results:
        if r["rule"].startswith(("geomall", "geomSWAP", "argmax_")):
            print(f"  {r['rule']:30s} mIoU={r['mIoU']:.4f} d={r['delta']:+.4f} fired={r['fired']:4d} ({r['fired_frac']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
