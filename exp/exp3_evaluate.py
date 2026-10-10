#!/usr/bin/env python
"""EXP-3 — Full evaluation of a selection rule against the pre-registered gate.

For every rule: mIoU, oIoU (exact, from stored per-proposal pixel counts),
per-class and per-quartile breakdown, part-expression stratum, paired bootstrap
CI, sign-flip permutation p, and win/loss/tie counts against argmax.

Pre-registered gate (from EXPERIMENT_PLAN §EXP-4 / the project brief):
    delta mIoU >= +2.0 points over argmax
    paired bootstrap 95 % CI excluding 0
    oIoU not degraded
    part-expression stratum delta >= -1.0 mIoU
and `parent_strict` must appear as a row in the same table.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

H_WORDS = {"left": -1, "leftmost": -1, "l": -1, "right": +1, "rightmost": +1}
V_WORDS = {"top": -1, "upper": -1, "above": -1, "topmost": -1, "bottom": +1, "lower": +1, "below": +1}
MID = {"middle", "center", "centre", "mid", "central"}
ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
       "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}
PART_WORDS = {"hand", "hands", "arm", "arms", "leg", "legs", "head", "face", "foot", "feet", "eye", "eyes",
              "ear", "ears", "wheel", "wheels", "tail", "wing", "wings", "nose", "mouth", "neck", "finger",
              "fingers", "paw", "paws", "hoof", "beak", "horn", "horns", "antenna", "trunk", "handle",
              "lid", "door", "window", "sleeve", "collar", "tire", "tyre", "propeller", "blade", "stem"}


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
        self.inter = np.array(rec["inter_gt"])
        self.union = np.array(rec["union_gt"])
        self.S = {k: np.array(v["fused"]) for k, v in rec["scores"].items()}
        self.Sfg = {k: np.array(v["fg"]) for k, v in rec["scores"].items()}
        self.rel = parse(rec["sent"])
        self.orc = rec["orc"]
        self.ps_idx = np.array(rec["ps_idx"])
        self.tops = {k: np.array(v) for k, v in rec["tops"].items()}
        self.pairs = {k: np.array(v) for k, v in rec["pairs"].items()}
        self.base = int(np.argmax(self.S["full"]))


def parent_strict(p):
    """One-line containment snap: replace the argmax by the smallest proposal
    containing >= 95 % of it. Index is precomputed over the FULL pool."""
    return int(p.ps_idx[p.base])


def confusion(p, key, k, delta, split_iou=0.30):
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


def resolve(p, cand, rel):
    cx, cy = p.cx[cand], p.cy[cand]
    if rel["ordinal"]:
        axis = rel["axis"] or "h"
        d = rel["direction"] or -1
        order = np.argsort(d * (cx if axis == "h" else cy))
        pos = rel["ordinal"] - 1
        return cand[int(order[pos])] if pos < len(order) else None
    if rel["h"]:
        return cand[int(np.argmin(cx))] if rel["h"] < 0 else cand[int(np.argmax(cx))]
    if rel["v"]:
        return cand[int(np.argmin(cy))] if rel["v"] < 0 else cand[int(np.argmax(cy))]
    if rel["mid"]:
        return cand[int(np.argmin(np.abs(cx - 0.5)))]
    return None


def rule_confusion(p, key="content", k=3, delta=0.35, split_iou=0.30):
    if p.rel is None:
        return p.base
    cand, _ = confusion(p, key, k, delta, split_iou)
    if len(cand) < 2:
        return p.base
    c = resolve(p, cand, p.rel)
    return int(c) if c is not None else p.base


RULES = {
    "argmax": lambda p: p.base,
    "parent_strict": parent_strict,
    "ours": lambda p: rule_confusion(p, "content", 3, 0.35),
    "ours_nospace": lambda p: rule_confusion(p, "nospace", 3, 0.35),
    "ours_k5": lambda p: rule_confusion(p, "content", 5, 0.35),
    "ours_k8": lambda p: rule_confusion(p, "content", 8, 0.35),
    "ours_d020": lambda p: rule_confusion(p, "content", 3, 0.20),
    "ours_d050": lambda p: rule_confusion(p, "content", 3, 0.50),
    "ours_d101": lambda p: rule_confusion(p, "content", 3, 1.01),
}


def boot(y_a, y_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = y_b - y_a
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    ds = d[idx].mean(1)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    # sign-flip permutation
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n))
    perm = (d * signs).mean(1)
    p = float((np.abs(perm) >= abs(d.mean())).mean())
    return float(d.mean()), float(lo), float(hi), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp1d/records_collect_s101.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp3"))
    ap.add_argument("--tag", default="eval_s101")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pools = [Pool(json.loads(l)) for l in args.records.open()]
    n = len(pools)
    preds = {name: np.array([fn(p) for p in pools]) for name, fn in RULES.items()}
    y = {name: np.array([pools[t].gt[preds[name][t]] for t in range(n)]) for name in RULES}
    inter = {name: np.array([pools[t].inter[preds[name][t]] for t in range(n)]) for name in RULES}
    union = {name: np.array([pools[t].union[preds[name][t]] for t in range(n)]) for name in RULES}

    gt_area = np.array([p.r["gt_area"] for p in pools])
    is_part = np.array([bool(set(w.strip(".,!?").lower() for w in p.r["sent"].split()) & PART_WORDS)
                        for p in pools])
    has_rel = np.array([p.rel is not None for p in pools])
    q = np.quantile(gt_area, [0.25, 0.5, 0.75])
    qidx = np.digitize(gt_area, q, right=True)
    oracle = np.array([p.gt[p.orc] for p in pools])

    base = y["argmax"]
    rows = []
    for name in RULES:
        d, lo, hi, pv = boot(base, y[name], n_boot=args.boot)
        rows.append(dict(
            rule=name, mIoU=float(y[name].mean()), delta=d, ci=[lo, hi], p=pv,
            oIoU=float(inter[name].sum() / union[name].sum()),
            oIoU_argmax=float(inter["argmax"].sum() / union["argmax"].sum()),
            fired=int((preds[name] != preds["argmax"]).sum()),
            wlt=dict(win=int((y[name] > base).sum()), loss=int((y[name] < base).sum()),
                     tie=int((y[name] == base).sum())),
            part=dict(n=int(is_part.sum()),
                      delta=float((y[name] - base)[is_part].mean()) if is_part.sum() else None),
            no_rel=dict(n=int((~has_rel).sum()), delta=float((y[name] - base)[~has_rel].mean())),
            quartiles=[float((y[name] - base)[qidx == i].mean()) for i in range(4)],
        ))

    gate = {}
    o = next(r for r in rows if r["rule"] == "ours")
    gate["delta_ge_2"] = bool(o["delta"] >= 0.02)
    gate["ci_excludes_0"] = bool(o["ci"][0] > 0)
    gate["oIoU_not_degraded"] = bool(o["oIoU"] >= o["oIoU_argmax"] - 1e-9)
    gate["part_ok"] = bool(o["part"]["delta"] >= -0.01)
    gate["parent_strict_row_present"] = "parent_strict" in RULES
    gate["PASS"] = all(gate.values())

    out = dict(tag=args.tag, n=n, oracle_mIoU=float(oracle.mean()),
               baseline_mIoU=float(base.mean()), has_rel_frac=float(has_rel.mean()),
               rows=rows, gate=gate)
    with (args.output / f"eval_{args.tag}.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"n={n} oracle_mIoU={oracle.mean():.4f} argmax_mIoU={base.mean():.4f} "
          f"spatial_expr={has_rel.mean():.2f}")
    print(f"{'rule':<16}{'mIoU':>8}{'delta':>9}{'CI':>20}{'p':>9}{'oIoU':>8}{'fired':>7}{'part':>9}")
    for r in rows:
        print(f"{r['rule']:<16}{r['mIoU']:>8.4f}{r['delta']:>+9.4f}"
              f"{('['+format(r['ci'][0],'+.4f')+','+format(r['ci'][1],'+.4f')+']'):>20}"
              f"{r['p']:>9.4f}{r['oIoU']:>8.4f}{r['fired']:>7d}"
              f"{(r['part']['delta'] if r['part']['delta'] is not None else float('nan')):>+9.4f}")
    print("gate:", json.dumps(gate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
