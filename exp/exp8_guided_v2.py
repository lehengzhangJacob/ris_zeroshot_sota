#!/usr/bin/env python
"""EXP-8 — Category-set relation guidance: the method, rebuilt on the diagnosis.

EXP-7 located the guided-point bottleneck precisely:

  * the released guidance can only return one of the top-3 proposals under the
    FULL expression score (`maxidxs = topk(score_clip, k1=3)`);
  * on wrong-instance failures the reference sits at mean score-rank 9.1, and is
    inside the released top-3 only 30.2 % of the time;
  * under the *category* score (the expression with spatial vocabulary removed)
    the reference is inside the top-8 for 70.1 % of those same cases;
  * when the reference is in the candidate set, a geometric resolver selects it
    79-84 % of the time.

So the fix is the set, not the resolver: enlarge the candidate set to the
category-consistent confusion set, keep the released relation/coherence
machinery, and only override when the relation can actually adjudicate.

Rule families measured here:
  geom      extreme geometry over the category set            (EXP-5 baseline)
  side+gem  relation-side filter, then the released coherence score
  side+sem  relation-side filter, then semantic score
  union     released top-3 unioned with the category set, side-filtered
  confirm   override only when the released pick fails the relation-side test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, confusion, resolve, PART_WORDS  # noqa: E402


def cat_set(r, key, k, delta, split_iou=0.20):
    return confusion(r, key, k, delta, split_iou)


def side_mask(r, cand, rel, swap=False):
    """Boolean: which candidates satisfy the relation's side constraint.

    Relative to the confusion set's own median along the relation axis, so the
    constraint is 'the left of these instances', not 'the left half of the
    image'.
    """
    cx = r.cx[cand]
    cy = r.cy[cand]
    h = -rel["h"] if swap else rel["h"]
    v = -rel["v"] if swap else rel["v"]
    if rel["mid"] and not (h or v):
        d = np.abs(cx - 0.5)
        return d <= np.median(d)
    if h:
        med = np.median(cx)
        m = cx <= med if h < 0 else cx >= med
    elif v:
        med = np.median(cy)
        m = cy <= med if v < 0 else cy >= med
    else:
        return np.ones(len(cand), bool)
    if swap:
        return ~m
    return m


def rank_pick(r, cand, rel, by, swap=False):
    m = side_mask(r, cand, rel, swap)
    sub = [c for c, keep in zip(cand, m) if keep]
    if not sub:
        return None
    if by == "gem":
        return sub[int(np.argmax(r.S_gem[sub]))]
    if by == "sem":
        return sub[int(np.argmax(r.S["sc_strip"][sub]))]
    if by == "sc":
        return sub[int(np.argmax(r.S["sc"][sub]))]
    return sub[int(np.argmax(r.S["sc_content"][sub]))]


def build_rules():
    R = {}
    R["argmax_g2l"] = lambda r: r.pure
    R["parent_strict"] = lambda r: int(r.ps[r.pure])
    R["hybridgl_guided"] = lambda r: r.final
    R["parent_strict_guided"] = lambda r: int(r.ps[r.final])

    def geom(key, k, delta):
        def f(r):
            if r.rel is None:
                return r.final
            cand = cat_set(r, key, k, delta)
            if len(cand) < 2:
                return r.final
            c = resolve(r, cand, r.rel)
            return int(c) if c is not None else r.final
        return f

    def side(key, k, delta, by):
        def f(r):
            if r.rel is None:
                return r.final
            cand = cat_set(r, key, k, delta)
            if len(cand) < 2:
                return r.final
            c = rank_pick(r, cand, r.rel, by)
            return int(c) if c is not None else r.final
        return f

    def union_side(k, delta, by):
        def f(r):
            if r.rel is None:
                return r.final
            cand = sorted(set(cat_set(r, "sc_content", k, delta)) | set(r.top.tolist()))
            if len(cand) < 2:
                return r.final
            c = rank_pick(r, cand, r.rel, by)
            return int(c) if c is not None else r.final
        return f

    def confirm(key, k, delta, by):
        """Keep the released pick unless it fails the relation-side test."""
        def f(r):
            if r.rel is None:
                return r.final
            cand = sorted(set(cat_set(r, key, k, delta)) | {int(r.final)})
            if len(cand) < 2:
                return r.final
            m = side_mask(r, cand, r.rel)
            keep = dict(zip(cand, m))
            if keep.get(int(r.final), True):
                return r.final
            sub = [c for c in cand if keep[c]]
            if not sub:
                return r.final
            if by == "gem":
                return sub[int(np.argmax(r.S_gem[sub]))]
            return sub[int(np.argmax(r.S["sc_strip"][sub]))]
        return f

    for key in ("sc_content", "sc_strip"):
        for k in (3, 5, 8, 12):
            for d in (0.20, 0.35, 1.01):
                R[f"geom_{key}_k{k}_d{d}"] = geom(key, k, d)
                R[f"sideGEM_{key}_k{k}_d{d}"] = side(key, k, d, "gem")
                R[f"sideSEM_{key}_k{k}_d{d}"] = side(key, k, d, "sem")
                R[f"confirm_{key}_k{k}_d{d}"] = confirm(key, k, d, "gem")
    for k in (5, 8, 12):
        for d in (0.20, 0.35):
            R[f"unionGEM_k{k}_d{d}"] = union_side(k, d, "gem")
            R[f"unionSEM_k{k}_d{d}"] = union_side(k, d, "sem")
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp8"))
    ap.add_argument("--tag", default="refcoco_guided")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    print(f"{len(recs)} records", flush=True)

    sel_ids = set(json.loads(args.select_refs.read_text()))
    splits = {"all": recs,
              "select": [r for r in recs if r.r["ref_id"] in sel_ids],
              "heldout": [r for r in recs if r.r["ref_id"] not in sel_ids]}
    R = build_rules()
    print(f"{len(R)} rules", flush=True)

    def evaluate(rs, names):
        y = {nm: np.array([rs[t].gt[R[nm](rs[t])] for t in range(len(rs))]) for nm in names}
        return y

    # rank on the selection split
    sel = splits["select"]
    names = list(R)
    ysel = evaluate(sel, names)
    ref_sel = ysel["hybridgl_guided"]
    ranked = sorted(names, key=lambda k: -ysel[k].mean())
    print(f"\nranking on selection split n={len(sel)}")
    print(f"{'rule':<30}{'mIoU':>8}{'d_guided':>10}")
    for nm in ranked[:18]:
        print(f"{nm:<30}{ysel[nm].mean():>8.4f}{(ysel[nm]-ref_sel).mean():>+10.4f}")
    print("  worst:", ranked[-1], f"{ysel[ranked[-1]].mean():.4f}")

    # report the family winners on the full split and held-out
    keep = ["argmax_g2l", "parent_strict", "hybridgl_guided", "parent_strict_guided"]
    for fam in ("geom", "sideGEM", "sideSEM", "confirm", "unionGEM"):
        best = next((nm for nm in ranked if nm.startswith(fam)), None)
        if best:
            keep.append(best)
    keep += [nm for nm in ranked[:3] if nm not in keep]

    summary = {}
    for sname, rs in splits.items():
        y = evaluate(rs, keep)
        part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS)
                         for r in rs])
        ref = y["hybridgl_guided"]
        rows = {}
        for nm in keep:
            d, lo, hi, p = boot(ref, y[nm])
            rows[nm] = dict(mIoU=float(y[nm].mean()), delta=d, ci=[lo, hi], p=p,
                            part_delta=float((y[nm] - ref)[part].mean()) if part.sum() else None,
                            fired=int((y[nm] != ref).sum()),
                            win=int((y[nm] > ref).sum()), loss=int((y[nm] < ref).sum()))
        summary[sname] = dict(n=len(rs), rows=rows)
        with (args.output / f"rows_{args.tag}_{sname}.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    print(f"\n=== FULL SPLIT n={summary['all']['n']} ===")
    s = summary["all"]["rows"]
    print(f"{'rule':<30}{'mIoU':>8}{'d_guided':>10}{'CI':>20}{'p':>8}{'fired':>7}{'win':>6}{'loss':>6}")
    for nm in sorted(s, key=lambda k: -s[k]["mIoU"]):
        r = s[nm]
        print(f"{nm:<30}{r['mIoU']:>8.4f}{r['delta']:>+10.4f}"
              f"{('['+format(r['ci'][0],'+.4f')+','+format(r['ci'][1],'+.4f')+']'):>20}"
              f"{r['p']:>8.4f}{r['fired']:>7d}{r['win']:>6d}{r['loss']:>6d}")
    print("\n=== HELD-OUT COMPLEMENT ===")
    hs = summary["heldout"]["rows"]
    print(f"{'rule':<30}{'mIoU':>8}{'d_guided':>10}{'CI':>20}{'fired':>7}")
    for nm in sorted(hs, key=lambda k: -hs[k]["mIoU"]):
        r = hs[nm]
        print(f"{nm:<30}{r['mIoU']:>8.4f}{r['delta']:>+10.4f}"
              f"{('['+format(r['ci'][0],'+.4f')+','+format(r['ci'][1],'+.4f')+']'):>20}{r['fired']:>7d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
