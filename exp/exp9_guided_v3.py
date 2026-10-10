#!/usr/bin/env python
"""EXP-9 — Wider search around the category-set candidate set.

EXP-8 showed the win comes from enlarging the released candidate set (top-3 by
the full expression score) to a category-consistent set, then applying the
relation side-constraint and the released coherence score. This script sweeps
the remaining design choices on the SELECTION split only, then reports the
winner on the full split and the held-out complement.

Also measures composition with the containment snap, which targets a different
error class (under-segmentation, 17.6 % of the gap mass at this operating
point); the composition is reported as an explicitly attributed extra row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, confusion, PART_WORDS  # noqa: E402


def side_of(r, cand, rel, pct=0.5):
    cx = r.cx[cand]
    cy = r.cy[cand]
    if rel["h"]:
        v, sign = cx, rel["h"]
    elif rel["v"]:
        v, sign = cy, rel["v"]
    elif rel["mid"]:
        v, sign = -np.abs(cx - 0.5), +1
    else:
        return None
    thr = np.quantile(v, pct) if sign < 0 else np.quantile(v, 1 - pct)
    return v <= thr if sign < 0 else v >= thr


def pick(r, cand, rel, pct, by):
    m = side_of(r, cand, rel, pct)
    if m is None:
        return None
    sub = np.array(cand)[m]
    if not len(sub):
        return None
    if by == "gem":
        k = r.S_gem[sub]
    elif by == "gem_sem":
        gg = r.S_gem[sub]
        k = (gg - gg.min()) / (gg.ptp() + 1e-9) + r.S["sc_content"][sub]
    elif by == "gem_strip":
        gg = r.S_gem[sub]
        k = (gg - gg.min()) / (gg.ptp() + 1e-9) + r.S["sc_strip"][sub]
    elif by == "sem":
        k = r.S["sc_content"][sub]
    elif by == "sc":
        k = r.S["sc"][sub]
    else:
        raise ValueError(by)
    return int(sub[int(np.argmax(k))])


def cand_set(r, mode, k, delta, disjoint):
    if mode == "conf":
        return sorted(confusion(r, "sc_content", k, delta, 0.20 if disjoint else 1.01))
    if mode == "topk":
        return sorted(np.argsort(-r.S["sc_content"], kind="stable")[:k].tolist())
    if mode == "union":
        return sorted(set(confusion(r, "sc_content", k, delta, 0.20 if disjoint else 1.01))
                      | set(r.top.tolist()))
    if mode == "unionTopk":
        return sorted(set(np.argsort(-r.S["sc_content"], kind="stable")[:k].tolist())
                      | set(r.top.tolist()))
    raise ValueError(mode)


def make_rule(mode, k, delta, disjoint, pct, by, snap):
    def f(r):
        if r.rel is None:
            return int(r.ps[r.final]) if snap else r.final
        cand = cand_set(r, mode, k, delta, disjoint)
        if len(cand) < 2:
            return int(r.ps[r.final]) if snap else r.final
        idx = pick(r, cand, r.rel, pct, by)
        if idx is None:
            return int(r.ps[r.final]) if snap else r.final
        return int(r.ps[idx]) if snap else int(idx)
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp9"))
    ap.add_argument("--tag", default="refcoco_guided")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    print(f"{len(recs)} records", flush=True)
    sel_ids = set(json.loads(args.select_refs.read_text()))
    seen = {}
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
        if r.r["ref_id"] in sel_ids:
            seen[id(r)] = True
    sel = [r for r in recs if r.r["ref_id"] in sel_ids]
    held = [r for r in recs if r.r["ref_id"] not in sel_ids]

    R = {"argmax_g2l": lambda r: r.pure,
         "parent_strict": lambda r: int(r.ps[r.pure]),
         "parent_strict_guided": lambda r: int(r.ps[r.final]),
         "hybridgl_guided": lambda r: r.final}
    grid = []
    for mode, k in itertools.product(("conf", "topk", "union", "unionTopk"), (5, 8, 12, 16, 24, 32)):
        if mode == "conf" and k < 8:
            continue
        for disjoint, pct, by, snap in itertools.product((True, False), (0.5, 0.34), ("gem", "gem_sem"), (True, False)):
            if not disjoint and mode in ("conf",):
                continue
            name = f"{mode}{k}{'_dj' if disjoint else ''}_p{int(pct*100)}_{by}{'_snap' if snap else ''}"
            R[name] = make_rule(mode, k, 0.35, disjoint, pct, by, snap)
            grid.append(name)
    print(f"{len(R)} rules", flush=True)

    ysel = {nm: np.array([sel[t].gt[R[nm](sel[t])] for t in range(len(sel))]) for nm in R}
    ref = ysel["hybridgl_guided"]
    ranked = sorted(R, key=lambda k: -ysel[k].mean())
    print(f"\nSELECTION SPLIT n={len(sel)} — top 25")
    print(f"{'rule':<40}{'mIoU':>8}{'d_guided':>10}")
    for nm in ranked[:25]:
        print(f"{nm:<40}{ysel[nm].mean():>8.4f}{(ysel[nm]-ref).mean():>+10.4f}")
    print("  ...bottom 5")
    for nm in ranked[-5:]:
        print(f"{nm:<40}{ysel[nm].mean():>8.4f}{(ysel[nm]-ref).mean():>+10.4f}")
    noSnap = [nm for nm in ranked if not nm.endswith("_snap") and nm not in
              ("argmax_g2l", "parent_strict", "parent_strict_guided", "hybridgl_guided")]
    chosen = noSnap[0]
    bestsnap = next((nm for nm in ranked if nm.endswith("_snap")), None)
    print(f"\nCHOSEN (no-snap family, selected on selection split): {chosen}")
    print(f"best composed row: {bestsnap}")

    keep = ["argmax_g2l", "parent_strict", "hybridgl_guided", "parent_strict_guided", chosen]
    if bestsnap and bestsnap not in keep:
        keep.append(bestsnap)
    for nm in ranked[:4]:
        if nm not in keep:
            keep.append(nm)

    summary = {}
    for sname, rs in (("all", recs), ("select", sel), ("heldout", held)):
        y = {nm: np.array([rs[t].gt[R[nm](rs[t])] for t in range(len(rs))]) for nm in keep}
        inter = {nm: np.array([rs[t].inter[R[nm](rs[t])] for t in range(len(rs))]) for nm in keep}
        un = {nm: np.array([rs[t].union[R[nm](rs[t])] for t in range(len(rs))]) for nm in keep}
        part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in rs])
        nor = np.array([r.rel is None for r in rs])
        r0 = y["hybridgl_guided"]
        rows = {}
        for nm in keep:
            d, lo, hi, p = boot(r0, y[nm])
            rows[nm] = dict(mIoU=float(y[nm].mean()), oIoU=float(inter[nm].sum() / un[nm].sum()),
                            delta=d, ci=[lo, hi], p=p,
                            part_delta=float((y[nm] - r0)[part].mean()) if part.sum() else None,
                            no_rel_delta=float((y[nm] - r0)[nor].mean()) if nor.sum() else None,
                            fired=int((y[nm] != r0).sum()),
                            win=int((y[nm] > r0).sum()), loss=int((y[nm] < r0).sum()))
        summary[sname] = dict(n=len(rs), rows=rows)
        with (args.output / f"rows_{args.tag}_{sname}.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    for sname in ("all", "heldout"):
        print(f"\n=== {sname.upper()} n={summary[sname]['n']} ===")
        s = summary[sname]["rows"]
        print(f"{'rule':<40}{'mIoU':>8}{'oIoU':>8}{'d_guided':>10}{'CI':>20}{'p':>8}{'fired':>7}{'part':>8}")
        for nm in sorted(s, key=lambda k: -s[k]["mIoU"]):
            v = s[nm]
            pd = v["part_delta"]
            print(f"{nm:<40}{v['mIoU']:>8.4f}{v['oIoU']:>8.4f}{v['delta']:>+10.4f}"
                  f"{('['+format(v['ci'][0],'+.4f')+','+format(v['ci'][1],'+.4f')+']'):>20}"
                  f"{v['p']:>8.4f}{v['fired']:>7d}{(pd if pd is not None else float('nan')):>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
