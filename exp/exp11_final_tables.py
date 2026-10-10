#!/usr/bin/env python
"""EXP-11 — Assemble the paper's tables from the stored passes.

Produces one JSON plus a markdown rendering with every number the paper cites.
Nothing here re-runs a model: all rows are re-analyses of the EXP-4 records
(RefCOCO val, released HybridGL scorer) and the EXP-1d records (unguided point).

Method under test (selected on the seed-101 selection split, reported on the
full split and the held-out complement):

    category-set relation guidance
      1. score every proposal with the released scorer against the expression
         with its spatial vocabulary removed  (the "category score")
      2. candidate set = top-k by that score
      3. if the expression carries a spatial relation, keep the half of the
         candidate set lying on the relation's side (median split along the
         relation axis) and return the candidate with the highest released
         coherence score (GEM attention x position mask)
      4. otherwise return the released selection unchanged
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import (CLASSES, Rec, boot, confusion, parse, reltype,  # noqa: E402
                                 PART_WORDS)
from exp9_guided_v3 import pick  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--taxonomy", type=Path, default=Path("exp/out_exp5/taxonomy_refcoco_guided.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    ap.add_argument("--tag", default="refcoco")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    n = len(recs)
    print(f"{n} records", flush=True)

    def cat(nm):
        return np.argsort(-r.S[nm], kind="stable")[:8].tolist() if False else None  # placeholder

    def ours(r, k=8, pct=0.5, by="gem", key="sc_content"):
        if r.rel is None:
            return r.final
        cand = sorted(np.argsort(-r.S[key], kind="stable")[:k].tolist())
        if len(cand) < 2:
            return r.final
        idx = pick(r, cand, r.rel, pct, by)
        return int(idx) if idx is not None else r.final

    ROWS = {
        "argmax_g2l": lambda r: r.pure,
        "hybridgl_guided": lambda r: r.final,
        "parent_strict": lambda r: int(r.ps[r.final]),
        "ours": ours,
        "ours_k5": lambda r: ours(r, k=5),
        "ours_k12": lambda r: ours(r, k=12),
        "ours_k16": lambda r: ours(r, k=16),
        "ours_k24": lambda r: ours(r, k=24),
        "ours_p34": lambda r: ours(r, pct=0.34),
        "ours_p66": lambda r: ours(r, pct=0.66),
        "ours_rankgem_sem": lambda r: ours(r, by="gem_sem"),
        "ours_ranksem": lambda r: ours(r, by="sem"),
        "ours_set_full": lambda r: ours(r, key="sc"),
        "ours_set_strip": lambda r: ours(r, key="sc_strip"),
        "ours_set_noun": lambda r: ours(r, key="sc_noun"),
        "ours_snap": lambda r: int(r.ps[ours(r)]),
    }

    y = {nm: np.array([recs[t].gt[fn(recs[t])] for t in range(n)]) for nm, fn in ROWS.items()}
    it = {nm: np.array([recs[t].inter[fn(recs[t])] for t in range(n)]) for nm, fn in ROWS.items()}
    un = {nm: np.array([recs[t].union[fn(recs[t])] for t in range(n)]) for nm, fn in ROWS.items()}

    part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in recs])
    relty = np.array([reltype(r.rel) for r in recs])
    gt_area = np.array([r.gt_area for r in recs])
    q = np.quantile(gt_area, [0.25, 0.5, 0.75])
    qi = np.digitize(gt_area, q, right=True)

    ref = y["hybridgl_guided"]
    arg = y["argmax_g2l"]
    rows = {}
    for nm in ROWS:
        d, lo, hi, p = boot(ref, y[nm])
        da, alo, ahi, ap = boot(arg, y[nm])
        rows[nm] = dict(
            mIoU=float(y[nm].mean()), oIoU=float(it[nm].sum() / un[nm].sum()),
            delta_vs_guided=d, ci_vs_guided=[lo, hi], p_vs_guided=p,
            delta_vs_argmax=da, ci_vs_argmax=[alo, ahi], p_vs_argmax=ap,
            fired=int((y[nm] != ref).sum()),
            win=int((y[nm] > ref).sum()), loss=int((y[nm] < ref).sum()),
            part_delta=float((y[nm] - ref)[part].mean()) if part.sum() else None,
            part_n=int(part.sum()),
            no_rel_delta=float((y[nm] - ref)[~part & (relty == "none")].mean())
            if (relty == "none").sum() else None,
            strata={rt: float((y[nm] - ref)[relty == rt].mean()) for rt in
                    ("none", "horizontal", "vertical", "middle", "ordinal") if (relty == rt).sum()},
            quartile_delta=[float((y[nm] - ref)[qi == i].mean()) for i in range(4)],
            oracle_hit_rate=float(np.mean([y[nm][t] == recs[t].gt[recs[t].orc] for t in range(n)])),
        )

    tax = json.loads(args.taxonomy.read_text()) if args.taxonomy.exists() else {}
    out = dict(tag=args.tag, n=n, oracle_mIoU=float(np.mean([r.gt[r.orc] for r in recs])),
               spatial_expr_frac=float(np.mean([r.rel is not None for r in recs])),
               rows=rows, taxonomy=tax)
    with (args.output / f"tables_{args.tag}.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    def line(nm):
        v = rows[nm]
        pd = v["part_delta"]
        return (f"| {nm} | {v['mIoU']:.4f} | {v['oIoU']:.4f} | {v['delta_vs_guided']:+.4f} "
                f"[{v['ci_vs_guided'][0]:+.4f},{v['ci_vs_guided'][1]:+.4f}] | {v['delta_vs_argmax']:+.4f} "
                f"| {v['fired']} | {(pd if pd is not None else float('nan')):+.4f} |")

    md = [f"# Results — {args.tag}", "",
          f"n = {n} expressions; oracle mIoU = {out['oracle_mIoU']:.4f}; "
          f"expressions carrying a spatial relation = {out['spatial_expr_frac']:.3f}", "",
          "## Main table", "",
          "| row | mIoU | oIoU | Δ vs released | 95% CI | Δ vs argmax | changed | part Δ |",
          "|---|---|---|---|---|---|---|---|"]
    for nm in ("argmax_g2l", "hybridgl_guided", "parent_strict", "ours", "ours_snap"):
        md.append(line(nm))
    md += ["", "## Sensitivity", "", "| row | mIoU | oIoU | Δ vs released | 95% CI | Δ vs argmax | changed | part Δ |",
           "|---|---|---|---|---|---|---|---|"]
    for nm in ("ours_k5", "ours_k12", "ours_k16", "ours_k24", "ours_p34", "ours_p66",
               "ours_rankgem_sem", "ours_ranksem", "ours_set_full", "ours_set_strip", "ours_set_noun"):
        md.append(line(nm))
    md += ["", "## Ours by relation type (Δ vs released)", ""]
    o = rows["ours"]
    for rt, v in o["strata"].items():
        md.append(f"- {rt}: {v:+.4f}")
    md += ["", "## Ours by GT-area quartile (Δ vs released)"]
    for i, v in enumerate(o["quartile_delta"]):
        md.append(f"- q{i+1}: {v:+.4f}")
    md += ["", f"oracle-hit rate: ours={o['oracle_hit_rate']:.3f} released={rows['hybridgl_guided']['oracle_hit_rate']:.3f}"]
    (args.output / f"tables_{args.tag}.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
