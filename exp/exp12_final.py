#!/usr/bin/env python
"""EXP-12 — Final evaluation of the method in `method.py`.

Selection split = the seed-101 sample of 250 references (716 expressions),
chosen before this script was written. Everything else is a held-out report.

Produces the paper's main table, sensitivity table, controls, and strata, plus
a JSON dump for the write-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, boot, PART_WORDS, reltype  # noqa: E402
from method import Selector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    ap.add_argument("--tag", default="refcoco")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    sel_ids = set(json.loads(args.select_refs.read_text()))
    print(f"{len(recs)} records", flush=True)

    ROWS = {
        "argmax_g2l": lambda r: r.pure,
        "hybridgl_guided": lambda r: r.final,
        "parent_strict": lambda r: int(r.ps[r.final]),
        "ours": Selector(),
        "ours_snap": lambda r: int(r.ps[Selector()(r)]),
        "ours_k5": Selector(k=5),
        "ours_ordfire": Selector(ordinals=True),
        "ours_ctrl_allpool": None,
        "ours_rank_noun": Selector(score="sc_noun"),
        "ours_k12": Selector(k=12),
        "ours_k16": Selector(k=16),
        "ours_k24": Selector(k=24),
        "ours_p34": Selector(pct=0.34),
        "ours_p66": Selector(pct=0.66),
        "ours_rank_sem": Selector(score="sc_content"),
        "ours_rank_sc": Selector(score="sc"),
        "ours_set_full": Selector(key="sc"),
        "ours_set_strip": Selector(key="sc_strip"),
        "ours_set_noun": Selector(key="sc_noun"),
        "ctrl_geom_extreme": None,   # filled below
        "ctrl_no_side": None,
        "ctrl_swap": None,
        "ctrl_allpool": None,
    }

    def geom_extreme(r):
        """Extreme geometry over the category set, ignoring coherence and side."""
        rel = r.rel
        if rel is None:
            return r.final
        cand = np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()
        if len(cand) < 2:
            return r.final
        from method import select
        if rel["ordinal"]:
            return r.final
        if rel["h"]:
            p = int(np.argmin(r.cx[cand])) if rel["h"] < 0 else int(np.argmax(r.cx[cand]))
        elif rel["v"]:
            p = int(np.argmin(r.cy[cand])) if rel["v"] < 0 else int(np.argmax(r.cy[cand]))
        elif rel["mid"]:
            p = int(np.argmin(np.abs(r.cx[cand] - 0.5)))
        else:
            return r.final
        return int(cand[p])

    def no_side(r):
        """Top-k category set, ranked by coherence, with no relation at all."""
        if r.rel is None:
            return r.final
        cand = np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()
        return int(cand[int(np.argmax(r.S_gem[cand]))])

    def swapped(r):
        """Same operator with the relation mirrored."""
        rel = r.rel
        if rel is None:
            return r.final
        rel2 = dict(rel, h=-rel["h"], v=-rel["v"])
        cand = np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()
        from method import select
        idx = select(cand, rel2, r.cx, r.cy, r.S_gem, 0.5)
        return int(idx) if idx is not None else r.final

    def allpool(r):
        """Relation applied to the relation-consistent half of the WHOLE pool,
        with no category set."""
        rel = r.rel
        if rel is None or rel["ordinal"]:
            return r.final
        cand = list(range(len(r.S_gem)))
        from method import select
        idx = select(cand, rel, r.cx, r.cy, r.S_gem, 0.5)
        return int(idx) if idx is not None else r.final

    ROWS["ctrl_geom_extreme"] = geom_extreme
    ROWS["ctrl_no_side"] = no_side
    ROWS["ctrl_swap"] = swapped
    ROWS["ctrl_allpool"] = allpool
    ROWS["ours_ctrl_allpool"] = allpool

    part = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS) for r in recs])
    relty = np.array([reltype(r.rel) for r in recs])
    gt_area = np.array([r.gt_area for r in recs])
    qi = np.digitize(gt_area, np.quantile(gt_area, [0.25, 0.5, 0.75]), right=True)

    def evaluate(rs, names, tag):
        idx = {nm: np.array([ROWS[nm](r) for r in rs]) for nm in names}
        y = {nm: np.array([rs[t].gt[idx[nm][t]] for t in range(len(rs))]) for nm in names}
        it = {nm: np.array([rs[t].inter[idx[nm][t]] for t in range(len(rs))]) for nm in names}
        un = {nm: np.array([rs[t].union[idx[nm][t]] for t in range(len(rs))]) for nm in names}
        ref, arg = y["hybridgl_guided"], y["argmax_g2l"]
        rows = {}
        for nm in names:
            d, lo, hi, p = boot(ref, y[nm])
            da, alo, ahi, ap = boot(arg, y[nm])
            rows[nm] = dict(
                mIoU=float(y[nm].mean()), oIoU=float(it[nm].sum() / un[nm].sum()),
                delta_vs_guided=d, ci_vs_guided=[lo, hi], p_vs_guided=p,
                delta_vs_argmax=da, ci_vs_argmax=[alo, ahi], p_vs_argmax=ap,
                fired=int((idx[nm] != idx["hybridgl_guided"]).sum()),
                win=int((y[nm] > ref).sum()), loss=int((y[nm] < ref).sum()))
            if tag == "all":
                pt = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & PART_WORDS)
                               for r in rs])
                rt = np.array([reltype(r.rel) for r in rs])
                rows[nm].update(
                    part_delta=float((y[nm] - ref)[pt].mean()) if pt.sum() else None, part_n=int(pt.sum()),
                    no_relation_delta=float((y[nm] - ref)[rt == "none"].mean()),
                    strata={k: float((y[nm] - ref)[rt == k].mean()) for k in
                            ("none", "horizontal", "vertical", "middle", "ordinal") if (rt == k).sum()},
                    strata_n={k: int((rt == k).sum()) for k in
                              ("none", "horizontal", "vertical", "middle", "ordinal") if (rt == k).sum()},
                    quartile_delta=[float((y[nm] - ref)[qi == i].mean()) for i in range(4)],
                    oracle_hit=float(np.mean([y[nm][t] == rs[t].gt[rs[t].orc] for t in range(len(rs))])),
                    mean_iou_when_correct=float(np.mean([y[nm][t] for t in range(len(rs))
                                                         if y[nm][t] >= 0.5])) if (y[nm] >= 0.5).any() else None)
        return rows

    names = list(ROWS)
    allr = recs
    selr = [r for r in recs if r.r["ref_id"] in sel_ids]
    heldr = [r for r in recs if r.r["ref_id"] not in sel_ids]

    # selection split ranking (sanity: the shipped configuration must be a good
    # but not cherry-picked choice there)
    srows = evaluate(selr, names, "select")
    print(f"\nSELECTION SPLIT n={len(selr)} (configuration chosen before this script)")
    for nm in sorted(srows, key=lambda k: -srows[k]["mIoU"])[:8]:
        print(f"  {nm:<22}{srows[nm]['mIoU']:.4f}{srows[nm]['delta_vs_guided']:+.4f}")

    out = {"selection_split_ranking": {k: srows[k]["mIoU"] for k in
                                       sorted(srows, key=lambda k: -srows[k]["mIoU"])}}
    for tagname, rs in (("all", allr), ("heldout", heldr)):
        rows = evaluate(rs, names, "all" if tagname == "all" else tagname)
        out[tagname] = dict(n=len(rs), rows=rows)
        if tagname == "all":
            out["oracle_mIoU"] = float(np.mean([r.gt[r.orc] for r in rs]))
            out["spatial_expr_frac"] = float(np.mean([r.rel is not None for r in rs]))
            out["n"] = len(rs)

    with (args.output / f"final_{args.tag}.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    def fmt(nm, rows):
        v = rows[nm]
        return (f"| {nm} | {v['mIoU']:.4f} | {v['oIoU']:.4f} | {v['delta_vs_guided']:+.4f} "
                f"[{v['ci_vs_guided'][0]:+.4f},{v['ci_vs_guided'][1]:+.4f}] | {v['delta_vs_argmax']:+.4f} | "
                f"{v['fired']} | {v['win']} | {v['loss']} |")

    md = [f"# Final results — {args.tag}", "",
          f"n = {out['n']} expressions · oracle mIoU {out['oracle_mIoU']:.4f} · "
          f"spatial expressions {out['spatial_expr_frac']:.3f}", "", "## Main table", "",
          "| row | mIoU | oIoU | Δ vs released [95% CI] | Δ vs argmax | changed | win | loss |",
          "|---|---|---|---|---|---|---|---|"]
    for nm in ("argmax_g2l", "hybridgl_guided", "parent_strict", "ours", "ours_snap"):
        md.append(fmt(nm, out["all"]["rows"]))
    md += ["", "## Sensitivity", "",
           "| row | mIoU | oIoU | Δ vs released [95% CI] | Δ vs argmax | changed | win | loss |",
           "|---|---|---|---|---|---|---|---|"]
    for nm in ("ours_k5", "ours_k12", "ours_k16", "ours_k24", "ours_p34", "ours_p66",
               "ours_rank_sem", "ours_rank_sc", "ours_set_full", "ours_set_strip", "ours_set_noun"):
        md.append(fmt(nm, out["all"]["rows"]))
    md += ["", "## Controls", "",
           "| row | mIoU | oIoU | Δ vs released [95% CI] | Δ vs argmax | changed | win | loss |",
           "|---|---|---|---|---|---|---|---|"]
    for nm in ("ctrl_geom_extreme", "ctrl_no_side", "ctrl_swap", "ctrl_allpool"):
        md.append(fmt(nm, out["all"]["rows"]))
    o = out["all"]["rows"]["ours"]
    md += ["", "## Ours — strata", "",
           f"- part expressions (n={o['part_n']}): Δ {o['part_delta']:+.4f}",
           f"- expressions with no spatial relation: Δ {o['no_relation_delta']:+.4f}", ""]
    for k, v in o["strata"].items():
        md.append(f"- relation '{k}' (n={o['strata_n'][k]}): Δ {v:+.4f}")
    md += ["", "### by GT-area quartile"]
    for i, v in enumerate(o["quartile_delta"]):
        md.append(f"- q{i+1}: {v:+.4f}")
    md += ["", f"oracle-hit rate: ours {o['oracle_hit']:.3f} vs released "
               f"{out['all']['rows']['hybridgl_guided']['oracle_hit']:.3f}", "",
           "## Held-out complement", "",
           "| row | mIoU | oIoU | Δ vs released [95% CI] | Δ vs argmax | changed | win | loss |",
           "|---|---|---|---|---|---|---|---|"]
    for nm in ("argmax_g2l", "hybridgl_guided", "parent_strict", "ours", "ours_snap"):
        md.append(fmt(nm, out["heldout"]["rows"]))
    (args.output / f"final_{args.tag}.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
