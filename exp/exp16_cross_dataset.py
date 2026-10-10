#!/usr/bin/env python
"""EXP-16 — Cross-dataset dose-response.

The three RefCOCO-family splits differ sharply in how often an expression
carries a spatial relation, for construction reasons: RefCOCO+ was collected
with location words excluded. If the operator's benefit is caused by resolving
spatial references, its per-dataset effect should track that frequency.

This script reports, per dataset: the taxonomy of the oracle gap, the
spatial-expression frequency, how often the operator fires, the overall delta,
and the delta *on the spatial stratum only*. The last column is the informative
one — a dataset where the operator is neutral overall but positive on its own
target stratum is consistent with the mechanism; neutral even there would not be.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, taxonomy, reltype, PART_WORDS  # noqa: E402
from method import Selector  # noqa: E402

SETS = [("refcoco", "guided_val"), ("refcocog", "guided_refcocog"), ("refcoco+", "guided_refcocoplus")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    args = ap.parse_args()
    own = Selector()
    out = {}
    rows = []
    for name, tag in SETS:
        rp = Path("exp/out_exp4") / f"records_{tag}.jsonl"
        pp = Path("exp/out_exp4") / f"pairs_{tag}.npz"
        if not (rp.exists() and pp.exists()):
            print(f"skip {name}")
            continue
        recs = [Rec(json.loads(l)) for l in rp.open()]
        pairs = np.load(pp)
        for r in recs:
            r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
        tax = taxonomy(recs, lambda r: r.final)
        rt = np.array([reltype(r.rel) for r in recs])
        sp = rt != "none"
        ref = np.array([r.gt[r.final] for r in recs])
        ours = np.array([r.gt[own(r)] for r in recs])
        snap = np.array([r.gt[int(r.ps[r.final])] for r in recs])
        fired = np.array([own(r) != r.final for r in recs])
        out[name] = dict(
            n=len(recs), spatial_frac=float(sp.mean()),
            fired_frac=float(fired.mean()), fired_n=int(fired.sum()),
            released_mIoU=float(ref.mean()), ours_mIoU=float(ours.mean()),
            delta_overall=float((ours - ref).mean()),
            delta_spatial=float((ours - ref)[sp].mean()) if sp.sum() else None,
            delta_nonspatial=float((ours - ref)[~sp].mean()),
            delta_on_fired=float((ours - ref)[fired].mean()) if fired.sum() else None,
            snap_delta=float((snap - ref).mean()),
            wrong_instance_gap_share=tax["by_class"]["wrong-instance"]["gap_share"],
            under_seg_gap_share=tax["by_class"]["under-segmentation"]["gap_share"],
            oracle_mIoU=tax["oracle_mIoU"])
        rows.append(name)
    with (args.output / "cross_dataset.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    hdr = ["dataset", "n", "spatial%", "fired%", "released", "ours", "Δ all", "Δ spatial", "Δ on fired",
           "Δ snap", "wrong-inst", "under-seg"]
    print(" | ".join(hdr))
    for name in rows:
        v = out[name]
        print(" | ".join([
            name, f"{v['n']}", f"{v['spatial_frac']*100:.1f}", f"{v['fired_frac']*100:.1f}",
            f"{v['released_mIoU']:.4f}", f"{v['ours_mIoU']:.4f}", f"{v['delta_overall']:+.4f}",
            f"{v['delta_spatial']:+.4f}" if v["delta_spatial"] is not None else "-",
            f"{v['delta_on_fired']:+.4f}" if v["delta_on_fired"] is not None else "-",
            f"{v['snap_delta']:+.4f}",
            f"{v['wrong_instance_gap_share']:.3f}", f"{v['under_seg_gap_share']:.3f}"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
