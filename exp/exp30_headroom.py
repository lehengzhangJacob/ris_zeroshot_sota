#!/usr/bin/env python
"""EXP-30 — Structural-cause numbers, fresh, from this run's records.

Computes the evidence behind the paper's structural claims on RefCOCO val:

  * fraction of oracle masks with IoU >= 0.5 (the pool is not the bottleneck);
  * on wrong-instance failures: fraction of expressions where the oracle is
    inside the released top-3 (the guidance's confinement), the mean oracle
    score-rank, and the fraction where it is in the category score's top-8;
  * the fraction of wrong-instance expressions that carry a spatial word;
  * the mIoU of a perfect chooser restricted to the released top-3 (ceiling)
    and the share of the released gain over argmax that the top-3 ceiling
    implies.

Writes exp/out_final/headroom.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, classify, parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPATIAL = {"left", "right", "leftmost", "rightmost", "top", "bottom", "upper", "lower",
           "above", "below", "middle", "center", "centre", "mid", "central", "front",
           "back", "behind", "next", "near", "beside", "between", "under", "over",
           "side", "corner", "topmost", "l", "first", "second", "third", "fourth", "fifth"}


def main() -> int:
    recs = [Rec(json.loads(l)) for l in (ROOT / "exp/out_extreme/records_refcoco.jsonl").open()]
    pairs = np.load(ROOT / "exp/out_extreme/pairs_refcoco.npz")
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    n = len(recs)
    print(f"{n} records")

    oracle_ge50 = np.array([r.gt[r.orc] >= 0.5 for r in recs])

    # classify the released selection
    cls = []
    for r in recs:
        i = r.final
        inter = r.pc[i, r.orc] * (r.area[i] + r.area[r.orc]) / (1 + r.pc[i, r.orc])
        sub_am = (inter / max(r.area[i], 1e-9)) >= 0.95 and r.area[i] < r.area[r.orc]
        sub_orc = (inter / max(r.area[r.orc], 1e-9)) >= 0.95 and r.area[r.orc] < r.area[i]
        cls.append(classify(r.gt[i], r.gt[r.orc], sub_am, sub_orc, r.pc[i, r.orc]))
    cls = np.array(cls)
    wi = cls == "wrong-instance"
    print(f"wrong-instance failures: {int(wi.sum())} ({wi.mean()*100:.1f}%)")

    # released top-3 and category top-8 containment
    in_top3 = np.array([r.orc in set(r.top.tolist()) for r in recs])
    K8 = [np.argsort(-r.S["sc_content"], kind="stable")[:8] for r in recs]
    in_cat8 = np.array([recs[i].orc in set(K8[i].tolist()) for i in range(n)])
    rank = np.array([int(np.flatnonzero(np.argsort(-r.S["sc"], kind="stable") == r.orc)[0]) + 1
                     for r in recs])
    spatial_word = np.array([bool(set(w.strip(".,!?").lower() for w in r.sent.split()) & SPATIAL)
                             for r in recs])
    parser_spatial = np.array([r.rel is not None for r in recs])

    # top-3 perfect chooser ceiling: per expression, best gt among top-3
    top3_iou = np.array([max(r.gt[t] for t in r.top) for r in recs])
    top3_ceiling = float(np.mean(top3_iou))
    rel_miou = float(np.mean([r.gt[r.final] for r in recs]))
    arg_miou = float(np.mean([r.gt[r.pure] for r in recs]))
    ceiling_share = (top3_ceiling - arg_miou) / (rel_miou - arg_miou) if rel_miou != arg_miou else None

    out = dict(
        n=n,
        oracle_ge50_frac=float(oracle_ge50.mean()),
        oracle_mIoU=float(np.mean([r.gt[r.orc] for r in recs])),
        wrong_instance_n=int(wi.sum()),
        wrong_instance_prev=float(wi.mean()),
        wrong_instance_mean_iou_when_wrong=float(np.mean([recs[i].gt[recs[i].final] for i in range(n) if wi[i]])),
        wi_spatial_word_frac=float(spatial_word[wi].mean()) if wi.sum() else None,
        spatial_word_frac_overall=float(spatial_word.mean()),
        wi_parser_spatial_frac=float(parser_spatial[wi].mean()) if wi.sum() else None,
        parser_spatial_frac_overall=float(parser_spatial.mean()),
        oracle_in_top3_overall=float(in_top3.mean()),
        oracle_in_top3_on_wi=float(in_top3[wi].mean()) if wi.sum() else None,
        oracle_rank_mean_on_wi=float(rank[wi].mean()) if wi.sum() else None,
        oracle_rank_median_on_wi=float(np.median(rank[wi])) if wi.sum() else None,
        oracle_in_cat_top8_on_wi=float(in_cat8[wi].mean()) if wi.sum() else None,
        oracle_in_cat_top8_overall=float(in_cat8.mean()),
        oracle_mean_on_wi=float(np.mean([recs[i].gt[recs[i].orc] for i in range(n) if wi[i]])) if wi.sum() else None,
        top3_ceiling_mIoU=top3_ceiling,
        released_mIoU=rel_miou,
        argmax_mIoU=arg_miou,
        top3_ceiling_share_of_released_gain=float(ceiling_share) if ceiling_share else None,
        generated="this run, exp/out_extreme/records_refcoco.jsonl + pairs_refcoco.npz",
    )
    (ROOT / "exp/out_final/headroom.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
