#!/usr/bin/env python
"""EXP-13 — Does the diagnostic predict where the operator helps?

The operator is worth +5.5 mIoU on RefCOCO val and neutral on RefCOCOg. This
script tests whether the diagnostic introduced in EXP-1/EXP-7 *predicts* that
outcome from measurements available before running the selector:

  * the share of expressions carrying a spatial relation
  * the share of expressions whose referent lies inside the released
    candidate set (top-3 by the full expression score) — the headroom the
    operator can act on
  * the share of the oracle gap mass carried by the wrong-instance class

If the operator's benefit tracks these, the scope boundary is a prediction of
the analysis rather than an unexplained failure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, reltype  # noqa: E402


def stats(path, pairs_path):
    recs = [Rec(json.loads(l)) for l in Path(path).open()]
    pairs = np.load(pairs_path)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    n = len(recs)
    rel = np.array([reltype(r.rel) for r in recs])
    in_top3 = np.array([r.orc in set(r.top.tolist()) for r in recs])
    cat_top8 = np.array([r.orc in set(np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist())
                         for r in recs])
    # wrong-instance share of the oracle gap for the released selection
    gap = np.array([r.gt[r.orc] - r.gt[r.final] for r in recs])
    wi = np.array([r.gt[r.final] < 0.1 and r.gt[r.orc] > 0.6 for r in recs])
    union = np.array([r.union[r.final] for r in recs])
    iou = np.array([r.gt[r.final] for r in recs])
    return dict(
        n=n,
        spatial_frac=float((rel != "none").mean()),
        horizontal_frac=float((rel == "horizontal").mean()),
        oracle_in_released_top3=float(in_top3.mean()),
        oracle_in_cat_top8=float(cat_top8.mean()),
        # headroom only over the spatially-constrained subset, where the
        # operator is allowed to act
        spatial_oracle_in_top3=float(in_top3[rel != "none"].mean()) if (rel != "none").any() else None,
        wrong_instance_frac=float(wi.mean()),
        wrong_instance_gap_share=float(gap[wi].sum() / gap.sum()) if gap.sum() else None,
        released_mIoU=float(iou.mean()),
        oracle_mIoU=float(np.mean([r.gt[r.orc] for r in recs])),
        top3_ceiling_mIoU=float(np.mean([max(r.gt[i] for i in r.top) for r in recs])),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("exp/out_final"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = {
        "refcoco": stats("exp/out_exp4/records_guided_val.jsonl",
                         "exp/out_exp4/pairs_guided_val.npz"),
        "refcocog": stats("exp/out_exp4/records_guided_refcocog.jsonl",
                          "exp/out_exp4/pairs_guided_refcocog.npz"),
    }
    with (args.output / "scope.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    keys = list(next(iter(out.values())).keys())
    print(f"{'metric':<28}{'refcoco':>12}{'refcocog':>12}")
    for k in keys:
        a, b = out["refcoco"][k], out["refcocog"][k]
        fa = f"{a:.4f}" if isinstance(a, float) else str(a)
        fb = f"{b:.4f}" if isinstance(b, float) else str(b)
        print(f"{k:<28}{fa:>12}{fb:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
