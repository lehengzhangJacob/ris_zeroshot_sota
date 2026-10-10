#!/usr/bin/env python
"""EXP-123 — FP_U shrink: singles-only fallback on predicted-no-target records.

The cIoU identity for the shipped rule: cIoU = I / (S_U + FP_U + A_U) with
FP_U = the predicted area of the 1,230 val no-target records that the gate
predicted target (10.49M px of pure dilution, zero intersection).  Those
records use the pair/union FALLBACK, whose argmax candidate may be a PAIR of
masks; restricting the fallback to singles shrinks FP_U with zero effect on
the targeted side (the shipped rule uses the fallback for no targeted record:
fallback_used == fn on val).  gIoU and N_acc are untouched by construction.

Measured here on the held-out train no-target rows (the same fallback, same
features): mean predicted area per record under (a) the shipped fallback and
(b) singles-only; the ratio projects the val FP_U drop.

    python exp/exp123_fp_shrink.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"pair clf fitted ({time.time()-t0:.0f}s)", flush=True)

    nt = [r for r in tr if r["no_target"]]
    areas_all, areas_single, npairs = [], [], 0
    for r in nt:
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        k_all = int(np.argmax(pp))
        m_all = candidate_mask(r["masks"], ids[k_all])
        areas_all.append(float(m_all.sum()))
        if ids[k_all][1] >= 0:
            npairs += 1
        single = [k for k, c in enumerate(ids) if c[1] < 0]
        if single:
            k_s = max(single, key=lambda k: pp[k])
            m_s = candidate_mask(r["masks"], ids[k_s])
            areas_single.append(float(m_s.sum()))
        else:
            areas_single.append(float(m_all.sum()))
    a_all = float(np.mean(areas_all))
    a_single = float(np.mean(areas_single))
    ratio = a_single / a_all if a_all else 1.0
    fp_val = json.load(open(OUTF / "gref_joint_full_val.json"))["metrics"]["masses"]["FP_U"]
    proj_fp = fp_val * ratio
    S_U = 111846621.0  # v14 union mass
    A_U = 10016077.79180994
    I = 65627497.0
    ciou_now = 100.0 * I / (S_U + fp_val + A_U)
    ciou_new = 100.0 * I / (S_U + proj_fp + A_U)
    out = dict(n_no_target_rows=len(nt), pair_share_of_argmax=100.0 * npairs / max(1, len(nt)),
               mean_pred_area_shipped=a_all, mean_pred_area_singles=a_single,
               ratio=ratio, fp_val=fp_val, projected_fp=proj_fp,
               cIoU_now=ciou_now, cIoU_projected=ciou_new, gain=ciou_new - ciou_now,
               note=("singles-only fallback affects ONLY predicted-no-target records; targeted "
                     "records never use the fallback in the shipped rule (fallback_used==fn on val)"),
               generated="exp123_fp_shrink.py")
    (OUTF / "gref_fp_shrink.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
