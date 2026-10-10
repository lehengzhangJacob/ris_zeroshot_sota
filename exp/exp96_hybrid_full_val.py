#!/usr/bin/env python
"""EXP-96 — frozen hybrid selector: one full-val pass (change 11).

exp95's 300-trial study selected the hybrid (GBM depth 4, p_inc 0.631):
calib r 0.5948 vs termbox 0.5529 (+0.0419) and pair 0.5177 (+0.0771).
Candidates = pair/union candidates (top-6 singles+pairs) + per-term picks +
termbox-union, one GBM over the union of both feature blocks; abstention
unchanged (pair gate + 3-probe ensemble); gate below p_inc abstains.

Refit on ALL tune rows, frozen, one full-val pass.  No val number influences
any choice.

    python exp/exp96_hybrid_full_val.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402
from exp95_hybrid_study import cand_mask, hybrid_candidates  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    g10 = json.loads((OUTF / "optuna_study_gref10.json").read_text())
    bp = g10["best_params"]
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    print(f"frozen hybrid: {bp}", flush=True)

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}

    # sub-models
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r, rec):
            Xf.append(F); yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))

    # hybrid selector fit on ALL tune rows
    Xh, yh = [], []
    for r in tr:
        rec = recs_by_rid.get(r["rid"])
        if rec is not None and not rec.get("terms"):
            rec = None
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        H, specs = hybrid_candidates(r, rec, F, ids, g9, pick_clf if rec else None)
        if H is None:
            continue
        if not r["no_target"]:
            y = np.zeros(len(specs), np.float32)
            gt = r["gt"]
            for c, sp in enumerate(specs):
                m = cand_mask(r, sp)
                I = float(np.logical_and(m, gt).sum())
                U = float(np.logical_or(m, gt).sum())
                y[c] = 1.0 if (U > 0 and I / U >= 0.5) else 0.0
        else:
            y = np.zeros(len(specs), np.float32)
        Xh.append(H); yh.append(y)
    hybrid_clf = make_clf("gbm", dict(n_estimators=150, max_depth=bp["gd"],
                                      learning_rate=0.1))
    hybrid_clf.fit(np.concatenate(Xh), np.concatenate(yh))
    print(f"hybrid fit on {len(Xh)} rows ({time.time()-t0:.0f}s)", flush=True)

    val = load_rows(Path("exp/out_extreme/records_grefcoco.jsonl"), "val", clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    p_sel = g8["p_sel"]
    p_inc = bp["p_inc"]
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    per_row = []
    n_hyb = 0
    for r in val:
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        pred = None
        if not abstain:
            rec = recs_by_rid.get(r["rid"])
            if rec is not None and not rec.get("terms"):
                rec = None
            H, specs = hybrid_candidates(r, rec, F, ids, g9, pick_clf if rec else None)
            if H is not None:
                proba = hybrid_clf.predict_proba(H)[:, 1]
                c = int(np.argmax(proba))
                if proba[c] >= p_inc:
                    pred = cand_mask(r, specs[c])
                    n_hyb += 1
        if pred is None:
            # abstained by the pair gate/ensemble, or gated below p_inc
            # (the study's rule: sub-threshold rows contribute nothing to r,
            # i.e. they abstain -- exact match)
            per_row.append((r["no_target"], True, 0.0, 0.0, r["gt_area"]))
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        per_row.append((r["no_target"], False, I, U, r["gt_area"]))
        if r["no_target"]:
            fn += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
    n = len(val)
    m = dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
             gIoU=100.0 * acc_iou / n,
             N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
             T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
             TP=tp, FN=fn, TN=tn, FP=fp, hybrid_used=n_hyb)
    print(f"FULL VAL hybrid: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} "
          f"N {m['N_acc']:.1f} T {m['T_acc']:.1f} (hybrid {n_hyb}) ({time.time()-t0:.0f}s)",
          flush=True)
    out = dict(model=dict(kind="hybrid: pair candidates + per-term picks + termbox-union",
                          study_params=bp, source_study="optuna_study_gref10.json"),
               metrics=m,
               history={"termbox": {"cIoU": 40.9593, "gIoU": 71.1064},
                        "pair_union": {"cIoU": 36.7453, "gIoU": 69.6279},
                        "ensemble": {"cIoU": 34.9419, "gIoU": 69.7711},
                        "single_probe": {"cIoU": 35.0884, "gIoU": 68.3729}},
               generated="exp96_hybrid_full_val.py")
    (OUTF / "gref_hybrid_full_val.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_hybrid_full_val.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
