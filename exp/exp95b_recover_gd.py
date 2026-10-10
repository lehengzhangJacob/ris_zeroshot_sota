#!/usr/bin/env python
"""EXP-95b — recover the hybrid study's winning GBM depth (deterministic).

The first exp95 run stored best_params without the suggested `gd`, and the
rerun (patched to store it) died partway.  The pipeline is deterministic
(seed 0, fixed splits, fixed sub-models), so the winning trial's depth is
recoverable exactly: rebuild the identical setup, evaluate gd in {2,3,4} at
the stored best (kind=gbm, p_inc=0.6311620664353685), and match the objective
value 0.6190364770348084 (and calib r 0.5948154752567522).  Writes the
recovered gd back into optuna_study_gref10.json.

    python exp/exp95b_recover_gd.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import build_candidates, precompute_row, topk_mask  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402
from exp95_hybrid_study import cand_mask, hybrid_candidates  # noqa: E402

OUTF = Path("exp/out_final")
TARGET_VALUE = 0.6190364770348084
TARGET_R = 0.5948154752567522
P_INC = 0.6311620664353685


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
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
    for r in tr:
        rec = recs_by_rid.get(r["rid"])
        if rec is not None and not rec.get("terms"):
            rec = None
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        H, specs = hybrid_candidates(r, rec, F, ids, g9, pick_clf if rec else None)
        r["_H"], r["_specs"] = H, specs
        y = np.zeros(len(specs), np.float32)
        if specs and not r["no_target"]:
            gt = r["gt"]
            for c, sp in enumerate(specs):
                m = cand_mask(r, sp)
                I = float(np.logical_and(m, gt).sum())
                U = float(np.logical_or(m, gt).sum())
                y[c] = 1.0 if (U > 0 and I / U >= 0.5) else 0.0
        r["_Hy"] = y
    rng = np.random.RandomState(0)
    refs = sorted({r["rid"] for r in tr})
    rng.shuffle(refs)
    n_fit = int(0.7 * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in tr if r["rid"] in fit_refs]
    calib_rows = [r for r in tr if r["rid"] not in fit_refs]
    print(f"setup done ({time.time()-t0:.0f}s); recovering gd...", flush=True)

    def eval_hybrid(rs, clf, p_inc):
        tI = tU = 0.0
        for r in rs:
            if r["_H"] is None:
                continue
            proba = clf.predict_proba(r["_H"])[:, 1]
            c = int(np.argmax(proba))
            if proba[c] < p_inc:
                continue
            pred = cand_mask(r, r["_specs"][c])
            tI += float(np.logical_and(pred, r["gt"]).sum())
            tU += float(np.logical_or(pred, r["gt"]).sum())
        return tI / max(1.0, tU)

    ship_term = None
    results = {}
    for gd in (2, 3, 4):
        clf = make_clf("gbm", dict(n_estimators=150, max_depth=gd, learning_rate=0.1))
        clf.fit(np.concatenate([r["_H"] for r in fit_rows if r["_H"] is not None]),
                np.concatenate([r["_Hy"] for r in fit_rows if r["_H"] is not None]))
        rc = eval_hybrid(calib_rows, clf, P_INC)
        anchor = 40.96 * (1 + (rc - 0.5529))
        value = min(anchor / 68.94, 71.11 / 74.34)
        results[gd] = (rc, value)
        print(f"gd={gd}: calib r {rc:.10f}  value {value:.10f}", flush=True)

    best_gd = min(results, key=lambda g: abs(results[g][1] - TARGET_VALUE))
    rc, value = results[best_gd]
    ok = abs(value - TARGET_VALUE) < 1e-9 and abs(rc - TARGET_R) < 1e-9
    print(f"recovered gd={best_gd} (value {value:.12f} vs target {TARGET_VALUE:.12f}; "
          f"exact match: {ok})", flush=True)
    if ok:
        p = OUTF / "optuna_study_gref10.json"
        d = json.loads(p.read_text())
        d["best_params"]["gd"] = best_gd
        d["gd_recovery"] = ("depth recovered exactly from the deterministic pipeline "
                            "(exp95b); value and calib r match to 1e-9")
        p.write_text(json.dumps(d, indent=2))
        print("artifact updated with gd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
