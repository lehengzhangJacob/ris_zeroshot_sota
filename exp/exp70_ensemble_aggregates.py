#!/usr/bin/env python
"""EXP-70 — exact wall aggregates at the shipped (ensemble) operating point.

The wall's frontier formula is

    cIoU = r * S_U / (S_U + (1-N_acc) * FP_U)

with S_U the union mass of predicted targeted samples, FP_U the union mass
added by predicted no-target samples, and r the realized cumulative targeted
IoU.  r_required = 0.6894 * (1 + (FP_U/S_U)*(1-N_acc)) depends on the
operating point, so when the shipped rule changes (single probe -> 3-probe
ensemble, exp69) the requirement must be re-measured, not carried over.

This script measures the aggregates for the shipped rule and the ensemble rule
on identical rows (same build as exp67-69) and reports both floors' exact
requirements at each point.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO, build_rows  # noqa: E402
from exp62_clause_selector import load_record, make_clf  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
FLOOR_CIOU, FLOOR_GIOU = 0.6894, 0.7434
VAL_PREV_NT = 0.6256


def measure(rows, abstain_fn, p_inc):
    """Official metrics plus the union-mass aggregates behind the frontier."""
    S_U = FP_U = 0.0            # predicted targeted / no-target union mass
    I_tot = 0.0                 # intersection over predicted targeted
    acc_iou = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        abstain = abstain_fn(r)
        if not abstain and not np.flatnonzero(r["scores"] >= p_inc).size:
            abstain = True
        if abstain:
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
            continue
        sel = np.flatnonzero(r["scores"] >= p_inc)
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn += 1
            FP_U += U
        else:
            tn += 1
            I_tot += I
            S_U += U
            acc_iou += (I / U if U else 0.0)
    n = len(rows)
    N = tp / (tp + fn) if (tp + fn) else 0.0
    T = tn / (tn + fp) if (tn + fp) else 0.0
    # mean targeted IoU over predicted targeted (second pass keeps pass 1 simple)
    mt = 0.0
    num_t = 0
    for r in rows:
        abstain = abstain_fn(r)
        if not abstain and not np.flatnonzero(r["scores"] >= p_inc).size:
            abstain = True
        if abstain or r["no_target"]:
            continue
        sel = np.flatnonzero(r["scores"] >= p_inc)
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        mt += (I / U if U else 0.0)
        num_t += 1
    m = mt / num_t if num_t else 0.0
    r_del = I_tot / S_U if S_U else 0.0
    fpu_su = FP_U / S_U if S_U else 0.0
    r_req = FLOOR_CIOU * (1 + fpu_su * (1 - N))
    m_req = (FLOOR_GIOU - VAL_PREV_NT * N) / ((1 - VAL_PREV_NT) * T) if T else None
    return dict(
        n=n, N_acc=round(100 * N, 2), T_acc=round(100 * T, 2),
        cIoU=round(100 * I_tot / (S_U + FP_U), 2) if (S_U + FP_U) else 0.0,
        gIoU=round(100 * acc_iou / n, 2),
        S_U=round(S_U, 1), FP_U=round(FP_U, 1), FP_U_over_S_U=round(fpu_su, 4),
        r_delivered=round(r_del, 4),
        r_required_for_floor=round(r_req, 4),
        r_multiple_needed=round(r_req / r_del, 2) if r_del else None,
        m_delivered=round(100 * m, 2),
        m_required_for_floor=None if m_req is None else round(100 * m_req, 2),
        targeted_predicted=num_t,
    )


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    params = (dict(C=st["C"]) if st["clf"] == "logreg" else
              dict(n_estimators=st["n_estimators"], max_depth=st["max_depth"],
                   learning_rate=st["learning_rate"]))
    p_inc, p_abs = st["p_inc"], st["p_abs"]
    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    feats = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        r = load_record(json.loads(line), gd_tr, by_tr, {}, {})
        if r is not None:
            feats.append(r)
    clf = make_clf(st["clf"], params)
    clf.fit(np.concatenate([r["X"] for r in feats]),
            np.concatenate([r["y"] for r in feats]))
    print(f"classifier fit ({time.time()-t0:.0f}s)", flush=True)

    rules = {
        "shipped_single_probe": (lambda r: (r.get("exists_no") is True) or r["vlm_none"]
                                 or r["max_p"] < p_abs),
        "shipped_ensemble": (lambda r: any(r["ens"])),
    }
    out = dict(p_inc=p_inc, p_abs=p_abs, rules={})
    for split, recs in (("train", "exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                        ("val", "exp/out_extreme/records_grefcoco.jsonl")):
        rows = build_rows(Path(recs), split, clf, params)
        print(f"{split}: {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
        out["rules"].setdefault("shipped_single_probe", {})[split] = measure(
            rows, rules["shipped_single_probe"], p_inc)
        out["rules"].setdefault("shipped_ensemble", {})[split] = measure(
            rows, rules["shipped_ensemble"], p_inc)
        for nm in rules:
            mm = out["rules"][nm][split]
            print(f"  {split:5s} {nm:22s} cIoU={mm['cIoU']:6.2f} gIoU={mm['gIoU']:6.2f} "
                  f"N={mm['N_acc']:5.2f} T={mm['T_acc']:5.2f} r_del={mm['r_delivered']:.3f} "
                  f"r_req={mm['r_required_for_floor']:.3f} (x{mm['r_multiple_needed']}) "
                  f"m_del={mm['m_delivered']:5.2f} m_req={mm['m_required_for_floor']}",
                  flush=True)
    out["generated"] = "exp70_ensemble_aggregates.py"
    (OUTF / "gref_ensemble_aggregates.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_ensemble_aggregates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
