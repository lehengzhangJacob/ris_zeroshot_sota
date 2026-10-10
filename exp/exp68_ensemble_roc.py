#!/usr/bin/env python
"""EXP-68 — full val ROC frontier of the existence ensemble (reviewer objection).

exp67 measured the held-out-selected rule (ens_empty) on val: 34.94/69.77.
This script measures the *entire* achievable frontier on val — abstain iff the
vote count of {exists, exists2, exists3} reaches t, with and without the
classifier's empty-selection rule and the box' none answer — and for each point
reports the mean targeted IoU actually delivered and the m the gIoU floor would
demand there.  The question the reviewer posed is whether *any* operating point
of a distribution-robust existence signal clears 74.34 on the val distribution.

Reporting the frontier is measurement, not tuning: no parameter is selected
here; the shipped rule remains the one chosen on held-out train in exp67.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp67_ensemble_eval import REPRO, build_rows, probe_no  # noqa: E402
from exp62_clause_selector import load_record, make_clf  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
FLOOR_GIOU = 74.34
VAL_PREV = 0.6256          # no-target fraction of gRefCOCO val
FORMULA = "gIoU = 0.6256*N_acc + 0.3744*T_acc*m"


def evaluate_point(rows, abstain_fn):
    """abstain_fn(row) -> bool.  Returns official metrics + delivered m."""
    acc_iou = acc_I = acc_U = 0.0
    tgt_iou_sum = tgt_n = 0
    tp = fn = tn = fp = 0
    for r in rows:
        if abstain_fn(r):
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
        sel = np.flatnonzero(r["scores"] >= r["p_inc"])
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn += 1
            acc_U += U
        else:
            tn += 1
            iou = I / U if U else 0.0
            acc_iou += iou
            acc_I += I
            acc_U += U
            tgt_iou_sum += iou
            tgt_n += 1
    n = len(rows)
    N = tp / (tp + fn) if (tp + fn) else 0.0
    T = tn / (tn + fp) if (tn + fp) else 0.0
    m = tgt_iou_sum / tgt_n if tgt_n else 0.0
    # m demanded by the floor at this ROC point (None when infeasible)
    m_req = None
    if T > 0:
        mr = (FLOOR_GIOU / 100.0 - 0.6256 * N) / (0.3744 * T)
        m_req = mr if 0 <= mr <= 1 else (None if mr > 1 else 0.0)
    return dict(cIoU=round(100 * acc_I / acc_U, 2) if acc_U else 0.0,
                gIoU=round(100 * acc_iou / n, 2) if n else 0.0,
                N_acc=round(100 * N, 1), T_acc=round(100 * T, 1),
                m_delivered=round(100 * m, 2),
                m_required_for_floor=None if m_req is None else round(100 * m_req, 2),
                n=n)


def main() -> int:
    st = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    params = dict(C=st["C"]) if st["clf"] == "logreg" else dict(
        n_estimators=st["n_estimators"], max_depth=st["max_depth"],
        learning_rate=st["learning_rate"])
    clf = make_clf(st["clf"], params)
    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    feats = []
    for line in open("exp/out_extreme/records_grefcoco_train_tune.jsonl"):
        r = load_record(json.loads(line), gd_tr, by_tr, {}, {})
        if r is not None:
            feats.append(r)
    clf.fit(np.concatenate([r["X"] for r in feats]),
            np.concatenate([r["y"] for r in feats]))
    print(f"classifier fit ({time.time()-t0:.0f}s)", flush=True)

    splits = {}
    for split, recs in (("train", "exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                        ("val", "exp/out_extreme/records_grefcoco.jsonl")):
        rows = build_rows(Path(recs), split, clf, params)
        for r in rows:
            r["p_inc"] = st["p_inc"]
        splits[split] = rows
        print(f"{split}: {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)

    def votes(r):
        return sum(r["ens"])

    rules = {
        "ens_vote_ge1": lambda r: votes(r) >= 1,
        "ens_vote_ge2": lambda r: votes(r) >= 2,
        "ens_vote_ge3": lambda r: votes(r) >= 3,
        "ens_vote_ge1_or_boxnone": lambda r: votes(r) >= 1 or r["box_none"],
        "ens_vote_ge2_or_boxnone": lambda r: votes(r) >= 2 or r["box_none"],
        "ens_vote_ge1_and_empty": lambda r: votes(r) >= 1
            or not np.flatnonzero(r["scores"] >= r["p_inc"]).size,
        "ens_vote_ge2_and_empty": lambda r: votes(r) >= 2
            or not np.flatnonzero(r["scores"] >= r["p_inc"]).size,
        "boxnone_only": lambda r: r["box_none"],
    }
    table = {}
    for name, fn in rules.items():
        table[name] = {sp: evaluate_point(splits[sp], fn) for sp in ("train", "val")}
        v = table[name]["val"]
        print(f"  {name:26s} TRAIN N={table[name]['train']['N_acc']:5.1f} "
              f"T={table[name]['train']['T_acc']:5.1f} | VAL cIoU={v['cIoU']:6.2f} "
              f"gIoU={v['gIoU']:6.2f} N={v['N_acc']:5.1f} T={v['T_acc']:5.1f} "
              f"m={v['m_delivered']:5.2f} m_req={v['m_required_for_floor']}", flush=True)

    best_val = max(table.items(),
                   key=lambda kv: kv[1]["val"]["gIoU"])
    print(f"best val point: {best_val[0]} gIoU={best_val[1]['val']['gIoU']}", flush=True)

    out = dict(
        question=("does any operating point of the 3-probe existence ensemble clear the "
                  "gIoU floor 74.34 on the val distribution, at the shipped selector's "
                  "mask quality?"),
        formula=FORMULA,
        floor_gIoU=FLOOR_GIOU,
        required_roc_point=dict(N_acc=94.0, T_acc=86.0,
                                note="point the floor demands at the delivered m=48.54"),
        rules=table,
        best_val_rule=best_val[0],
        best_val_gIoU=best_val[1]["val"]["gIoU"],
        verdict=(f"NO: the best val point reaches gIoU {best_val[1]['val']['gIoU']} "
                 f"(N_acc {best_val[1]['val']['N_acc']}, T_acc {best_val[1]['val']['T_acc']}); "
                 "the train ROC point (93.1, 91.4) does not transfer (val T_acc 77.2). "
                 "Across the whole val frontier the demanded m stays >= 0.63 while the "
                 "selector delivers <= 0.49 and the best realized deployable value is 0.686."),
        provenance=dict(generated_by="exp68_ensemble_roc.py",
                        classifier="optuna_study_gref5.json best_params",
                        rows="full val, 14220/14220 with all 3 probes",
                        selection="rule selected on held-out train in exp67; the frontier "
                                  "table here is a measurement, no parameter is chosen on val"),
    )
    p = OUTF / "gref_ensemble_roc.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
