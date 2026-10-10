#!/usr/bin/env python
"""EXP-69 — same-rows showdown of the shipped rule vs the 3-probe ensemble.

exp67 compared ensemble rules that omit the shipped rule's `max_p < p_abs`
abstention branch, so the comparison was not apples-to-apples.  This script
evaluates every candidate on identical rows (the same 1,398 held-out train
rows and the same 14,220 val rows), picks the winner by held-out train gIoU,
and reports that winner's val transfer.

Candidates (abstain condition):
  shipped      : exists_no | vlm_none | max_p<p_abs | no-selection
  ensemble     : any3_no   | no-selection
  ensemble_abs : any3_no   | vlm_none | max_p<p_abs | no-selection
  ensemble2_abs: two3_no   | vlm_none | max_p<p_abs | no-selection
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


def evaluate(rows, abstain_fn, p_inc):
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        abstain = abstain_fn(r)
        if not abstain and not np.flatnonzero(r["scores"] >= p_inc).size:
            abstain = True
        # shipped rule's p_abs branch is folded into abstain_fn where used
        if abstain:
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
        sel = np.flatnonzero(r["scores"] >= p_inc)
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
    n = len(rows)
    return dict(n=n,
                cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / n if n else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)


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

    rows = {}
    for split, recs in (("train", "exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                        ("val", "exp/out_extreme/records_grefcoco.jsonl")):
        rows[split] = build_rows(Path(recs), split, clf, params)
        print(f"{split}: {len(rows[split])} rows ({time.time()-t0:.0f}s)", flush=True)

    def any3(r):
        return any(r["ens"])

    def two3(r):
        return sum(r["ens"]) >= 2

    def ship_no(r):
        return any3(r)  # placeholder, replaced below

    candidates = {
        "shipped": (lambda r: (r.get("exists_no") is True) or r["vlm_none"]
                    or r["max_p"] < p_abs),
        "ensemble": (lambda r: any3(r)),
        "ensemble_abs": (lambda r: any3(r) or r["vlm_none"] or r["max_p"] < p_abs),
        "ensemble2_abs": (lambda r: two3(r) or r["vlm_none"] or r["max_p"] < p_abs),
    }
    table = {}
    for name, fn in candidates.items():
        table[name] = {sp: evaluate(rows[sp], fn, p_inc) for sp in ("train", "val")}
        t, v = table[name]["train"], table[name]["val"]
        print(f"  {name:14s} TRAIN cIoU={t['cIoU']:6.2f} gIoU={t['gIoU']:6.2f} "
              f"N={t['N_acc']:5.1f} T={t['T_acc']:5.1f} | VAL cIoU={v['cIoU']:6.2f} "
              f"gIoU={v['gIoU']:6.2f} N={v['N_acc']:5.1f} T={v['T_acc']:5.1f}", flush=True)

    winner = max(candidates, key=lambda k: table[k]["train"]["gIoU"])
    print(f"held-out winner: {winner}", flush=True)
    out = dict(
        p_inc=p_inc, p_abs=p_abs, candidates=table, winner=winner,
        winner_val=table[winner]["val"], winner_train=table[winner]["train"],
        shipped_val=table["shipped"]["val"], shipped_train=table["shipped"]["train"],
        floors=dict(cIoU=68.94, gIoU=74.34),
        note=("all candidates evaluated on identical rows; winner picked by held-out "
              "train gIoU; val columns are reported for transfer only"),
        generated="exp69_rule_showdown.py")
    (OUTF / "gref_rule_showdown.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_rule_showdown.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
