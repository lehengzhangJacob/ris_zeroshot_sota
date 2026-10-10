#!/usr/bin/env python
"""EXP-67 — three-probe existence ensemble: held-out rule pick + full-val pass.

The reviewer's objection: gIoU may be transfer-bound, so a distribution-robust
existence signal is the right untried experiment.  Three yes/no probes
(exists / exists2 / exists3, exp58+variants) are now cached on the full val.

Rules under test (both use the frozen exp62 classifier for mask selection):
  (a) ensemble-only : abstain iff ANY probe answers no
  (b) ensemble+empty: also abstain when no proposal clears p_inc (the std rule
                      with the single probe replaced by the ensemble)

The rule is chosen on the held-out train sample (gIoU objective); the winner
then takes one full-val pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def probe_no(tag, fn, ph):
    name = MODEL_TAG if tag is None else f"{MODEL_TAG}|{tag}"
    p = CACHE_DIR / (hashlib.sha1(f"{name}|{fn}|{ph.lower()}".encode()).hexdigest() + ".json")
    if not p.is_file():
        return None
    raw = (json.loads(p.read_text()).get("raw") or "").strip().lower()
    if raw.startswith("y"):
        return False
    if raw.startswith("n"):
        return True
    return None


def build_rows(records, split, clf, params, limit=0):
    gd = GreferDataset(REPRO / "data/refer", split)
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for i, line in enumerate(open(records)):
        if limit and i >= limit:
            break
        rec = json.loads(line)
        row = load_record(rec, gd, by_id, mc, gc)
        if row is None:
            continue
        fn = rec["file_name"]
        ph = (rec["parsed"].get("sentence") or "").strip()
        row["ens"] = [probe_no(t, fn, ph) for t in ("exists", "exists2", "exists3")]
        row["box_none"] = row["vlm_none"]
        if any(v is None for v in row["ens"]):
            continue
        rows.append(row)
    for r in rows:
        p = clf.predict_proba(r["X"])[:, 1]
        r["scores"] = p
        r["max_p"] = float(p.max())
    return rows


def evaluate(rows, rule, p_inc):
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        has_pred = bool(np.flatnonzero(r["scores"] >= p_inc).size)
        abstain = any(r["ens"])
        if not abstain and rule == "ens_empty" and not has_pred:
            abstain = True
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--val-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/gref_ensemble_val.json"))
    args = ap.parse_args()

    st = json.loads(args.classifier.read_text())["best_params"]
    params = dict(C=st["C"]) if st["clf"] == "logreg" else dict(
        n_estimators=st["n_estimators"], max_depth=st["max_depth"],
        learning_rate=st["learning_rate"])
    clf = make_clf(st["clf"], params)
    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    feats = []
    for line in open(args.train_records):
        r = load_record(json.loads(line), gd_tr, by_tr, {}, {})
        if r is not None:
            feats.append(r)
    clf.fit(np.concatenate([r["X"] for r in feats]),
            np.concatenate([r["y"] for r in feats]))
    print(f"classifier fit ({time.time()-t0:.0f}s)", flush=True)

    tr = build_rows(args.train_records, "train", clf, params)
    print(f"train rows with all 3 probes: {len(tr)}", flush=True)
    picks = {}
    for rule in ("ens_only", "ens_empty"):
        m = evaluate(tr, rule, st["p_inc"])
        picks[rule] = m
        print(f"  TRAIN {rule:10s} cIoU={m['cIoU']:6.2f} gIoU={m['gIoU']:6.2f} "
              f"N={m['N_acc']:5.1f} T={m['T_acc']:5.1f}", flush=True)
    winner = max(picks, key=lambda k: picks[k]["gIoU"])
    print(f"held-out winner: {winner}", flush=True)

    val = build_rows(args.val_records, "val", clf, params)
    print(f"val rows with all 3 probes: {len(val)}/{14220} ({time.time()-t0:.0f}s)", flush=True)
    vm = evaluate(val, winner, st["p_inc"])
    print(f"FULL VAL [{winner}]: cIoU={vm['cIoU']:.2f} gIoU={vm['gIoU']:.2f} "
          f"N_acc={vm['N_acc']:.1f} T_acc={vm['T_acc']:.1f} (n={vm['n']})", flush=True)
    out = dict(winner=winner, train=picks, val_transfer=vm,
               p_inc=st["p_inc"], n_val=len(val),
               official_floors=dict(cIoU=68.94, gIoU=74.34),
               note=("three yes/no existence probes ensemble; rule chosen on held-out train by "
                     "gIoU; single val pass after freezing"),
               generated="exp67_ensemble_eval.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
