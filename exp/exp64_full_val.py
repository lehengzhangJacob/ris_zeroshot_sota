#!/usr/bin/env python
"""EXP-64 — evaluate the frozen gRefCOCO model on the FULL val split.

Same frozen components as the shipped row (learned selector with clause
features from exp62's study 5, existence branch, std rule), evaluated over all
14,220 val expressions instead of the 2,000-record subset — so all four main
tables rest on complete validation splits.

    python exp/exp64_full_val.py --out exp/out_final/gref_full_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")


def official(rows, p_inc, p_abs, use_exists):
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        abstained = ((r["max_p"] < p_abs) or r["vlm_none"]
                     or (use_exists and r.get("exists_no") is True))
        sel = np.flatnonzero(r["scores"] >= p_inc) if not abstained else np.array([], int)
        if not abstained and sel.size == 0:
            abstained = True
        if abstained:
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
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
    return dict(n=len(rows),
                cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / len(rows) if rows else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
                TP=tp, FN=fn, TN=tn, FP=fp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--val-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/gref_full_val.json"))
    args = ap.parse_args()

    st = json.loads(args.study.read_text())["best_params"]
    params = (dict(C=st["C"]) if st["clf"] == "logreg" else
              dict(n_estimators=st["n_estimators"], max_depth=st["max_depth"],
                   learning_rate=st["learning_rate"]))
    p_inc, p_abs = st["p_inc"], st["p_abs"]
    use_exists = bool(st.get("use_exists", True))
    print(f"frozen: {st['clf']} {params} p_inc={p_inc:.3f} p_abs={p_abs:.3f} "
          f"use_exists={use_exists}", flush=True)

    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    gd_va = GreferDataset(REPRO / "data/refer", "val")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    by_va = {r["ref_id"]: r for r in gd_va.refs}
    mc, gc = {}, {}
    train_rows = [r for r in (load_record(json.loads(l), gd_tr, by_tr, mc, gc)
                              for l in open(args.train_records)) if r is not None]
    print(f"train rows {len(train_rows)} ({time.time()-t0:.0f}s)", flush=True)
    val_rows = []
    n_skip = 0
    for l in open(args.val_records):
        r = load_record(json.loads(l), gd_va, by_va, mc, gc)
        if r is None:
            n_skip += 1
        else:
            val_rows.append(r)
    print(f"val rows {len(val_rows)} (skipped {n_skip}; cache incomplete) "
          f"({time.time()-t0:.0f}s)", flush=True)

    clf = make_clf(st["clf"], params)
    clf.fit(np.concatenate([r["X"] for r in train_rows]),
            np.concatenate([r["y"] for r in train_rows]))
    score_rows(val_rows, clf)
    m = official(val_rows, p_inc, p_abs, use_exists)
    print(f"FULL VAL: cIoU={m['cIoU']:.2f} gIoU={m['gIoU']:.2f} "
          f"N_acc={m['N_acc']:.1f} T_acc={m['T_acc']:.1f}  (n={m['n']})")
    out = dict(model=dict(kind=st["clf"], params=params, p_inc=p_inc,
                          p_abs=p_abs, use_exists=use_exists),
               metrics=m, n_val_records=len(val_rows), n_skipped=n_skip,
               records=str(args.val_records), source_study=str(args.study),
               generated="exp64_full_val.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
