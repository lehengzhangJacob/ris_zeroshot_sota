#!/usr/bin/env python
"""EXP-59 — abstention frontier of the frozen exp57 model + prevalence fix.

The fourth study tuned its thresholds on a ~50/50 fit/calib split, but the
official val set is 62.6 % no-target, where abstention is worth more.  This
script

  1. refits the frozen (study-3/4) classifier on ALL held-out train rows and
     scores the val2000 rows once;
  2. sweeps the abstention threshold on the HELD-OUT CALIB split, computing the
     official metrics at VAL prevalence (row reweighting — a dataset statistic,
     not val labels), and freezes the best threshold;
  3. reports that frozen threshold's val transfer, plus the val-side frontier
     (diagnostic only, never used for selection).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp57_learned_selector import (load_record, make_clf, score_rows,  # noqa: E402
                                    evaluate, REPRO)
from grefer_data import GreferDataset  # noqa: E402

VAL_NT_FRAC = 8896 / 14220          # 0.6256, the official val prevalence


def build_rows(records: Path, split: str):
    gd = GreferDataset(REPRO / "data/refer", split)
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for rec in map(json.loads, open(records)):
        row = load_record(rec, gd, by_id, mc, gc)
        if row is not None:
            rows.append(row)
    return rows


def weighted_metrics(rows, p_inc, p_abs, use_exists, w_nt, w_t):
    """Official metrics on weighted rows (prevalence correction)."""
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0.0
    wsum = 0.0
    for r in rows:
        w = w_nt if r["no_target"] else w_t
        wsum += w
        abstained = ((r["max_p"] < p_abs) or r["vlm_none"]
                     or (use_exists and r.get("exists_no") is True))
        sel = np.flatnonzero(r["scores"] >= p_inc) if not abstained else np.array([], int)
        if not abstained and sel.size == 0:
            abstained = True
        if abstained:
            if r["no_target"]:
                tp += w
                acc_iou += w
            else:
                fp += w
                acc_U += w * r["gt_area"]
            continue
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn += w
            acc_U += w * U
        else:
            tn += w
            acc_iou += w * (I / U if U else 0.0)
            acc_I += w * I
            acc_U += w * U
    return dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / wsum if wsum else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--val-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_val2000.jsonl"))
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref3.json"))
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_tuned_val2000.json"))
    args = ap.parse_args()

    st = json.loads(args.study.read_text())
    bp = st["best_params"]
    kind = bp["clf"]
    params = (dict(C=bp["C"]) if kind == "logreg" else
              dict(n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
                   learning_rate=bp["learning_rate"]))
    p_inc = bp["p_inc"]
    use_exists = bool(bp.get("use_exists", False))
    print(f"frozen model from {args.study}: {kind} {params} p_inc={p_inc:.3f} "
          f"use_exists={use_exists}", flush=True)

    t0 = time.time()
    train_rows = build_rows(args.train_records, "train")
    val_rows = build_rows(args.val_records, "val")
    print(f"loaded train {len(train_rows)} val {len(val_rows)} ({time.time()-t0:.0f}s)",
          flush=True)

    # fit/calib split (same rule as exp57)
    rng = random.Random(args.seed)
    refs = sorted({r["rid"] for r in train_rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in train_rows if r["rid"] in fit_refs]
    calib_rows = [r for r in train_rows if r["rid"] not in fit_refs]

    clf = make_clf(kind, params)
    X = np.concatenate([r["X"] for r in fit_rows])
    y = np.concatenate([r["y"] for r in fit_rows])
    clf.fit(X, y)
    score_rows(calib_rows, clf)

    # ---- sweep on calib, scored at VAL prevalence --------------------------
    # calib is ~50/50; weight rows so the weighted no-target fraction equals
    # the official val prevalence (a dataset statistic, not per-sample labels)
    n_nt_c = sum(r["no_target"] for r in calib_rows)
    n_t_c = len(calib_rows) - n_nt_c
    w_nt = (VAL_NT_FRAC / (1 - VAL_NT_FRAC)) * (n_t_c / max(1, n_nt_c))
    print(f"calib rows {len(calib_rows)} ({n_nt_c} nt / {n_t_c} t); "
          f"weights w_t=1 w_nt={w_nt:.3f}", flush=True)
    best = None
    for p_abs in np.arange(0.0, 0.65, 0.025):
        m = weighted_metrics(calib_rows, p_inc, p_abs, use_exists,
                             w_nt=w_nt, w_t=1.0)
        v = min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)
        if best is None or v > best[0]:
            best = (v, float(p_abs), m)
    print(f"\ncalib-optimal at val prevalence: p_abs={best[1]:.3f} "
          f"calib cIoU={best[2]['cIoU']:.2f} gIoU={best[2]['gIoU']:.2f} "
          f"N_acc={best[2]['N_acc']:.1f} T_acc={best[2]['T_acc']:.1f}")
    p_abs_frozen = best[1]

    # ---- frozen transfer on val -------------------------------------------
    clf = make_clf(kind, params)          # refit on ALL train rows
    X = np.concatenate([r["X"] for r in train_rows])
    y = np.concatenate([r["y"] for r in train_rows])
    clf.fit(X, y)
    score_rows(val_rows, clf)
    vm = evaluate(val_rows, p_inc, p_abs_frozen, use_exists=use_exists)
    print(f"VAL (frozen p_abs={p_abs_frozen:.3f}): cIoU={vm['cIoU']:.2f} "
          f"gIoU={vm['gIoU']:.2f} N_acc={vm['N_acc']:.1f} T_acc={vm['T_acc']:.1f}")

    # ---- val frontier (diagnostic only) ------------------------------------
    frontier = []
    for p_abs in np.arange(0.0, 0.65, 0.025):
        m = evaluate(val_rows, p_inc, float(p_abs), use_exists=use_exists)
        frontier.append(dict(p_abs=float(p_abs), **m))
    print("\nval frontier (diagnostic; not used for selection):")
    for f in frontier:
        print(f"  p_abs={f['p_abs']:.3f} cIoU={f['cIoU']:6.2f} gIoU={f['gIoU']:6.2f} "
              f"N_acc={f['N_acc']:5.1f} T_acc={f['T_acc']:5.1f}")

    out = dict(model=dict(kind=kind, params=params, p_inc=p_inc,
                          use_exists=use_exists),
               calib_optimal=dict(p_abs=p_abs_frozen, metrics=best[2]),
               val_frozen=vm,
               val_frontier=frontier,
               n=len(val_rows),
               records=str(args.val_records),
               provenance=(f"classifier + inclusion threshold from {args.study}; "
                           "abstention threshold chosen on the held-out calib "
                           "split at val prevalence; val scored once after freezing"),
               generated="exp59_abstain_frontier.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
