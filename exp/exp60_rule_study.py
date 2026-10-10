#!/usr/bin/env python
"""EXP-60 — decision-rule study: should a `yes` from the existence probe
override the selector's abstention?

The shipped rule abstains whenever no proposal clears p_inc — a precision
posture that also drops ~20 % of *targeted* samples.  The existence probe is
93 % accurate when it answers, so its `yes` is evidence the sample is
targeted.  This study searches the rule structure jointly with p_inc:

    rule == "std"        abstain = box_none | exists_no | not has_sel
    rule == "exoverride" abstain = box_none | exists_no | (not has_sel & exists unknown)
    rule == "exonly"     abstain = box_none | exists_no
    rule == "boxweak"    abstain = exists_no | (not has_sel & box_none)

Per-row prediction statistics are precomputed on a p_inc grid; the study runs
300 Optuna trials on the held-out fit/calib split scored at VAL prevalence,
freezes the winner, and takes one val pass.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp57_learned_selector import (load_record, make_clf, score_rows, REPRO)  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

VAL_NT_FRAC = 8896 / 14220
PINC_GRID = [0.60, 0.65, 0.70, 0.75, 0.795, 0.85, 0.90, 0.95]


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


def precompute(rows):
    """Per row: at each p_inc grid value, (has_sel, I, U, iou) + gt_area."""
    for r in rows:
        stats = []
        for p in PINC_GRID:
            sel = np.flatnonzero(r["scores"] >= p)
            if sel.size == 0:
                stats.append((False, 0.0, 0.0, 0.0))
                continue
            pred = r["masks"][sel].any(0)
            I = float(np.logical_and(pred, r["gt"]).sum())
            U = float(np.logical_or(pred, r["gt"]).sum())
            stats.append((True, I, U, I / U if U else 0.0))
        r["stats"] = stats
        r["gt_area"] = float(r["gt_area"])


def rule_abstains(rule, box_none, exists_no, has_sel):
    if exists_no is True:
        return True
    if rule == "std":
        return box_none or (not has_sel)
    if rule == "exoverride":
        return box_none or (not has_sel and exists_no is None)
    if rule == "exonly":
        return box_none
    if rule == "boxweak":
        return not has_sel and box_none
    raise ValueError(rule)


def evaluate_rule(rows, rule, p_inc_idx, w_nt, w_t, weighted=True):
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0.0
    wsum = 0.0
    for r in rows:
        w = (w_nt if r["no_target"] else w_t) if weighted else 1.0
        wsum += w
        has_sel, I, U, iou = r["stats"][p_inc_idx]
        abstained = rule_abstains(rule, r["vlm_none"],
                                  r.get("exists_no"), has_sel)
        if abstained:
            if r["no_target"]:
                tp += w
                acc_iou += w
            else:
                fp += w
                acc_U += w * r["gt_area"]
            continue
        if r["no_target"]:
            fn += w
            acc_U += w * U
        else:
            tn += w
            acc_iou += w * iou
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
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref3.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/optuna_study_gref4.json"))
    args = ap.parse_args()

    st = json.loads(args.classifier.read_text())["best_params"]
    params = (dict(C=st["C"]) if st["clf"] == "logreg" else
              dict(n_estimators=st["n_estimators"], max_depth=st["max_depth"],
                   learning_rate=st["learning_rate"]))
    kind = st["clf"]

    t0 = time.time()
    train_rows = build_rows(args.train_records, "train")
    val_rows = build_rows(args.val_records, "val")
    print(f"loaded train {len(train_rows)} val {len(val_rows)} ({time.time()-t0:.0f}s)",
          flush=True)

    rng = random.Random(args.seed)
    refs = sorted({r["rid"] for r in train_rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in train_rows if r["rid"] in fit_refs]
    calib_rows = [r for r in train_rows if r["rid"] not in fit_refs]

    clf = make_clf(kind, params)
    clf.fit(np.concatenate([r["X"] for r in fit_rows]),
            np.concatenate([r["y"] for r in fit_rows]))
    score_rows(calib_rows, clf)
    t1 = time.time()
    precompute(calib_rows)
    print(f"calib precomputed ({time.time()-t1:.0f}s)", flush=True)

    n_nt_c = sum(r["no_target"] for r in calib_rows)
    n_t_c = len(calib_rows) - n_nt_c
    w_nt = (VAL_NT_FRAC / (1 - VAL_NT_FRAC)) * (n_t_c / max(1, n_nt_c))

    def objective(trial):
        rule = trial.suggest_categorical("rule",
                                         ["std", "exoverride", "exonly", "boxweak"])
        p_idx = trial.suggest_categorical("p_inc_idx", list(range(len(PINC_GRID))))
        m = evaluate_rule(calib_rows, rule, p_idx, w_nt, 1.0)
        trial.set_user_attr("metrics", m)
        return min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials)
    best = study.best_trial
    print(f"{args.trials} trials in {time.time()-t0:.0f}s")
    print(f"best calib (val prevalence): {best.params} -> {best.user_attrs['metrics']}")

    # ---- freeze and transfer to val ---------------------------------------
    clf_all = make_clf(kind, params)
    clf_all.fit(np.concatenate([r["X"] for r in train_rows]),
                np.concatenate([r["y"] for r in train_rows]))
    score_rows(val_rows, clf_all)
    precompute(val_rows)
    vm = evaluate_rule(val_rows, best.params["rule"], best.params["p_inc_idx"],
                       1.0, 1.0, weighted=False)
    print(f"VAL transfer: rule={best.params['rule']} "
          f"p_inc={PINC_GRID[best.params['p_inc_idx']]} -> "
          f"cIoU={vm['cIoU']:.2f} gIoU={vm['gIoU']:.2f} "
          f"N_acc={vm['N_acc']:.1f} T_acc={vm['T_acc']:.1f}")

    # diagnostic: all rules at their best p_inc on val (never used for selection)
    diag = {}
    for rule in ("std", "exoverride", "exonly", "boxweak"):
        row = {}
        for i, p in enumerate(PINC_GRID):
            row[p] = evaluate_rule(val_rows, rule, i, 1.0, 1.0, weighted=False)
        diag[rule] = row
    out = dict(n_trials=args.trials, best_params=best.params,
               calib_metrics=best.user_attrs["metrics"], val_transfer=vm,
               pinc_grid=PINC_GRID, val_diagnostic=diag,
               classifier=str(args.classifier),
               generated="exp60_rule_study.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
