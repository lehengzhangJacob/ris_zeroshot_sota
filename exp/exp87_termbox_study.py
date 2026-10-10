#!/usr/bin/env python
"""EXP-87 — per-term-box selection study on the val-matched population (change 10).

The structural discovery: gRefCOCO val targeted records are 100% multi-target
(n_targets>=2 for all 5,324), while the held-out fit sample is 33% multi.  The
union IS the task on val, and a per-term box construction (one box per parsed
term -> one mask per box -> union) applies to every val targeted record.  On
the val-matched held-out population (291 multi-target rows, exp84 boxes) the
hand pick rule sxi = argmax(base_score * bbox_IoU) beats the shipped pair
selector on r (0.5187 vs 0.4819); a learned picker and set structure may add.

This is the 300-trial held-out study: per-term pick rules {sxi hand, logreg,
GBM} x {top-1, top-2 with margin} x thresholds, fit/calib by ref within the
291, objective min(cIoU/68.94, gIoU/74.34) at val prevalence on calib; the
shipped pair rule is evaluated on the same calib rows as the bar.  Val is
untouched; a val pass follows only if the winner clears the bar by >0.01
absolute calib r.

    python exp/exp87_termbox_study.py --trials 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2
NT_FRAC = 0.6256


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/optuna_study_gref9.json"))
    args = ap.parse_args()

    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
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
    dedup = {r["rid"]: r for r in tr}

    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    print(f"rows {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    # shipped pair rule on the same rows (fit once on the full held-out fit set)
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    # per-term pick features + labels, per row
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        masks = r["masks"]
        gt = r["gt"]
        bs = r["_bs"]
        areas = masks.sum((1, 2)).astype(np.float64)
        iou_gt = (np.logical_and(masks, gt).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(masks, gt).sum((1, 2)), 1.0))
        fs, ts = [], []
        for ti, term in enumerate(rec["terms"]):
            p = npath(rec["file_name"], term)
            if not p.is_file():
                continue
            box = json.loads(p.read_text()).get("box")
            if box is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
            x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
            if x1 <= x0 or y1 <= y0:
                continue
            area_box = float((x1 - x0) * (y1 - y0))
            inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
            bxi = inb / (areas + area_box - inb + 1e-9)
            inbf = inb / np.maximum(areas, 1.0)
            n = len(bs)
            F = np.stack([
                bs, bxi, inbf, areas / masks[0].size,
                np.argsort(np.argsort(bs)) / max(1, n - 1),
                np.full(n, ti, np.float32),
                np.full(n, len(rec["terms"]), np.float32),
                np.full(n, bxi.max(), np.float32),
            ], 1).astype(np.float32)
            fs.append(F)
            ts.append((bxi, inbf))
        r["_termF"] = fs
        r["_termB"] = ts
        r["_iou_gt"] = iou_gt
        r["shipped_pred"] = candidate_mask(
            masks, build_candidates(r, r["_bs"], r["_box"], g8["K"])[1][
                int(np.argmax(pair_clf.predict_proba(
                    build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]))])
    print(f"features built ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(args.seed)
    refs = sorted({r["rid"] for r in rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in rows if r["rid"] in fit_refs]
    calib_rows = [r for r in rows if r["rid"] not in fit_refs]
    print(f"fit {len(fit_rows)} calib {len(calib_rows)}", flush=True)

    def shipped_r(rs):
        I = U = 0.0
        for r in rs:
            I += float(np.logical_and(r["shipped_pred"], r["gt"]).sum())
            U += float(np.logical_or(r["shipped_pred"], r["gt"]).sum())
        return I / max(1.0, U)

    ship_fit, ship_calib = shipped_r(fit_rows), shipped_r(calib_rows)
    print(f"shipped pair r: fit {ship_fit:.4f} calib {ship_calib:.4f}", flush=True)

    def union_pick(r, picks):
        if not picks:
            return None
        m = r["masks"][picks[0]].copy()
        for j in picks[1:]:
            m = np.logical_or(m, r["masks"][j])
        return m

    def eval_rule(rs, pick_fn):
        """pick_fn(r, ti) -> list of mask indices, or None to fall back to shipped."""
        tI = tU = 0.0
        for r in rs:
            pred = None
            if r["_termF"]:
                picks = []
                for ti in range(len(r["_termF"])):
                    js = pick_fn(r, ti)
                    if js:
                        picks.extend(js)
                picks = list(dict.fromkeys(picks))
                pred = union_pick(r, picks)
            if pred is None:
                pred = r["shipped_pred"]
            tI += float(np.logical_and(pred, r["gt"]).sum())
            tU += float(np.logical_or(pred, r["gt"]).sum())
        return tI / max(1.0, tU)

    def objective(trial):
        kind = trial.suggest_categorical("kind", ["sxi", "logreg", "gbm"])
        topk = trial.suggest_int("topk", 1, 2)
        margin = trial.suggest_float("margin", 0.5, 1.0)
        fallback = "shipped"
        if kind == "sxi":
            alpha = trial.suggest_float("alpha", 0.0, 1.5)
            beta = trial.suggest_float("beta", 0.0, 1.5)

            def pick(r, ti):
                bxi = r["_termB"][ti][0]
                sc = (r["_termF"][ti][:, 0]) * (alpha + beta * bxi)
                order = np.argsort(sc)[::-1]
                js = [int(order[0])]
                if topk == 2 and len(order) > 1 and sc[order[1]] >= margin * sc[order[0]]:
                    js.append(int(order[1]))
                return js
            fn = pick
        else:
            C = trial.suggest_float("C", 1e-4, 1e1, log=True)
            Xf, yf = [], []
            for r in fit_rows:
                for ti, F in enumerate(r["_termF"]):
                    lab = (r["_iou_gt"] >= 0.5).astype(np.float32)
                    Xf.append(F)
                    yf.append(lab)
            X = np.concatenate(Xf)
            y = np.concatenate(yf)
            if kind == "logreg":
                clf = make_clf("logreg", dict(C=C))
            else:
                clf = make_clf("gbm", dict(n_estimators=120, max_depth=3,
                                           learning_rate=0.1))
            clf.fit(X, y)

            def pick(r, ti):
                p = clf.predict_proba(r["_termF"][ti])[:, 1]
                order = np.argsort(p)[::-1]
                js = [int(order[0])]
                if topk == 2 and len(order) > 1 and p[order[1]] >= margin * p[order[0]]:
                    js.append(int(order[1]))
                return js
            fn = pick

        rc = eval_rule(calib_rows, fn)
        trial.set_user_attr("calib_r", rc)
        # map r to the min-ratio objective: cIoU scales ~ with r at fixed masses;
        # use the shipped calib r as the anchor: cIoU_proj = 36.75 * rc/ship_calib
        # (mass-approximate); gIoU unchanged (abstention untouched) at the
        # shipped operating point.
        ciou_proj = 36.75 * (rc / ship_calib) if ship_calib else 0.0
        trial.set_user_attr("proj_cIoU", ciou_proj)
        return min(ciou_proj / 68.94, 69.63 / 74.34)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # seed with the configurations measured outside the search (exp86 sxi, and
    # a conservative learned picker) so the baseline is always in the space
    study.enqueue_trial({"kind": "sxi", "topk": 1, "margin": 0.7,
                         "alpha": 0.2, "beta": 1.0})
    study.enqueue_trial({"kind": "sxi", "topk": 1, "margin": 0.7,
                         "alpha": 0.0, "beta": 1.0})
    study.enqueue_trial({"kind": "logreg", "topk": 1, "margin": 0.7, "C": 0.01})
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    print(f"best calib r {best.user_attrs['calib_r']:.4f} "
          f"(shipped {ship_calib:.4f}, margin {best.user_attrs['calib_r']-ship_calib:+.4f}) "
          f"proj cIoU {best.user_attrs['proj_cIoU']:.2f} params {best.params}", flush=True)
    out = dict(n_trials=args.trials, best_params=best.params,
               calib_r=best.user_attrs["calib_r"], shipped_calib_r=ship_calib,
               margin=best.user_attrs["calib_r"] - ship_calib,
               proj_cIoU=best.user_attrs["proj_cIoU"],
               shipped_fit_r=ship_fit, n_fit=len(fit_rows), n_calib=len(calib_rows),
               population="291 val-matched multi-target held-out train rows",
               note=("obj: min(proj_cIoU/68.94, 69.63/74.34); proj_cIoU anchors on the "
                     "shipped 36.75 scaled by calib r ratio; val pass only if margin "
                     "> 0.01 absolute calib r"),
               generated="exp87_termbox_study.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
