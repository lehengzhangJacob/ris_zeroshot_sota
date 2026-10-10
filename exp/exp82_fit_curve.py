#!/usr/bin/env python
"""EXP-82 — learning curve of the pair/union selector vs fit-set size.

The shipped pair classifier is fitted on 1,398 held-out train records.  If its
realization is still rising steeply with fit size, scaling the fit set (more
gRefCOCO TRAIN refs — never val/test) is the one untried lever; if the curve is
flat at 1,398, the features/model are the cap and more extraction is not worth
its hours.

Frozen everything: base classifier (study-5), pair rule (K=6, C=2.85e-4,
p_sel=0.340, ens) exactly as shipped; only the number of refs the PAIR model is
fitted on varies: 10%, 25%, 50%, 75%, 100% of the 1,398-row held-out sample.
Evaluated on the exp78 calib split (423 rows) at val prevalence, plus the
cumulative targeted r on the same rows.

This is measurement only: no parameter is chosen here, and val is untouched.

    python exp/exp82_fit_curve.py
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
from exp67_ensemble_eval import probe_no  # noqa: E402

OUTF = Path("exp/out_final")
NT_FRAC = 0.6256


def main() -> int:
    study = json.loads((OUTF / "optuna_study_gref8.json").read_text())
    bp = study["best_params"]
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    K, p_sel, use_ens = bp["K"], bp["p_sel"], bp["use_ens"]

    t0 = time.time()
    import json as _json
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"train rows {len(tr)} ({time.time()-t0:.0f}s)", flush=True)

    # exp78 split for the calib evaluation (seed 0, calib_frac 0.3)
    rng = np.random.RandomState(0)
    refs = sorted({r["rid"] for r in tr})
    rng.shuffle(refs)
    n_fit = int(0.7 * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in tr if r["rid"] in fit_refs]
    calib_rows = [r for r in tr if r["rid"] not in fit_refs]
    print(f"fit refs {len(fit_refs)} calib rows {len(calib_rows)}", flush=True)

    # precompute candidates once (features/labels are fit-size-independent)
    for r in tr:
        r["_F"], r["_y"], r["_I"], r["_U"] = precompute_row(r, r["_bs"])
        r["_km"] = topk_mask(len(r["_F"]), len(r["_bs"]), K)

    def fit_and_eval(subset):
        Xs = [r["_F"][r["_km"]] for r in subset]
        ys = [r["_y"][r["_km"]] for r in subset]
        clf = make_clf("logreg", dict(C=bp["C"]))
        clf.fit(np.concatenate(Xs), np.concatenate(ys))
        n_nt = sum(1 for r in calib_rows if r["no_target"])
        n_t = len(calib_rows) - n_nt
        w_nt = (NT_FRAC / (1 - NT_FRAC)) * (n_t / max(1, n_nt))
        acc_iou = acc_I = acc_U = wsum = 0.0
        tp = fn = tn = fp = 0.0
        tI = tU = 0.0
        for r in calib_rows:
            w = w_nt if r["no_target"] else 1.0
            wsum += w
            F = r["_F"][r["_km"]]
            p = clf.predict_proba(F)[:, 1]
            cpos = np.flatnonzero(r["_km"])[int(np.argmax(p))]
            abstain = (p.max() < p_sel) or (use_ens and any(r["ens"]))
            if abstain:
                if r["no_target"]:
                    tp += w
                    acc_iou += w
                else:
                    fp += w
                    acc_U += w * r["gt_area"]
                continue
            I = float(r["_I"][cpos])
            U = float(r["_U"][cpos])
            if r["no_target"]:
                fn += w
                acc_U += w * U
            else:
                tn += w
                acc_iou += w * (I / U if U else 0.0)
                acc_I += w * I
                acc_U += w * U
                tI += I
                tU += U
        return dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                    gIoU=100.0 * acc_iou / wsum if wsum else 0.0,
                    N_acc=100.0 * tp / max(1e-9, tp + fn),
                    T_acc=100.0 * tn / max(1e-9, tn + fp),
                    targeted_r_cumulative=tI / tU if tU else 0.0)

    out = {}
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        n = max(10, int(frac * len(fit_refs)))
        sub = fit_rows[:n] if n < len(fit_rows) else fit_rows
        m = fit_and_eval(sub)
        out[f"fit_frac_{frac:.2f}"] = dict(n_refs=n, **m)
        print(f"fit {n:4d} refs ({frac:.0%}): cIoU {m['cIoU']:6.2f} gIoU {m['gIoU']:6.2f} "
              f"N {m['N_acc']:5.1f} T {m['T_acc']:5.1f} "
              f"targeted_r {m['targeted_r_cumulative']:.4f}", flush=True)

    r1 = out["fit_frac_1.00"]["targeted_r_cumulative"]
    r_lo = out["fit_frac_0.25"]["targeted_r_cumulative"]
    slope = (r1 - r_lo) / (1.0 - 0.25)
    art = dict(measured_on="exp78 calib split (423 rows, val prevalence), frozen shipped rule",
               curve=out,
               realization_slope_per_100pct_r=round(slope, 4),
               interpretation=("slope over 25%->100% fit refs; a large positive slope means more "
                               "fit data is the lever (launch the train-split extraction); a flat "
                               "curve means features/model capacity is the cap"),
               generated="exp82_fit_curve.py")
    (OUTF / "gref_fit_curve.json").write_text(json.dumps(art, indent=2))
    print(f"slope over 25->100%: {slope:+.4f} r per 100% of 1398 refs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
