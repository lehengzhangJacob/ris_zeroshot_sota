#!/usr/bin/env python
"""EXP-73 — does forced-choice pairwise preference improve the selector?

Gate experiment for the pairwise signal (exp72 extraction).  Same architecture
and protocol as exp62/exp57: per-proposal logistic selector on held-out train,
fit/calib split by ref, abstention = shipped 3-probe ensemble OR max_p < p_abs
OR no selection, objective = min(cIoU/68.94, gIoU/74.34) on calib (val
prevalence).  Two variants share every row:

  base  : the exp62 feature matrix (12 + 5 clause features)
  +pair : base PLUS [pair preference mean, pair win count, has_pair] for the
          proposal, computed from the top-4 shortlist's cached forced-choice
          answers (both orders averaged)

300 Optuna trials per variant (same budget as every study in this run).
Selection is by calib only; val is not touched here.  If +pair wins, the val
extraction (12x13,838 queries) is warranted; if not, the signal is dead and
the graveyard gets entry 21.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf, score_rows  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUTF = Path("exp/out_final")
TOPK = 4
NT_FRAC = 0.6256          # val no-target fraction (prevalence weighting)


def pair_path(fn, i, j, ph):
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    key = hashlib.sha1(
        f"{MODEL_TAG}|pair|{fn}|{i}|{j}|{ph.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def pref(fn, a, b, ph):
    """P(a beats b) in [0,1], averaging both presentation orders; 0.5 unknown."""
    wins = []
    for i, j in ((a, b), (b, a)):
        p = pair_path(fn, i, j, ph)
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        raw = (d.get("raw") or "").strip().lower()
        first_won = raw.startswith("first")
        second_won = raw.startswith("second")
        if not (first_won or second_won):
            continue
        wins.append(1.0 if first_won else 0.0)   # first == i here
    if not wins:
        return None
    return float(np.mean(wins))


def build_pair_feats(rec, row, ph, top):
    """(n,3) pair-feature columns; `rec` only supplies file_name."""
    """(n,3) array: pref mean, win count, has_pair, for classifier top-4."""
    n = len(row["X"])
    F = np.zeros((n, 3), np.float32)
    F[:, 0] = 0.5
    pf = {}
    for a, b in combinations(top, 2):
        v = pref(rec["file_name"], a, b, ph)
        pf[(a, b)] = v
    have = sum(1 for v in pf.values() if v is not None)
    if have == 0:
        return F
    for i in top:
        vals = []
        for a, b in combinations(top, 2):
            if i not in (a, b):
                continue
            v = pf.get((a, b))
            if v is None:
                continue
            vals.append(v if i == a else 1.0 - v)
        if not vals:
            continue
        F[i, 0] = float(np.mean(vals))
        F[i, 1] = float(sum(1 for v in vals if v > 0.5) - sum(1 for v in vals if v < 0.5))
        F[i, 2] = 1.0
    return F


def evaluate_ens(rows, p_inc, p_abs, use_ens):
    """Shipped rule family: abstain on any-probe-no | max_p<p_abs | no selection."""
    total = len(rows)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        gt_nt = r["no_target"]
        abstained = ((r["max_p"] < p_abs) or r["vlm_none"]
                     or (use_ens and any(r["ens"])))
        sel = np.flatnonzero(r["scores"] >= p_inc) if not abstained else np.array([], int)
        if not abstained and sel.size == 0:
            abstained = True
        if abstained:
            if gt_nt:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
        pred = r["masks"][sel].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if gt_nt:
            fn += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
    return dict(n=total,
                cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / total if total else 0.0,
                N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)


def weighted(rows):
    """Prevalence-reweighted cIoU/gIoU at val mix (importance weights)."""
    acc_iou = acc_I = acc_U = 0.0
    wsum = 0.0
    n_nt = sum(1 for r in rows if r["no_target"])
    n_t = len(rows) - n_nt
    w_nt = (NT_FRAC / (1 - NT_FRAC)) * (n_t / max(1, n_nt))
    tp_w = fn_w = tn_w = fp_w = 0.0
    wsum = n_t + w_nt * n_nt
    for r in rows:
        w = w_nt if r["no_target"] else 1.0
        if r["abstained"]:
            if r["no_target"]:
                tp_w += w
                acc_iou += w
            else:
                fp_w += w
                acc_U += w * r["gt_area"]
            continue
        pred = r["masks"][r["sel"]].any(0)
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn_w += w
            acc_U += w * U
        else:
            tn_w += w
            acc_iou += w * (I / U if U else 0.0)
            acc_I += w * I
            acc_U += w * U
    return dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / wsum if wsum else 0.0,
                N_acc=100.0 * tp_w / (tp_w + fn_w) if (tp_w + fn_w) else 0.0,
                T_acc=100.0 * tn_w / (tn_w + fp_w) if (tn_w + fp_w) else 0.0)


def mark_rule(rows, p_inc, p_abs, use_ens):
    for r in rows:
        abstained = ((r["max_p"] < p_abs) or r["vlm_none"]
                     or (use_ens and any(r["ens"])))
        sel = np.flatnonzero(r["scores"] >= p_inc) if not abstained else np.array([], int)
        if not abstained and sel.size == 0:
            abstained = True
        r["abstained"] = abstained
        r["sel"] = sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/optuna_study_gref7.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    st5 = json.loads(args.classifier.read_text())["best_params"]
    t0 = time.time()
    gd = GreferDataset(REPRO / "data/refer", "train")
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open(args.records):
        rec = json.loads(line)
        ph = (rec["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        r = load_record(rec, gd, by_id, mc, gc)
        if r is None:
            continue
        r["ens"] = [probe_no(t, rec["file_name"], ph)
                    for t in ("exists", "exists2", "exists3")]
        if any(v is None for v in r["ens"]):
            continue
        r["file_name"] = rec["file_name"]
        r["phrase"] = ph
        r["rec_key"] = (rec["ref_id"], rec["file_name"], ph)
        rows.append(r)
    print(f"rows with ensemble answers {len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    # fixed shortlist classifier (shipped study-5 params) -> top-4 for pair feats
    params5 = dict(C=st5["C"])
    clf5 = make_clf("logreg", params5)
    rng = np.random.RandomState(args.seed)
    refs = sorted({r["rid"] for r in rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    for r in rows:
        r["split"] = "fit" if r["rid"] in fit_refs else "calib"
    fit_rows = [r for r in rows if r["split"] == "fit"]
    calib_rows = [r for r in rows if r["split"] == "calib"]
    clf5.fit(np.concatenate([r["X"] for r in fit_rows]),
             np.concatenate([r["y"] for r in fit_rows]))
    score_rows(rows, clf5)
    print(f"fit rows {len(fit_rows)} calib rows {len(calib_rows)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # pair features per row
    n_missing = 0
    for r in rows:
        top = [int(j) for j in np.argsort(r["scores"])[::-1][:TOPK]]
        r["PF"] = build_pair_feats(r, r, r["phrase"], top)
        if r["PF"][:, 2].sum() == 0:
            n_missing += 1
    print(f"pair feats built; {n_missing} rows without any pair answer", flush=True)

    # direct signal diagnostic: on targeted rows, does the pairwise Copeland
    # winner match the GT-best proposal in the top-4?  (feature validity,
    # separate from whether the selector can exploit it)
    agree = n_tonly = 0
    for r in rows:
        if r["no_target"] or r["PF"][:, 2].sum() == 0:
            continue
        cand = np.flatnonzero(r["PF"][:, 2] > 0)
        ious = np.array([float(np.logical_and(r["masks"][c], r["gt"]).sum())
                         / max(1.0, float(np.logical_or(r["masks"][c], r["gt"]).sum()))
                         for c in cand])
        arg_gt = cand[int(np.argmax(ious))]
        arg_pair = cand[int(np.argmax(r["PF"][cand, 0]))]
        agree += int(arg_gt == arg_pair)
        n_tonly += 1
    diag_pair_agree = 100.0 * agree / n_tonly if n_tonly else 0.0
    print(f"pair-winner vs GT-best agreement: {agree}/{n_tonly} "
          f"({diag_pair_agree:.1f}%)", flush=True)

    for r in rows:
        r["XA"] = np.concatenate([r["X"], r["PF"]], 1)

    def objective(trial, variant):
        C = trial.suggest_float("C", 1e-4, 1e1, log=True)
        p_inc = trial.suggest_float("p_inc", 0.40, 0.98)
        p_abs = trial.suggest_float("p_abs", 0.01, 0.60)
        use_ens = trial.suggest_categorical("use_ens", [True, False])
        clf = make_clf("logreg", dict(C=C))
        key = "X" if variant == "base" else "XA"
        clf.fit(np.concatenate([r[key] for r in fit_rows]),
                np.concatenate([r["y"] for r in fit_rows]))
        for r in calib_rows:
            p = clf.predict_proba(r[key])[:, 1]
            r["scores"] = p
            r["max_p"] = float(p.max())
        m = evaluate_ens(calib_rows, p_inc, p_abs, use_ens)
        trial.set_user_attr("metrics", m)
        trial.set_user_attr("params", dict(C=C, p_inc=p_inc, p_abs=p_abs,
                                           use_ens=use_ens))
        return min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)

    results = {}
    for variant in ("base", "pair"):
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda t: objective(t, variant), n_trials=args.trials,
                       show_progress_bar=False)
        bm = study.best_trial.user_attrs["metrics"]
        bp = study.best_trial.user_attrs["params"]
        results[variant] = dict(value=study.best_value, metrics=bm, params=bp)
        print(f"{variant:4s}: calib value {study.best_value:.4f}  "
              f"cIoU {bm['cIoU']:.2f} gIoU {bm['gIoU']:.2f} "
              f"N {bm['N_acc']:.1f} T {bm['T_acc']:.1f}  {bp}", flush=True)
        # val-prevalence reweight of the calib winner (refit it first: the row
        # scores currently hold the LAST trial's classifier, not the best one)
        clf_b = make_clf("logreg", dict(C=bp["C"]))
        key = "X" if variant == "base" else "XA"
        clf_b.fit(np.concatenate([r[key] for r in fit_rows]),
                  np.concatenate([r["y"] for r in fit_rows]))
        for r in calib_rows:
            p = clf_b.predict_proba(r[key])[:, 1]
            r["scores"] = p
            r["max_p"] = float(p.max())
        mark_rule(calib_rows, bp["p_inc"], bp["p_abs"], bp["use_ens"])
        wm = weighted(calib_rows)
        results[variant]["calib_at_val_prevalence"] = wm
        print(f"      at val prevalence: cIoU {wm['cIoU']:.2f} gIoU {wm['gIoU']:.2f}", flush=True)

    winner = "pair" if results["pair"]["value"] > results["base"]["value"] + 0.005 else "base"
    out = dict(n_trials=args.trials, variants=results, winner=winner,
               pair_winner_vs_gtbest_agreement_pct=diag_pair_agree,
               split="gRefCOCO train fit/calib by ref (seed %d, calib %.2f)" % (
                   args.seed, args.calib_frac),
               objective="min(cIoU/68.94, gIoU/74.34) on calib, shipped ensemble rule family",
               shortlist="top-4 by shipped study-5 classifier (fixed)",
               note=("gate for exp72 pairwise forced-choice features; val untouched. "
                     "winner requires > +0.005 over base to trigger val extraction"),
               generated="exp73_pair_study.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"winner: {winner}; wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
