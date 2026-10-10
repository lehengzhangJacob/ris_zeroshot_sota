#!/usr/bin/env python
"""EXP-97 — hybrid candidate study v2 (no selection gate; selection only): per-term picks AND pair/union candidates (change 11).

exp94 decomposed the shipped termbox rule: it realizes 86% of its own
in-box oracle (0.5059 vs 0.5855 with LLaVA boxes; 0.6873 with GT boxes) --
but that construction's ceiling is BELOW the cIoU requirement (0.8467), while
the pair/union candidate family reaches a 0.836 full-pool pair oracle and the
pool subset oracle 0.8842.  Neither family alone spans the headroom.

The hybrid: candidates = the pair/union candidates (top-6 singles + pairs of
the frozen base classifier, 21) PLUS the per-term picks (one per term box,
plus the current termbox-union) -- all scored by one learned selector over the
union of both feature blocks, abstention unchanged, pair-fallback when a row
has no term boxes.  The candidate oracle is the union of both families.

300 trials, held-out tune rows (termbox features zero-padded where absent),
objective min-ratio at val prevalence; gate = beat BOTH shipped rules on the
same calib rows by >0.01 absolute r.  Val untouched.

    python exp/exp95_hybrid_study.py --trials 300
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
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp84_nounbox_sample import npath  # noqa: E402
from exp89_termbox_full_val import term_features  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2
NT_FRAC = 0.6256


def hybrid_candidates(r, rec, pair_F, pair_ids, g9, pick_clf):
    """Pair candidates + termbox candidates; returns (X, specs)."""
    specs = []
    Xs = []
    for Fi, (i, j) in zip(pair_F, pair_ids):
        specs.append(("pair", i, j))
        Xs.append(np.concatenate([Fi, np.zeros(4, np.float32)]))
    if rec is not None and pick_clf is not None:
        picks = []
        for F_t in term_features(r, rec):
            pt = pick_clf.predict_proba(F_t)[:, 1]
            order = np.argsort(pt)[::-1]
            a = int(order[0])
            specs.append(("term", a, -1))
            Xs.append(np.concatenate([
                np.zeros(10, np.float32),
                np.array([1, 0, float(pt[a]), float(F_t[a, 1])], np.float32)]))
            picks.append(a)
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                b = int(order[1])
                specs.append(("term", b, -1))
                Xs.append(np.concatenate([
                    np.zeros(10, np.float32),
                    np.array([1, 0, float(pt[b]), float(F_t[b, 1])], np.float32)]))
                picks.append(b)
        picks = list(dict.fromkeys(picks))
        if picks:
            specs.append(("termunion", -1, -1))
            Xs.append(np.concatenate([
                np.zeros(10, np.float32),
                np.array([0, 1, 1.0, 0.0], np.float32)]))
            r["_term_picks"] = picks
    if not Xs:
        return None, None
    return np.asarray(Xs, np.float32), specs


def cand_mask(r, spec):
    kind, i, j = spec
    if kind == "pair":
        return candidate_mask(r["masks"], (i, j))
    if kind == "term":
        return r["masks"][i]
    m = r["masks"][r["_term_picks"][0]].copy()
    for p in r["_term_picks"][1:]:
        m = np.logical_or(m, r["masks"][p])
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/optuna_study_gref11.json"))
    args = ap.parse_args()

    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}

    # frozen shipped sub-models: pair fallback + termbox picker
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features(r, rec):
            Xf.append(F)
            yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))
    print(f"sub-models fitted ({time.time()-t0:.0f}s)", flush=True)

    # precompute hybrid candidates + labels per row
    for r in tr:
        rec = recs_by_rid.get(r["rid"])
        if rec is not None and not rec.get("terms"):
            rec = None
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        H, specs = hybrid_candidates(r, rec, F, ids, g9, pick_clf if rec else None)
        r["_H"], r["_specs"] = H, specs
        y = np.zeros(len(specs), np.float32) if specs else np.zeros(0, np.float32)
        if specs and not r["no_target"]:
            gt = r["gt"]
            for c, sp in enumerate(specs):
                m = cand_mask(r, sp)
                I = float(np.logical_and(m, gt).sum())
                U = float(np.logical_or(m, gt).sum())
                y[c] = 1.0 if (U > 0 and I / U >= 0.5) else 0.0
        r["_Hy"] = y
        r["_pp"] = None
    print(f"hybrid candidates built ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(args.seed)
    refs = sorted({r["rid"] for r in tr})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    fit_rows = [r for r in tr if r["rid"] in fit_refs]
    calib_rows = [r for r in tr if r["rid"] not in fit_refs]
    print(f"fit {len(fit_rows)} calib {len(calib_rows)} "
          f"(hybrid rows in calib: {sum(1 for r in calib_rows if r['_H'] is not None and len(r['_H'])>21)})",
          flush=True)

    p_sel = g8["p_sel"]

    def shipped_ref(rs, mode):
        """mode 'pair' or 'termbox' : the two shipped rules' calib r."""
        tI = tU = 0.0
        for r in rs:
            F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
            pp = pair_clf.predict_proba(F)[:, 1]
            abstain = (pp.max() < p_sel) or any(r["ens"])
            if abstain:
                continue
            pred = None
            if mode == "termbox":
                rec = recs_by_rid.get(r["rid"])
                if rec is not None and rec.get("terms"):
                    picks = []
                    for F_t in term_features(r, rec):
                        pt = pick_clf.predict_proba(F_t)[:, 1]
                        order = np.argsort(pt)[::-1]
                        picks.append(int(order[0]))
                        if g9["topk"] == 2 and len(order) > 1 and \
                                pt[order[1]] >= g9["margin"] * pt[order[0]]:
                            picks.append(int(order[1]))
                    picks = list(dict.fromkeys(picks))
                    if picks:
                        pred = r["masks"][picks[0]].copy()
                        for p_ in picks[1:]:
                            pred = np.logical_or(pred, r["masks"][p_])
            if pred is None:
                pred = candidate_mask(r["masks"], ids[int(np.argmax(pp))])
            tI += float(np.logical_and(pred, r["gt"]).sum())
            tU += float(np.logical_or(pred, r["gt"]).sum())
        return tI / max(1.0, tU)

    ship_pair = shipped_ref(calib_rows, "pair")
    ship_term = shipped_ref(calib_rows, "termbox")
    print(f"shipped calib r: pair {ship_pair:.4f} termbox {ship_term:.4f}", flush=True)

    def eval_hybrid(rs, clf, p_inc):
        tI = tU = 0.0
        for r in rs:
            if r["_H"] is None:
                continue
            proba = clf.predict_proba(r["_H"])[:, 1]
            c = int(np.argmax(proba))
            if proba[c] < p_inc:
                continue
            pred = cand_mask(r, r["_specs"][c])
            tI += float(np.logical_and(pred, r["gt"]).sum())
            tU += float(np.logical_or(pred, r["gt"]).sum())
        return tI / max(1.0, tU)

    def objective(trial):
        kind = trial.suggest_categorical("kind", ["logreg", "gbm"])
        p_inc = 0.0          # v2: no selection gate -- abstention stays as shipped
        if kind == "logreg":
            C = trial.suggest_float("C", 1e-5, 1e1, log=True)
            clf = make_clf("logreg", dict(C=C))
        else:
            gd = trial.suggest_int("gd", 2, 4)
            clf = make_clf("gbm", dict(n_estimators=150, max_depth=gd,
                                       learning_rate=0.1))
        Xf = [r["_H"] for r in fit_rows if r["_H"] is not None]
        yf = [r["_Hy"] for r in fit_rows if r["_H"] is not None]
        clf.fit(np.concatenate(Xf), np.concatenate(yf))
        rc = eval_hybrid(calib_rows, clf, p_inc)
        trial.set_user_attr("calib_r", rc)
        trial.set_user_attr("params", dict(kind=kind,
                                           **({"C": C} if kind == "logreg" else
                                              {"gd": gd})))
        anchor = 40.96 * (1 + (rc - ship_term))
        return min(anchor / 68.94, 71.11 / 74.34)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({"kind": "logreg", "C": 0.01})
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_trial
    rc = best.user_attrs["calib_r"]
    print(f"best hybrid calib r {rc:.4f} vs pair {ship_pair:.4f} / termbox {ship_term:.4f} "
          f"(margins {rc-ship_pair:+.4f}/{rc-ship_term:+.4f}) params {best.params}",
          flush=True)
    out = dict(n_trials=args.trials, best_params=best.user_attrs["params"],
               best_calib_r=rc, shipped_pair_calib_r=ship_pair,
               shipped_termbox_calib_r=ship_term,
               margin_over_pair=rc - ship_pair, margin_over_termbox=rc - ship_term,
               n_fit=len(fit_rows), n_calib=len(calib_rows),
               note=("hybrid = pair candidates + per-term picks + termbox-union, one "
                     "selector over both feature blocks; val untouched; val pass only if "
                     "both margins > 0.01"),
               generated="exp97_hybrid_study_v2.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
