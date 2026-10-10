#!/usr/bin/env python
"""EXP-78 — a LEARNED pair/union selector for the 2-object structure (model change 9).

exp77 measured the defect: over the selector's own pool, a perfect single pick
reaches cumulative targeted IoU 0.696, a perfect pair 0.825, a greedy union
0.872 — while the shipped threshold-union delivers 0.561.  The per-proposal
selector cannot express "these two masks together are the referent", and its
threshold-union demonstrably underperforms even single-mask picking.

This trains a selector whose CANDIDATES are unions: top-K proposals by the
frozen base classifier, candidates = K singles + C(K,2) pairs.  Features per
candidate: base probabilities, union area, pair overlap, box coverage of the
union (VLM box), released CLIP/GEM scores, pair flag.  Label: IoU(candidate
union, GT union) >= 0.5 (targeted only; no-target records contribute
negatives).  Prediction: the single best candidate above a threshold; the
shipped 3-probe ensemble rule handles no-target abstention unchanged.

Pre-registered protocol: fit/calib split by ref inside held-out train (never
val); 300 Optuna trials; objective min(cIoU/68.94, gIoU/74.34) on calib at
VAL PREVALENCE (the official metric's class mix); the study is compared
against the shipped selector's same-frame value; val is touched only if the
pair selector wins.

    python exp/exp78_pair_selector.py --trials 300
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
from exp62_clause_selector import load_record, make_clf  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUTF = Path("exp/out_final")
NT_FRAC = 0.6256
KMAX = 12


def build_candidates(r, base_scores, box_xy, K):
    """Candidate rows for top-K singles+pairs of record r. Returns (F, ids)."""
    H, W = r["masks"].shape[1], r["masks"].shape[2]
    n = len(base_scores)
    top = list(np.argsort(base_scores)[::-1][:min(K, n)])
    masks = r["masks"]
    sc = r.get("_sc"); gem = r.get("_gem")
    have_box = box_xy is not None
    bx0, by0, bx1, by1 = box_xy if have_box else (0, 0, 0, 0)
    box_area = max(1.0, float((bx1 - bx0) * (by1 - by0)))
    # per-proposal cached quantities
    msum = masks.sum((1, 2)).astype(np.float64)
    inbox = masks[:, by0:by1, bx0:bx1].sum((1, 2)).astype(np.float64) if have_box \
        else np.zeros(n)
    F, ids = [], []

    def emit(i, j, m, ov):
        a = float(m.sum())
        F.append([
            float(max(base_scores[i], base_scores[j])),
            float(min(base_scores[i], base_scores[j])),
            a / (H * W),
            ov,
            float(np.logical_and(m[by0:by1, bx0:bx1], True).sum()) / box_area if have_box else 0.0,
            (float(m[by0:by1, bx0:bx1].sum()) / a) if have_box and a > 0 else 0.0,
            float((sc[i] if j < 0 else max(sc[i], sc[j]))),
            float((gem[i] if j < 0 else max(gem[i], gem[j]))),
            1.0 if j >= 0 else 0.0,
            1.0 if have_box else 0.0,
        ])
        ids.append((int(i), int(j)))

    for i in top:
        emit(i, -1, masks[i], 0.0)
    for i, j in combinations(top, 2):
        m = np.logical_or(masks[i], masks[j])
        inter = float(np.logical_and(masks[i], masks[j]).sum())
        union = float(m.sum())
        emit(i, j, m, inter / max(1.0, union))
    return np.asarray(F, np.float32), ids


def candidate_mask(masks, cand):
    i, j = cand
    return masks[i] if j < 0 else np.logical_or(masks[i], masks[j])


def evaluate_pair(rows, clf, K, p_sel, use_ens, weighted=True,
                  kmask=None):
    """Official metrics with pair-candidate selection; weights at val prevalence.

    Rows carry precomputed (F, y, I, U) over the KMAX candidate list; `kmask`
    selects the shortlist-K candidates.
    """
    n_nt = sum(1 for r in rows if r["no_target"])
    n_t = len(rows) - n_nt
    w_nt = (NT_FRAC / (1 - NT_FRAC)) * (n_t / max(1, n_nt)) if weighted else 1.0
    acc_iou = acc_I = acc_U = wsum = 0.0
    tp = fn = tn = fp = 0.0
    for r in rows:
        w = w_nt if r["no_target"] else 1.0
        wsum += w
        km = kmask if kmask is not None else r["_km"][K]
        F = r["_F"][km]
        p = clf.predict_proba(F)[:, 1]
        ci = np.flatnonzero(km)[int(np.argmax(p))]
        abstain = (p.max() < p_sel) or (use_ens and any(r["ens"]))
        if abstain:
            if r["no_target"]:
                tp += w
                acc_iou += w
            else:
                fp += w
                acc_U += w * r["gt_area"]
            continue
        I = float(r["_I"][ci])
        U = float(r["_U"][ci])
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
                N_acc=100.0 * tp / max(1e-9, tp + fn),
                T_acc=100.0 * tn / max(1e-9, tn + fp))


def evaluate_shipped(rows, clf_base, p_inc, use_ens, weighted=True):
    """The shipped threshold-union rule, same frame (reference point)."""
    n_nt = sum(1 for r in rows if r["no_target"])
    n_t = len(rows) - n_nt
    w_nt = (NT_FRAC / (1 - NT_FRAC)) * (n_t / max(1, n_nt)) if weighted else 1.0
    acc_iou = acc_I = acc_U = wsum = 0.0
    tp = fn = tn = fp = 0.0
    for r in rows:
        w = w_nt if r["no_target"] else 1.0
        wsum += w
        p = clf_base.predict_proba(r["X"])[:, 1]
        sel = np.flatnonzero(p >= p_inc)
        abstain = (sel.size == 0) or (use_ens and any(r["ens"]))
        if abstain:
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
                N_acc=100.0 * tp / max(1e-9, tp + fn),
                T_acc=100.0 * tn / max(1e-9, tn + fp))


def precompute_row(r, base_scores):
    """All KMAX-singles + all pairs: features, labels, and (I, U) masses."""
    F, ids = build_candidates(r, base_scores, r["_box"], KMAX)
    gt = r["gt"]
    y = np.zeros(len(ids), np.float32)
    I = np.zeros(len(ids), np.float32)
    U = np.zeros(len(ids), np.float32)
    for c, (i, j) in enumerate(ids):
        m = candidate_mask(r["masks"], (i, j))
        I[c] = float(np.logical_and(m, gt).sum())
        U[c] = float(np.logical_or(m, gt).sum())
        if not r["no_target"]:
            y[c] = 1.0 if (U[c] > 0 and I[c] / U[c] >= 0.5) else 0.0
    return F, y, I, U


def topk_mask(F_len, n_actual, K):
    """Boolean mask over one row's precomputed candidate list for shortlist K."""
    s12 = min(KMAX, n_actual)
    m = np.zeros(F_len, bool)
    kk = min(K, n_actual)
    m[:kk] = True                       # singles 0..kk-1 of the s12 stored
    pos = s12
    for a in range(s12):
        for b in range(a + 1, s12):
            if a < kk and b < kk:
                m[pos] = True
            pos += 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/optuna_study_gref8.json"))
    args = ap.parse_args()

    st5 = json.loads(args.classifier.read_text())["best_params"]
    t0 = time.time()
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
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
        r["_sc"] = np.asarray(rec["sc"][:len(r["X"])], np.float32)
        r["_gem"] = np.asarray(rec["score_gem"][:len(r["X"])], np.float32)
        # box in stride-2 coords (same conversion as load_record)
        key = hashlib.sha1(f"{MODEL_TAG}|{rec['file_name']}|{ph.lower()}".encode()).hexdigest()
        bp = CACHE_DIR / (key + ".json")
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        r["_box"] = None
        if box is not None:
            H, W = r["masks"].shape[1], r["masks"].shape[2]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0 // 2), max(0, y0 // 2)
            x1, y1 = min(W, -(-x1 // 2)), min(H, -(-y1 // 2))
            if x1 > x0 and y1 > y0:
                r["_box"] = (x0, y0, x1, y1)
        rows.append(r)
    # base classifier (frozen study-5) for the shortlist + reference rule
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    rng = np.random.RandomState(args.seed)
    refs = sorted({r["rid"] for r in rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    for r in rows:
        r["split"] = "fit" if r["rid"] in fit_refs else "calib"
    fit_rows = [r for r in rows if r["split"] == "fit"]
    calib_rows = [r for r in rows if r["split"] == "calib"]
    clf_base.fit(np.concatenate([r["X"] for r in fit_rows]),
                 np.concatenate([r["y"] for r in fit_rows]))
    for r in rows:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"rows {len(rows)} fit {len(fit_rows)} calib {len(calib_rows)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # precompute candidate features+labels+masses at KMAX once per row
    for r in rows:
        r["_F"], r["_y"], r["_I"], r["_U"] = precompute_row(r, r["_bs"])
        s12 = min(KMAX, len(np.asarray(r["_bs"])))
        r["_km"] = {K: topk_mask(len(r["_F"]), s12, K) for K in (6, 8, 10, 12)}
    print(f"candidates built ({time.time()-t0:.0f}s)", flush=True)

    # shipped reference on the same calib rows (val-prevalence frame)
    ref = evaluate_shipped(calib_rows, clf_base, st5["p_inc"], True, weighted=True)
    print(f"shipped ref @val-prev: cIoU {ref['cIoU']:.2f} gIoU {ref['gIoU']:.2f} "
          f"N {ref['N_acc']:.1f} T {ref['T_acc']:.1f}", flush=True)

    def objective(trial):
        K = trial.suggest_categorical("K", [6, 8, 10, 12])
        C = trial.suggest_float("C", 1e-4, 1e1, log=True)
        p_sel = trial.suggest_float("p_sel", 0.30, 0.90)
        use_ens = trial.suggest_categorical("use_ens", [True, False])
        kind = trial.suggest_categorical("clf", ["logreg", "gbm"])
        if kind == "logreg":
            clf = make_clf("logreg", dict(C=C))
        else:
            clf = make_clf("gbm", dict(n_estimators=120,
                                       max_depth=trial.suggest_int("gd", 2, 4),
                                       learning_rate=0.1))
        # fit on fit rows' candidates at this K (precomputed; slice by mask)
        Xs, ys = [], []
        for r in fit_rows:
            km = r["_km"][K]
            Xs.append(r["_F"][km])
            ys.append(r["_y"][km])
        clf.fit(np.concatenate(Xs), np.concatenate(ys))
        m = evaluate_pair(calib_rows, clf, K, p_sel, use_ens, weighted=True)
        trial.set_user_attr("metrics", m)
        trial.set_user_attr("params", dict(K=K, C=C, p_sel=p_sel, use_ens=use_ens,
                                           clf=kind))
        return min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    bm = study.best_trial.user_attrs["metrics"]
    bp = study.best_trial.user_attrs["params"]
    print(f"best: value {study.best_value:.4f} cIoU {bm['cIoU']:.2f} gIoU {bm['gIoU']:.2f} "
          f"N {bm['N_acc']:.1f} T {bm['T_acc']:.1f} {bp}", flush=True)
    ref_v = min(ref["cIoU"] / 68.94, ref["gIoU"] / 74.34)
    out = dict(n_trials=args.trials, best_value=study.best_value,
               best_metrics=bm, best_params=bp,
               shipped_reference=ref, shipped_reference_value=ref_v,
               win_margin=study.best_value - ref_v,
               split="fit/calib by ref inside held-out train (seed %d, calib %.2f)" % (
                   args.seed, args.calib_frac),
               objective=("min(cIoU/68.94, gIoU/74.34) on calib at VAL PREVALENCE; "
                          "matches the official metric's class mix"),
               note=("pair/union selector: candidates = top-K singles + pairs of the frozen "
                     "study-5 classifier; val untouched; val pass only if win_margin > 0.005"),
               generated="exp78_pair_selector.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out} (win margin {out['win_margin']:+.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
