#!/usr/bin/env python
"""EXP-152 — the train->val TRANSFER INVERSION (operator directive 2026-10-13).

Measured facts this cycle must explain (from our own artifacts, no forced
supervised comparison; targets are our own same-class record):
  * learned arms look excellent on held-out train rows and fail on val
    (naive selector +6.8 held-out -> -12.5 val; best learned arm held-out
    56.09 cIoU -> val 50.52 vs the 51.32 record);
  * val-time priors are systematically weaker (predicted part-IoU 0.413 train
    vs 0.355 val);
  * the populations differ (val targeted records are multi-target; a large
    share of train-side targeted rows are single-target).

Modes:
  build-tables  build + cache the train/val per-part tables (X, y, meta) with
                per-row metadata (n_targets, term count, masks-per-image)
  characterize  falsifiable decompositions: (a) prior-quality calibration shift,
                (b) population/composition mismatch, (c) per-feature drift,
                (d) image-level shift
  recipe        transfer-aware arm: composition-matched re-weighting (train
                metadata only) + rank features; held-out tuned, one val pass
  policy        combination-policy search over parts (the pair-oracle headroom
                lead): searched subset policy, held-out tuned, one val pass
  all           characterize + recipe + policy -> exp/out_final/transfer_wall.json

    CUDA_VISIBLE_DEVICES=1 python exp/exp152_transfer_wall.py --mode build-tables
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

RIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RIS / "exp"))
from exp138_learned_union_selector import (  # noqa: E402
    TRAIN, VAL, build_candidates, build_parts, load_rows, make_clf,
    precompute_row, terms_of, topk_mask,
)

OUTF = RIS / "exp/out_final"
WORK = RIS / "exp/out_transfer"
WORK.mkdir(parents=True, exist_ok=True)
FEATS = ["is_pool", "is_sam", "area_frac", "inbox_llava", "inbox_det0",
         "base_score", "gem_score", "pick_rank", "det_index", "sam_index",
         "term_index", "const"]


def rank_within_rows(X, row_of):
    X = X.copy()
    cols = [2, 3, 4, 5, 6]
    by_row = {}
    for i, (ri, _) in enumerate(row_of):
        by_row.setdefault(ri, []).append(i)
    for ri, idxs in by_row.items():
        n = len(idxs)
        if n <= 1:
            X[np.asarray(idxs), cols] = 0.5
            continue
        sub = X[np.asarray(idxs)][:, cols]
        for c in range(len(cols)):
            order = np.argsort(np.argsort(sub[:, c]))
            X[np.asarray(idxs)[order], cols[c]] = np.linspace(0, 1, n)
    return X


def setup(args):
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    cfg = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    cfg.setdefault("gd_thresh", 0.25)
    p_sel = float(g8["p_sel"])
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(TRAIN, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    return dict(st5=st5, g8=g8, cfg=cfg, p_sel=p_sel, clf_base=clf_base, pair_clf=pair_clf)


def record_meta(path):
    """Per-record metadata from the records file (train-side only fields)."""
    meta = {}
    for line in open(path):
        rec = json.loads(line)
        meta[int(rec["ref_id"])] = dict(
            no_target=bool(rec.get("no_target")),
            n_targets=len([a for a in (rec.get("ann_ids") or rec.get("ann_id") or [])
                           if isinstance(a, int) and a >= 0]),
            n_sent=len(rec.get("sent_ids") or []),
            n_masks=rec.get("n"),
        )
    return meta


def build_side(side, args, ctx):
    out = WORK / f"{side}_parts.npz"
    if out.is_file():
        print(f"{side}: cached parts exist", flush=True)
        return
    path = TRAIN if side == "train" else VAL
    train_side = side == "train"
    rows = load_rows(path, side, ctx["clf_base"])
    for r in rows:
        r["_bs"] = ctx["clf_base"].predict_proba(r["X"])[:, 1]
    terms_by_rid = {}
    for line in open(path):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            terms_by_rid[int(rec["ref_id"])] = (rec["file_name"], ts)
    X_all, y_all, row_of, ious_of = [], [], [], []
    keep_rows = []
    t0 = time.time()
    for ri, r in enumerate(rows):
        if r["no_target"]:
            continue
        fn_ts = terms_by_rid.get(r["rid"])
        if not fn_ts:
            continue
        pp = ctx["pair_clf"].predict_proba(
            build_candidates(r, r["_bs"], r["_box"], ctx["g8"]["K"])[0])[:, 1]
        if (pp.max() < ctx["p_sel"]) or any(r["ens"]):
            continue
        built = build_parts(r, fn_ts[0], fn_ts[1], ctx["cfg"], train_side=train_side)
        if built is None:
            continue
        masks, X, meta = built
        ious = [float(np.logical_and(m, r["gt"]).sum()) /
                max(1.0, float(np.logical_or(m, r["gt"]).sum())) for m in masks]
        keep_rows.append(ri)
        for k in range(len(masks)):
            X_all.append(X[k]); y_all.append(ious[k]); row_of.append((ri, k))
        ious_of.append(ious)
        if (ri + 1) % 500 == 0:
            print(f"  {side} {ri+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    X_all = np.asarray(X_all, np.float32)
    y_all = np.asarray(y_all, np.float32)
    row_ids = np.asarray([ri for ri, _ in row_of], np.int64)
    np.savez_compressed(out, X=X_all, y=y_all, row_ids=row_ids,
                        n_parts=len(X_all), n_rows=len(keep_rows))
    rmeta = record_meta(path)
    meta = {str(ri): dict(rid=rows[ri]["rid"],
                          n_targets=int(rmeta.get(rows[ri]["rid"], {}).get("n_targets", -1)),
                          n_terms=len(terms_by_rid.get(rows[ri]["rid"], (None, []))[1]),
                          n_parts=len(ious_of[j]))
            for j, ri in enumerate(keep_rows)}
    (WORK / f"{side}_rows.json").write_text(json.dumps(meta))
    print(f"{side}: {len(X_all)} parts over {len(keep_rows)} rows -> {out}", flush=True)


def holdout_split(n_rows):
    refs = sorted(range(n_rows))
    rng = np.random.RandomState(0)
    rng.shuffle(refs)
    fit = set(refs[:int(0.7 * len(refs))])
    return fit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["build-tables", "characterize", "recipe", "policy", "all"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--side", default="", help="build-tables: train|val|both")
    args = ap.parse_args()

    if args.mode == "build-tables":
        ctx = setup(args)
        if args.side in ("", "both", "train"):
            build_side("train", args, ctx)
        if args.side in ("", "both", "val"):
            build_side("val", args, ctx)
        return 0

    doc = json.loads((OUTF / "transfer_wall.json").read_text()) \
        if (OUTF / "transfer_wall.json").is_file() else dict(
            what="train->val transfer inversion (2026-10-13 directive)",
            modes={})
    ctx = setup(args)
    tr = np.load(WORK / "train_parts.npz")
    va = np.load(WORK / "val_parts.npz")
    tr_rows = json.loads((WORK / "train_rows.json").read_text())
    va_rows = json.loads((WORK / "val_rows.json").read_text())

    from sklearn.ensemble import HistGradientBoostingRegressor

    def fit_model(fit_mask_rows):
        Xr = rank_within_rows(tr["X"], list(zip(tr["row_ids"], range(len(tr["row_ids"])))))
        fit = np.isin(tr["row_ids"], list(fit_mask_rows))
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.08,
                                          random_state=0)
        m.fit(Xr[fit], tr["y"][fit])
        return m, Xr

    if args.mode in ("characterize", "all"):
        out = {}
        # ---- (b) population / composition ---------------------------------
        tr_t = [v for v in tr_rows.values() if v["n_targets"] >= 0]
        va_t = [v for v in va_rows.values()]
        def comp(rows_):
            c = Counter(v["n_targets"] for v in rows_)
            n = max(1, len(rows_))
            return {str(k): round(v / n, 4) for k, v in sorted(c.items())}
        out["population"] = dict(
            train_targeted_rows=len(tr_t), val_targeted_rows=len(va_t),
            train_n_targets_composition=comp(tr_t), val_n_targets_composition=comp(va_t),
            train_mean_parts_per_row=round(float(np.mean([v["n_parts"] for v in tr_t])), 3),
            val_mean_parts_per_row=round(float(np.mean([v["n_parts"] for v in va_t])), 3),
        )
        # ---- (a) prior-quality shift via a train-fitted model -------------
        fit_rows = holdout_split(len(tr_t))
        model, Xr_tr = fit_model(fit_rows)
        # val-side rank transform must be per val row
        vrow_of = list(zip(va["row_ids"], range(len(va["row_ids"]))))
        Xr_va = rank_within_rows(va["X"], vrow_of)
        pred_tr = model.predict(Xr_tr)
        pred_va = model.predict(Xr_va)
        def calib(pred, y, nb=5):
            qs = np.quantile(pred, np.linspace(0, 1, nb + 1))
            rows = []
            for i in range(nb):
                lo, hi = qs[i], qs[i + 1]
                m = (pred >= lo) & (pred <= hi if i == nb - 1 else pred < hi)
                if m.sum():
                    rows.append(dict(pred=round(float(pred[m].mean()), 4),
                                     actual=round(float(y[m].mean()), 4), n=int(m.sum())))
            return rows
        out["prior_quality_shift"] = dict(
            train_calibration=calib(pred_tr, tr["y"]),
            val_calibration=calib(pred_va, va["y"]),
            train_mean_pred=round(float(pred_tr.mean()), 4),
            val_mean_pred=round(float(pred_va.mean()), 4),
            train_mean_actual=round(float(tr["y"].mean()), 4),
            val_mean_actual=round(float(va["y"].mean()), 4),
            note=("predicted vs actual per-part IoU by predicted-score quintile; the gap "
                  "between the two mean_actual columns at matched predicted scores is the "
                  "prior-quality shift"),
        )
        # ---- (c) per-feature drift ----------------------------------------
        drift = {}
        for i, name in enumerate(FEATS):
            a, b = tr["X"][:, i], va["X"][:, i]
            qa = np.quantile(a, [0.1, 0.5, 0.9])
            qb = np.quantile(b, [0.1, 0.5, 0.9])
            drift[name] = dict(train_medians=[round(float(v), 4) for v in qa],
                               val_medians=[round(float(v), 4) for v in qb],
                               median_shift=round(float(qb[1] - qa[1]), 4))
        out["feature_drift"] = drift
        # ---- (d) image-level ----------------------------------------------
        trm = record_meta(TRAIN)
        vam = record_meta(VAL)
        def _median_masks(meta, keep_rids):
            vals = [meta[r]["n_masks"] for r in keep_rids
                    if r in meta and meta[r].get("n_masks")]
            return float(np.median(vals)) if vals else 0.0
        out["image_level"] = dict(
            train_masks_per_image_median=_median_masks(
                trm, [v["rid"] for v in tr_t]),
            val_masks_per_image_median=_median_masks(
                vam, [v["rid"] for v in va_t]),
            train_no_target_share=round(float(np.mean([1.0 if m["no_target"] else 0.0
                                                       for m in trm.values()])), 4),
            val_no_target_share=round(float(np.mean([1.0 if m["no_target"] else 0.0
                                                     for m in vam.values()])), 4),
        )
        doc["modes"]["characterize"] = out
        (OUTF / "transfer_wall.json").write_text(json.dumps(doc, indent=2))
        print("characterize written:", json.dumps(out["population"], indent=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
