#!/usr/bin/env python
"""EXP-169 — no-detector gRefCOCO rescue arms (directive 2026-10-17).

The training-free no-detector path measures cIoU 38.71 (floor 44.4) with
noun-phrase boxes.  This script runs the untested rescue arms on the full
official val split, all WITHOUT GroundingDINO (no detector boxes, no
detector-box-prompted SAM masks), under the same abstention:

  base_noun            replicate exp163 (noun-phrase terms, sxi pick in box)
  full_clause          per-clause FULL-text boxes (method_gres.clauses), sxi
                       pick in each clause box, union  [cache: exp168a]
  full_clause_presence same, but drop clauses whose existence probe (on the
                       clause text, exists namespace) answers "no"
  pair_union           the pair/union candidate family (learned pair selector
                       argmax over build_candidates) as the primary picker
  learned_head_noun    the learned selector head (same 8 terms as the
                       RefCOCO-family head), fitted on the gRefCOCO train
                       tune rows (declared trained), picking within the
                       noun-phrase box per term

    python exp/exp169_gref_rescue.py --out exp/out_final/gref_rescue.json
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
from cache_map import cache_file  # noqa: E402
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from method_gres import clauses  # noqa: E402
import exp159_head_rebuild as e159  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")


def base_box_path(fn, phrase):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{phrase.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def load_box(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("box")


def load_fg_masks(cache_path, fstore, mstore):
    """Full-resolution (masks, normalized fg) from the record's own npz."""
    if cache_path in mstore:
        return mstore[cache_path]
    p = Path(cache_path)
    if not p.is_file():
        mstore[cache_path] = None
        return None
    with np.load(p, allow_pickle=False) as z:
        shape = tuple(int(x) for x in z["mask_shape"])
        masks = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                              ).reshape(shape).astype(bool)
        fg = z["foreground"].astype(np.float32)
    fg /= np.linalg.norm(fg, axis=1, keepdims=True) + 1e-9
    mstore[cache_path] = (masks, fg)
    while len(mstore) > 24:
        mstore.pop(next(iter(mstore)))
    return mstore[cache_path]


def pick_masks_sxi(r, F, topk, margin, cfg):
    score = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
    order = np.argsort(score)[::-1]
    js = [int(order[0])]
    if topk == 2 and len(order) > 1 and score[order[1]] >= margin * score[order[0]]:
        js.append(int(order[1]))
    return [r["masks"][j] for j in js]


def union_of(parts):
    if not parts:
        return None
    pred = parts[0].copy()
    for x in parts[1:]:
        pred |= x
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_14229.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_rescue.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    cfg = json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"]
    cfg.setdefault("cap", 0.5)
    alpha, beta = cfg["alpha"], cfg["beta"]
    topk, margin = cfg["topk"], cfg["margin"]

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"tune rows {len(tr)}; base+pair fitted ({time.time()-t0:.0f}s)", flush=True)

    # ---- learned head fitted on the gRefCOCO train tune rows --------------
    te = e159.TextEncoder(args.device)
    from sklearn.ensemble import HistGradientBoostingClassifier
    fstore, mstore = {}, {}
    tune_recs = [json.loads(l) for l in
                 open("exp/out_extreme/records_grefcoco_train_tune.jsonl")]
    tune_terms = [terms_of(r["parsed"]) for r in tune_recs]
    te.batch([t for ts in tune_terms for t in ts])
    Fs, labels, n_terms = [], [], 0
    for rr, terms in zip(tune_recs, tune_terms):
        if rr["no_target"]:
            continue
        loaded = load_fg_masks(rr["cache_file"], fstore, mstore)
        if loaded is None:
            continue
        masks, fg = loaded
        if len(masks) != len(rr["gt_iou"]):
            continue
        for term in terms:
            box = load_box(lpath(rr["file_name"], term))
            if box is None:
                continue
            our = fg @ te(term)
            feats = e159.arm_c_features(masks, fg, rr, box, our)
            Fs.append(feats)
            labels.append((np.asarray(rr["gt_iou"][:len(masks)]) >= 0.5).astype(np.int8))
            n_terms += 1
    y = np.concatenate(labels)
    head = None
    if n_terms >= 20:
        X = np.concatenate([e159.feat_matrix(F, e159.ALL_COLS) for F in Fs])
        head = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                              learning_rate=0.08, random_state=0)
        head.fit(X, y)
        print(f"learned head fitted: {n_terms} tune terms, {len(y)} proposals, "
              f"pos {y.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"WARNING: only {n_terms} tune terms; learned head skipped", flush=True)

    # ---- val pass ---------------------------------------------------------
    val = load_rows(args.records, "val", clf_base)
    if args.limit:
        val = val[: args.limit]
    if args.nshards > 1:
        val = val[args.shard::args.nshards]
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s); per-record alignment", flush=True)

    arms = ("base_noun", "full_clause", "full_clause_presence",
            "pair_union", "learned_head_noun")
    acc = {a: dict(iou=0.0, I=0.0, U=0.0) for a in arms}
    per = {a: [] for a in arms}
    tp = fn_c = tn = fp = 0
    n_used = 0
    stats = dict(clause_missing=0, clause_present_dropped=0, head_picks=0,
                 head_terms_skipped=0)
    p_sel = g8["p_sel"]
    t1 = time.time()
    for k, r in enumerate(val):
        rr = r.get("rec")
        fn = rr["file_name"] if rr else None
        clauses_ = clauses(rr["parsed"]["sentence"]) if rr else []
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        if abstain:
            per_rows = dict(no_target=bool(r["no_target"]), abstain=True, I=0.0,
                            U=0.0 if r["no_target"] else float(r["gt_area"]),
                            iou=1.0 if r["no_target"] else 0.0,
                            gt_area=float(r["gt_area"]))
            for a in arms:
                per[a].append(dict(per_rows))
            if r["no_target"]:
                tp += 1
            else:
                fp += 1
            n_used += 1
            continue

        def fallback_pred():
            if r["no_target"]:
                single = [kk for kk, c in enumerate(ids) if c[1] < 0]
                kk = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
            else:
                kk = int(np.argmax(pp))
            return candidate_mask(r["masks"], ids[kk])

        picks = {}
        # base noun terms
        parts = []
        for term in (terms_of(rr["parsed"]) if rr else []):
            box = load_box(lpath(fn, term))
            if box is None:
                continue
            Fb, _ = box_feats(r["masks"], r["_bs"], box)
            if Fb is not None:
                parts.extend(pick_masks_sxi(r, Fb, topk, margin,
                                            dict(alpha=alpha, beta=beta)))
        picks["base_noun"] = union_of(parts)
        # full-clause boxes
        parts = []
        parts_p = []
        for c in clauses_:
            box = load_box(base_box_path(fn, c))
            if box is None:
                stats["clause_missing"] += 1
                continue
            Fb, _ = box_feats(r["masks"], r["_bs"], box)
            if Fb is None:
                continue
            cl = pick_masks_sxi(r, Fb, topk, margin, dict(alpha=alpha, beta=beta))
            parts.extend(cl)
            absent = probe_no("exists", fn, c)
            if absent is True:
                stats["clause_present_dropped"] += 1
            else:
                parts_p.extend(cl)
        picks["full_clause"] = union_of(parts)
        picks["full_clause_presence"] = union_of(parts_p)
        # pair/union family
        picks["pair_union"] = candidate_mask(r["masks"], ids[int(np.argmax(pp))])
        # learned head within noun-phrase boxes (full-resolution features,
        # exactly as in the tune fit)
        loaded = load_fg_masks(rr["cache_file"], fstore, mstore) if rr else None
        if head is not None and loaded is not None and rr:
            masks_full, fg_full = loaded
            terms = terms_of(rr["parsed"])
            te.batch(terms)
            hparts = []
            for term in terms:
                box = load_box(lpath(fn, term))
                if box is None:
                    stats["head_terms_skipped"] += 1
                    continue
                our = fg_full @ te(term)
                feats = e159.arm_c_features(masks_full, fg_full, rr, box, our)
                proba = head.predict_proba(
                    e159.feat_matrix(feats, e159.ALL_COLS))[:, 1]
                cand = np.flatnonzero(feats["loc"] > 0)
                if len(cand) == 0:
                    continue
                j = int(cand[int(np.argmax(proba[cand]))])
                hparts.append(masks_full[j][::2, ::2])  # match the row frame
                stats["head_picks"] += 1
            picks["learned_head_noun"] = union_of(hparts)
        else:
            picks["learned_head_noun"] = None

        gt = r["gt"]
        for a in arms:
            pred = picks.get(a)
            if pred is None:
                pred = fallback_pred()
            I = float(np.logical_and(pred, gt).sum())
            U = float(np.logical_or(pred, gt).sum())
            per[a].append(dict(no_target=bool(r["no_target"]), abstain=False,
                               I=I, U=U,
                               iou=0.0 if r["no_target"] else (I / U if U else 0.0),
                               gt_area=float(r["gt_area"])))
        # abstention accounting is arm-independent: use the base row semantics
        if r["no_target"]:
            fn_c += 1
        else:
            tn += 1
        n_used += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(val)} ({time.time()-t1:.0f}s)", flush=True)

    def metrics(rows):
        ci_I = sum(x["I"] for x in rows)
        ci_U = sum(x["U"] for x in rows)
        gi = sum(x["iou"] for x in rows)
        n_nt_ab = 0
        n_nt = 0
        for x in rows:
            if x["no_target"]:
                n_nt += 1
                if x["abstain"]:
                    n_nt_ab += 1
        return dict(cIoU=100.0 * ci_I / ci_U if ci_U else 0.0,
                    gIoU=100.0 * gi / len(rows),
                    N_acc=100.0 * n_nt_ab / n_nt if n_nt else 0.0,
                    n=len(rows),
                    _raw=dict(I=ci_I, U=ci_U, iou=gi, n=len(rows),
                              n_nt=n_nt, n_nt_ab=n_nt_ab))

    res = {a: metrics(per[a]) for a in arms}
    # T_acc: fraction of predicted-targeted rows that were actually targeted is
    # not directly available per arm; the abstention is arm-independent, so use
    # the exp163 convention (tn/(tn+fp)) from the shared abstention stats.
    res_all = dict(n=n_used, arms=res, abstention_stats=dict(TP=tp, FP=fp),
                   stats=stats, shard=args.shard, nshards=args.nshards,
                   generated="exp169_gref_rescue.py",
                   note=("no GroundingDINO anywhere; per-arm picks share the "
                         "abstention (arm-independent); T_acc/N_acc follow the "
                         "exp163/common declaration semantics"))
    for a in arms:
        print(f"  {a:24s} cIoU {res[a]['cIoU']:7.3f}  gIoU {res[a]['gIoU']:7.3f}  "
              f"N-acc {res[a]['N_acc']:6.2f}")
    args.out.write_text(json.dumps(res_all, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
