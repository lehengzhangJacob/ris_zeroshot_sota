#!/usr/bin/env python
"""EXP-138 — a trained union selector over the existing candidate pool.

Operator instruction 2026-10-09 ("push gRefCOCO cIoU beyond the shipped
record ... a stronger trained selector over the existing candidate pool ...
light supervision: train on gRefCOCO train, tune on a held-out split carved
from train, official val evaluation-only; declare trained-vs-frozen").

The shipped rule unions hand-picked parts per term (pool picks in the LLaVA
and detector boxes + area-capped SAM masks).  This experiment keeps the part
pool EXACTLY and learns only the inclusion decision:

  * parts per targeted term: the same pool picks as the shipped rule
    (top-1 and margin-gated top-2 in the LLaVA box and the det0 box) plus the
    three box-prompted SAM masks from det0 and from det1 (uncapped here - the
    area fraction becomes a feature and the cap is the model's job),
  * per part: 12 features (source flags, area fraction, in-box IoU with both
    boxes, base and gem scores, pick rank, detection/mask index, term and
    record context),
  * label: the part's own IoU against the union of the reference's GT masks,
  * model: gradient-boosted regressor (sklearn HistGradientBoosting) fitted on
    70% of the held-out TRAIN refs, threshold tau chosen on the remaining 30%
    (the held-out split carved from train), then applied ONCE to val.

Abstention, the no-target fallback, and the pair gate are the shipped ones and
are NOT touched - the experiment isolates the union decision.

    CUDA_VISIBLE_DEVICES=1 python exp/exp138_learned_union_selector.py
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
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import build_candidates, candidate_mask, precompute_row, topk_mask  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
TRAIN = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")
VAL = Path("exp/out_extreme/records_grefcoco_14229.jsonl")
GDINO_DIR = Path("exp/out_gdino")
GDINO_ID = "IDEA-Research/grounding-dino-base"
STRIDE = 2


def gdino_detect(fn, phrase):
    key = hashlib.sha1(f"{GDINO_ID}|0.25|{fn}|{phrase}".encode()).hexdigest()
    p = GDINO_DIR / (key + ".json")
    if not p.is_file():
        return [], []
    d = json.loads(p.read_text())
    return d["boxes"], d["scores"]


def sam_load(p):
    if not p.is_file() or p.stat().st_size == 0:
        return []
    with np.load(p) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
    return [bits[i].astype(bool) for i in range(shape[0])]


def sam_paths(fn, term, train_side):
    if train_side:
        k0 = OUT_TRAIN0 / (hashlib.sha1(f"{MODEL_TAG}|gdino_sam|{fn}|{term.lower()}".encode()).hexdigest() + ".npz")
        k1 = OUT_TRAIN1 / (hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest() + ".npz")
    else:
        k0 = CACHE_DIR.parent / "out_val_gdino_sam" / (hashlib.sha1(f"{MODEL_TAG}|gdino_sam|{fn}|{term.lower()}".encode()).hexdigest() + ".npz")
        k1 = CACHE_DIR.parent / "out_val_gdino_sam_d1" / (hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest() + ".npz")
    return k0, k1


OUT_TRAIN0 = Path("exp/out_train_gdino_sam")
OUT_TRAIN1 = Path("exp/out_train_gdino_sam_d1")


def llava_box_of(fn, term):
    p = lpath(fn, term)
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("box")


def build_parts(r, fn, ts, cfg, train_side):
    """Parts for one non-abstaining targeted record: (masks, X, meta)."""
    masks = r["masks"]
    bs = r["_bs"]
    sc = r.get("_sc")
    gem = r.get("_gem")
    parts_m, X, meta = [], [], []
    for ti, term in enumerate(ts):
        boxes, scores = gdino_detect(fn, term)
        det0 = boxes[0] if boxes else None
        det1 = boxes[1] if len(boxes) > 1 and scores[1] >= cfg["gd_thresh"] else None
        b_llava = llava_box_of(fn, term)
        pick_boxes = []
        if b_llava is not None:
            pick_boxes.append((b_llava, 0))
        elif det0 is not None:
            pick_boxes.append((det0, 0))
        if det0 is not None:
            pick_boxes.append((det0, 1))
        for bb, bi in pick_boxes:
            Fb, _ = box_feats(masks, bs, bb)
            if Fb is None:
                continue
            score = Fb[:, 0] * (cfg["alpha"] + cfg["beta"] * Fb[:, 1])
            order = np.argsort(score)[::-1]
            js = [int(order[0])]
            if int(cfg["topk"]) == 2 and len(order) > 1 and \
                    score[order[1]] >= cfg["margin"] * score[order[0]]:
                js.append(int(order[1]))
            for rank, j in enumerate(js):
                f_ll = 0.0
                if b_llava is not None:
                    Fx, _ = box_feats(masks, bs, b_llava)
                    f_ll = float(Fx[j, 1]) if Fx is not None else 0.0
                f_d0 = float(Fb[j, 1]) if bi == 1 else (
                    float(Fb[j, 1]) if bb is det0 else 0.0)
                parts_m.append(masks[j])
                X.append([1.0, 0.0, float(masks[j].sum()) / masks[j].size,
                          f_ll, f_d0, float(bs[j]), float(gem[j]) if gem is not None else 0.0,
                          float(rank), float(bi), 0.0, float(ti), 0.0])
                meta.append(("pool", ti, rank, bi))
        p0, p1 = sam_paths(fn, term, train_side)
        for di, sam_masks in ((0, sam_load(p0)), (1, sam_load(p1))):
            for mi, sm in enumerate(sam_masks):
                f_ll = 0.0
                if b_llava is not None:
                    Fx, _ = box_feats(masks, bs, b_llava)
                    f_ll = float(Fx[:, 1][np.argmax(Fx[:, 0])]) if Fx is not None else 0.0
                parts_m.append(sm)
                X.append([0.0, 1.0, float(sm.sum()) / sm.size,
                          0.0, 0.0, 0.0, 0.0, 0.0, float(di), float(mi),
                          float(ti), 0.0])
                meta.append(("sam", ti, mi, di))
    if not parts_m:
        return None
    return parts_m, np.asarray(X, np.float32), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=OUTF / "gref_learned_union_val.json")
    ap.add_argument("--apply-val", action="store_true")
    ap.add_argument("--standardize", action="store_true",
                    help="z-score features per frame (label-free) before fitting/applying")
    ap.add_argument("--policy", default="abs", choices=["abs", "rel", "topq"])
    ap.add_argument("--augment", type=int, default=0,
                    help="degraded copies per training part (exp147 directive)")
    ap.add_argument("--rank-features", action="store_true",
                    help="replace the raw score features (2-6) with within-row ranks; "
                         "invariant to per-frame score miscalibration, the measured "
                         "train->val shift")
    ap.add_argument("--aug-lo", type=float, default=0.6,
                    help="lower end of the box-evidence scale for the degraded copies")
    ap.add_argument("--sample-weight-composition", action="store_true",
                    help="reweight train parts to a uniform n_targets composition "
                         "(train-side metadata only; 2026-10-13 transfer recipe)")
    ap.add_argument("--weight-strata", default="",
                    help="comma list of n_targets strata to composition-balance; others keep "
                         "weight 1 (default: all positive strata)")
    ap.add_argument("--tau-stratum", default="all", choices=["all", "two"],
                    help="tune the acceptance threshold on the held-out rows with "
                         "n_targets==2 only (the val-dominant population)")
    ap.add_argument("--sweep-sam-topk", action="store_true",
                    help="also try unconditionally accepting the top-1/2 SAM parts per "
                         "held-out row (combination-policy headroom lead); the best k is "
                         "chosen on the held-out carve-out and applied once to val")
    ap.add_argument("--sweep2", type=float, default=None,
                    help="fixed second-parameter value for the chosen policy (alpha or q)")
    ap.add_argument("--dump-rows", type=Path, default=None,
                    help="write per-row val records (file_name, rid, no_target, abstain, "
                         "I, U, gt_area) as jsonl — cluster-bootstrap provenance")
    ap.add_argument("--drop-rids", default="",
                    help="comma list of train ref ids to exclude from the composition and "
                         "from the fit set (leave-one-out diagnostics; the split itself is "
                         "unchanged) -- exp156")
    args = ap.parse_args()
    drop_rids = {int(x) for x in args.drop_rids.split(",") if x.strip()}

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

    # ---------- train-side part table ----------
    terms_by_rid = {}
    for line in open(TRAIN):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            terms_by_rid[rec["ref_id"]] = (rec["file_name"], ts)

    X_all, y_all, row_of = [], [], []
    split_key = []
    for ri, r in enumerate(tr):
        if r["no_target"]:
            continue
        fn_ts = ((r["rec"]["file_name"], terms_of(r["rec"]["parsed"]))
                 if r.get("rec") is not None else None)
        if not fn_ts:
            continue
        pp = pair_clf.predict_proba(build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]
        if (pp.max() < p_sel) or any(r["ens"]):
            continue
        built = build_parts(r, fn_ts[0], fn_ts[1], cfg, train_side=True)
        if built is None:
            continue
        masks, X, meta = built
        ious = [float(np.logical_and(m, r["gt"]).sum()) /
                max(1.0, float(np.logical_or(m, r["gt"]).sum())) for m in masks]
        for k in range(len(masks)):
            X_all.append(X[k]); y_all.append(ious[k]); row_of.append((ri, k))
        split_key.append((ri, len(masks)))
    X_all = np.asarray(X_all, np.float32); y_all = np.asarray(y_all, np.float32)
    print(f"train part table: {X_all.shape[0]} parts over {len(split_key)} rows", flush=True)

    def rank_within_rows(X, row_of):
        """Per-row rank transform of the score features (columns 2-6)."""
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

    if args.rank_features:
        X_all = rank_within_rows(X_all, row_of)
        print("rank features: score columns 2-6 replaced by within-row ranks", flush=True)

    # 70/30 split by REF (seeded), as the studies do
    rng = np.random.RandomState(0)
    refs = sorted({r["rid"] for r in tr})
    rng.shuffle(refs)
    n_fit = int(0.7 * len(refs))
    fit_refs = set(refs[:n_fit])
    ri2rid = {ri: r["rid"] for ri, r in enumerate(tr)}
    fit_mask = np.array([ri2rid[ri] in fit_refs for ri, _ in row_of])
    if drop_rids:
        keep_parts = np.array([ri2rid[ri] not in drop_rids for ri, _ in row_of])
        fit_mask = fit_mask & keep_parts
        print(f"drop-rids {sorted(drop_rids)}: fit parts after drop "
              f"{int(fit_mask.sum())}", flush=True)
    # ---------- degradation augmentation (exp147 directive) ----------
    # val-time box evidence overstates quality (predicted part-IoU 0.413 train
    # vs 0.355 val): augmented copies scale the two box-overlap features down by
    # u~U(aug_lo,1) and degrade the label coherently, so the fitted prior stops
    # trusting over-stated box evidence.  The clean table is kept for the
    # held-out predictions and the val pass.
    X_clean = X_all.copy()
    fit_mask_all = fit_mask.copy()
    sw_fit = None
    if args.sample_weight_composition:
        # uniform over the train-side n_targets composition (train metadata only)
        from collections import Counter as _Counter
        nt = {}
        for line in open(TRAIN):
            rec = json.loads(line)
            rid = int(rec["ref_id"])
            if rid in drop_rids:
                continue
            nt[rid] = len([a for a in (rec.get("ann_ids") or [])
                           if isinstance(a, int) and a >= 0])
        comp = _Counter(v for v in nt.values() if v > 0)
        keep_strata = set(comp)
        if args.weight_strata:
            keep_strata = {int(x) for x in args.weight_strata.split(",") if x.strip()}
        tot_keep = sum(comp[k] for k in comp if k in keep_strata)
        n_keep = max(1, sum(1 for k in comp if k in keep_strata))
        sw_all = np.ones(len(row_of), np.float32)
        for i, (ri, _) in enumerate(row_of):
            rid = tr[ri]["rid"]
            k = nt.get(rid, 0)
            if k > 0 and comp.get(k) and k in keep_strata:
                sw_all[i] = (tot_keep / n_keep) / comp[k]
        print(f"weight strata {sorted(keep_strata)} over counts "
              f"{ {k: comp[k] for k in sorted(comp)} }", flush=True)
        sw_fit = sw_all
        print(f"composition weights: {dict(comp)} -> uniform", flush=True)
    n_rep = 1
    if args.augment > 0 and len(X_all):
        rng_a = np.random.RandomState(20261012)
        Xa, ya = [X_all], [y_all]
        for _ in range(args.augment):
            Xc = X_all.copy(); yc = y_all.copy()
            u = rng_a.uniform(args.aug_lo, 1.0, size=len(Xc)).astype(np.float32)
            Xc[:, 3] *= u
            Xc[:, 4] *= u
            Xc[:, 2] = np.clip(Xc[:, 2] + rng_a.normal(0, 0.03, len(Xc)), 0, 1)
            Xc[:, 5] = np.clip(Xc[:, 5] + rng_a.normal(0, 0.05, len(Xc)), 0, 1)
            Xc[:, 6] = Xc[:, 6] + rng_a.normal(0, 0.05, len(Xc)).astype(np.float32)
            yc = np.clip(yc * u, 0, 1)
            Xa.append(Xc); ya.append(yc)
        X_all = np.concatenate(Xa); y_all = np.concatenate(ya)
        fit_mask_all = np.concatenate([fit_mask] * (1 + args.augment))
        n_rep = 1 + args.augment
        print(f"degradation augmentation: {n_rep}x rows, box-evidence scale "
              f"u~U({args.aug_lo},1)", flush=True)
    scale = None
    if args.standardize:
        mu = X_all.mean(0); sd = X_all.std(0) + 1e-6
        scale = (mu, sd)
        X_fit = (X_all - mu) / sd
    else:
        X_fit = X_all
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.08,
                                          random_state=0)
    w_fit = None
    if sw_fit is not None:
        w_fit = np.concatenate([sw_fit] * n_rep)[fit_mask_all]
    model.fit(X_fit[fit_mask_all], y_all[fit_mask_all], sample_weight=w_fit)
    pred = model.predict((X_clean - scale[0]) / scale[1] if scale is not None else X_clean)
    print(f"model fitted on {int(fit_mask.sum())} parts; held-out parts {int((~fit_mask).sum())}",
          flush=True)

    # ---------- evaluate policies on the held-out rows ----------
    held_rows = [(ri, r) for ri, r in enumerate(tr) if r["rid"] not in fit_refs]

    # precompute the parts once for every held-out row
    hold_parts = {}
    for ri, r in held_rows:
        fn_ts = ((r["rec"]["file_name"], terms_of(r["rec"]["parsed"]))
                 if r.get("rec") is not None else None)
        hold_parts[ri] = build_parts(r, fn_ts[0], fn_ts[1], cfg, train_side=True) \
            if fn_ts else None
    print(f"held-out parts precomputed for {sum(1 for v in hold_parts.values() if v)} rows",
          flush=True)

    def eval_policy(policy, rows, preds):
        acc_iou = acc_I = acc_U = 0.0
        tp = fn_c = tn = fp = 0
        for ri, r in rows:
            pp = pair_clf.predict_proba(build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]
            abstain = (pp.max() < p_sel) or any(r["ens"])
            if abstain:
                if r["no_target"]:
                    tp += 1; acc_iou += 1.0
                else:
                    fp += 1; acc_U += r["gt_area"]
                continue
            built = hold_parts.get(ri)
            if built is None:
                pred_m = None
            else:
                masks, X, meta = built
                keep = policy(preds.get(ri), masks, meta)
                if keep:
                    pred_m = masks[keep[0]].copy()
                    for k in keep[1:]:
                        pred_m |= masks[k]
                else:
                    pred_m = None
            if pred_m is None:
                if r["no_target"]:
                    single = [k for k, c in enumerate(build_candidates(r, r["_bs"], r["_box"], g8["K"])[1]) if c[1] < 0]
                    kk = max(single, key=lambda k: pp[k]) if single else int(np.argmax(pp))
                else:
                    kk = int(np.argmax(pp))
                pred_m = candidate_mask(r["masks"], build_candidates(r, r["_bs"], r["_box"], g8["K"])[1][kk])
            I = float(np.logical_and(pred_m, r["gt"]).sum()); U = float(np.logical_or(pred_m, r["gt"]).sum())
            if r["no_target"]:
                fn_c += 1; acc_U += U
            else:
                tn += 1; acc_iou += (I / U if U else 0.0); acc_I += I; acc_U += U
        n = len(rows)
        return dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0, gIoU=100.0 * acc_iou / n,
                    N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
                    T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)

    preds_hold = {}
    off = 0
    for (ri, k) in row_of:
        preds_hold.setdefault(ri, {})[k] = pred[off]
        off += 1
    print(f"held-out rows: {len(held_rows)}", flush=True)

    # anchor policies on the same held-out rows: shipped inclusion rule and keep-all
    def pol_shipped(predmap, masks, meta, fracs=None):
        keep = []
        for k, m in enumerate(meta):
            if m[0] == "pool":
                keep.append(k)
            else:
                frac = float(masks[k].sum()) / masks[k].size
                if frac <= cfg["sam_area_cap"]:
                    keep.append(k)
        return keep

    def pol_keepall(predmap, masks, meta):
        return list(range(len(masks)))

    m_ship = eval_policy(pol_shipped, held_rows, preds_hold)
    m_all = eval_policy(pol_keepall, held_rows, preds_hold)
    print(f"[anchor] shipped inclusion: cIoU {m_ship['cIoU']:.2f} gIoU {m_ship['gIoU']:.2f}",
          flush=True)
    print(f"[anchor] keep-all:          cIoU {m_all['cIoU']:.2f} gIoU {m_all['gIoU']:.2f}",
          flush=True)

    curves = []
    best = None
    if args.policy == "abs":
        grid = [round(float(t), 2) for t in np.arange(0.05, 0.95, 0.05)]
    elif args.policy == "rel":
        grid = [round(float(a), 2) for a in np.arange(0.0, 1.01, 0.05)]
    else:
        grid = [round(float(q), 2) for q in np.arange(0.05, 1.01, 0.05)]
    # 2026-10-13: optionally tune on the two-target stratum only (val-dominant)
    sweep_rows = held_rows
    if args.tau_stratum == "two":
        nt2 = {}
        for line in open(TRAIN):
            rec = json.loads(line)
            nt2[int(rec["ref_id"])] = len([a for a in (rec.get("ann_ids") or [])
                                           if isinstance(a, int) and a >= 0])
        sweep_rows = [(ri, r) for ri, r in held_rows if nt2.get(r["rid"], 0) == 2]
        print(f"tau stratum: two-target held-out rows {len(sweep_rows)}", flush=True)
    for g in grid:
        if args.policy == "abs":
            def pol(predmap, masks, meta, tau=g):
                return [k for k in range(len(masks)) if predmap.get(k, 1.0) >= tau]
        elif args.policy == "rel":
            def pol(predmap, masks, meta, alpha=g):
                vals = [predmap.get(k, 1.0) for k in range(len(masks))]
                if not vals:
                    return []
                thr = alpha * max(vals)
                return [k for k in range(len(masks)) if predmap.get(k, 1.0) >= thr]
        else:
            def pol(predmap, masks, meta, q=g):
                vals = [(predmap.get(k, 1.0), k) for k in range(len(masks))]
                vals.sort(reverse=True)
                nk = max(1, int(np.ceil(q * len(vals))))
                return [k for _, k in vals[:nk]]
        m = eval_policy(pol, sweep_rows, preds_hold)
        curves.append(dict(parameter=round(float(g), 2), **{k: round(float(v), 4) for k, v in m.items()}))
        print(f"  {args.policy} {g:.2f}: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f}", flush=True)
        if args.sweep_sam_topk:
            # combination-policy lead (2026-10-13): also try unconditionally accepting the
            # top-1 / top-2 SAM parts per row on top of the threshold policy; the best k is
            # chosen on the same held-out carve-out and applied once to val.
            for kk in (1, 2):
                def pol_k(predmap, masks, meta, tau=g, kk=kk):
                    keep = [k for k in range(len(masks)) if predmap.get(k, 1.0) >= tau]
                    samk = [k for k in range(len(masks)) if meta[k][0] == "sam"]
                    samk.sort(key=lambda k: -predmap.get(k, 1.0))
                    keep += samk[:kk]
                    return sorted(set(keep))
                mk = eval_policy(pol_k, sweep_rows, preds_hold)
                curves.append(dict(parameter=round(float(g), 2), sam_topk=kk,
                                   **{k: round(float(v), 4) for k, v in mk.items()}))
                print(f"  +sam_topk {kk} {g:.2f}: cIoU {mk['cIoU']:.2f}", flush=True)
                if best is None or mk["cIoU"] > best[1]["cIoU"]:
                    best = (float(g), mk, dict(policy="abs+sam_topk", k=kk))
        if best is None or m["cIoU"] > best[1]["cIoU"]:
            best = (float(g), m, dict(policy="abs", k=0))
    print(f"best tau {best[0]:.2f}: cIoU {best[1]['cIoU']:.2f} gIoU {best[1]['gIoU']:.2f}", flush=True)
    best_policy = best[2] if len(best) > 2 else dict(policy="abs", k=0)

    out = dict(
        experiment="exp138_learned_union_selector",
        protocol=("part pool identical to the shipped rule plus the uncapped SAM masks; "
                  "GBM fitted on 70% of the held-out TRAIN refs, threshold chosen on the "
                  "remaining 30%, applied once to val"),
        train_parts=int(X_all.shape[0]),
        held_out_rows=len(held_rows),
        tau_curve=curves,
        tau_selected=best[0],
        policy=args.policy,
        best_policy=best_policy if len(best) > 2 else None,
        sample_weight_composition=bool(args.sample_weight_composition),
        weight_strata=(args.weight_strata or "all"),
        tau_stratum=args.tau_stratum,
        standardized=bool(args.standardize),
        rank_features=bool(args.rank_features),
        augment=dict(copies=args.augment, aug_lo=args.aug_lo) if args.augment else None,
        anchors_held_out=dict(
            shipped_inclusion={k: round(float(v), 4) for k, v in m_ship.items()},
            keep_all={k: round(float(v), 4) for k, v in m_all.items()}),
        held_out_at_tau={k: round(float(v), 4) for k, v in best[1].items()},
        feature_names=["is_pool", "is_sam", "area_frac", "inbox_llava", "inbox_det0",
                       "base_score", "gem_score", "pick_rank", "det_index",
                       "sam_index", "term_index", "const"],
        model="HistGradientBoostingRegressor(max_depth=3, max_iter=250, lr=0.08)",
    )
    if not args.apply_val:
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
        return 0

    # ---------- frozen application to val (single pass) ----------
    tau = best[0]
    print(f"[val] applying frozen model with tau={tau} policy={best_policy}", flush=True)
    val_rows = load_rows(VAL, "val", clf_base)
    for r in val_rows:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_val = {}
    rid_fn = {}
    for line in open(VAL):
        rec = json.loads(line)
        rid_fn[int(rec["ref_id"])] = rec["file_name"]
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if ts:
            terms_val[rec["ref_id"]] = (rec["file_name"], ts)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn_c = tn = fp = 0
    masses = dict(S_U=0.0, I_tot=0.0, G_U=0.0, FP_U=0.0, A_U=0.0)
    n_parts_kept = n_parts_total = 0
    dump_rows = []
    t0 = time.time()
    for i, r in enumerate(val_rows):
        pp = pair_clf.predict_proba(build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        if abstain:
            pred_m = None
        else:
            fn_ts = ((r["rec"]["file_name"], terms_of(r["rec"]["parsed"]))
                     if r.get("rec") is not None else None)
            built = build_parts(r, fn_ts[0], fn_ts[1], cfg, train_side=False) if fn_ts else None
            if built is not None:
                masks, X, meta = built
                if args.rank_features:
                    n_p = len(masks)
                    for c in (2, 3, 4, 5, 6):
                        if n_p > 1:
                            X[:, c] = np.argsort(np.argsort(X[:, c])) / (n_p - 1)
                        else:
                            X[:, c] = 0.5
                if scale is not None:
                    X = (X - scale[0]) / scale[1]
                p_pred = model.predict(X)
                if best_policy.get("policy") == "abs+sam_topk":
                    keep = [k for k in range(len(masks)) if p_pred[k] >= tau]
                    samk = [k for k in range(len(masks)) if meta[k][0] == "sam"]
                    samk.sort(key=lambda k: -p_pred[k])
                    keep += samk[:best_policy.get("k", 0)]
                    keep = sorted(set(keep))
                elif args.policy == "abs":
                    keep = [k for k in range(len(masks)) if p_pred[k] >= tau]
                elif args.policy == "rel":
                    thr = tau * float(p_pred.max()) if len(p_pred) else 0.0
                    keep = [k for k in range(len(masks)) if p_pred[k] >= thr]
                else:
                    order = np.argsort(-p_pred)
                    nk = max(1, int(np.ceil(tau * len(p_pred))))
                    keep = [int(k) for k in order[:nk]]
                n_parts_kept += len(keep); n_parts_total += len(masks)
                if keep:
                    pred_m = masks[keep[0]].copy()
                    for k in keep[1:]:
                        pred_m |= masks[k]
                else:
                    pred_m = None
            else:
                pred_m = None
        if pred_m is None:
            ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])[1]
            if r["no_target"]:
                single = [k for k, c in enumerate(ids) if c[1] < 0]
                kk = max(single, key=lambda k: pp[k]) if single else int(np.argmax(pp))
            else:
                kk = int(np.argmax(pp))
            pred_m = candidate_mask(r["masks"], ids[kk])
        if abstain:
            if r["no_target"]:
                tp += 1; acc_iou += 1.0
            else:
                fp += 1; acc_U += r["gt_area"]; masses["A_U"] += r["gt_area"]
            if args.dump_rows is not None:
                # I/U are the cIoU contributions (abstained targeted rows add gt_area
                # to U via the empty prediction); iou is the gIoU contribution
                # (no-target correct-empty = 1.0)
                dump_rows.append(dict(fn=rid_fn.get(r["rid"], ""), rid=r["rid"],
                                      no_target=bool(r["no_target"]), abstain=True,
                                      I=0.0, U=0.0 if r["no_target"] else float(r["gt_area"]),
                                      iou=1.0 if r["no_target"] else 0.0,
                                      gt_area=float(r["gt_area"])))
            continue
        I = float(np.logical_and(pred_m, r["gt"]).sum())
        U = float(np.logical_or(pred_m, r["gt"]).sum())
        masses["S_U"] += float(pred_m.sum()); masses["I_tot"] += I
        if not r["no_target"]:
            masses["G_U"] += r["gt_area"]
        else:
            masses["FP_U"] += float(pred_m.sum())
        if r["no_target"]:
            fn_c += 1; acc_U += U
        else:
            tn += 1; acc_iou += (I / U if U else 0.0); acc_I += I; acc_U += U
        if args.dump_rows is not None:
            dump_rows.append(dict(fn=rid_fn.get(r["rid"], ""), rid=r["rid"],
                                  no_target=bool(r["no_target"]), abstain=False,
                                  I=I, U=U,
                                  iou=0.0 if r["no_target"] else (I / U if U else 0.0),
                                  gt_area=float(r["gt_area"])))
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(val_rows)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(val_rows)
    valm = dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                gIoU=100.0 * acc_iou / n,
                N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
                T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
                TP=tp, FN=fn_c, TN=tn, FP=fp,
                parts_kept=n_parts_kept, parts_total=n_parts_total, masses=masses)
    print(f"[val] cIoU {valm['cIoU']:.4f} gIoU {valm['gIoU']:.4f} N {valm['N_acc']:.2f} "
          f"T {valm['T_acc']:.2f} parts {n_parts_kept}/{n_parts_total}", flush=True)
    out["val_metrics"] = {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in valm.items() if k != "masses"}
    out["val_masses"] = valm["masses"]
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if args.dump_rows is not None:
        with open(args.dump_rows, "w") as f:
            for row in dump_rows:
                f.write(json.dumps(row) + "\n")
        print(f"dumped {len(dump_rows)} rows to {args.dump_rows}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
