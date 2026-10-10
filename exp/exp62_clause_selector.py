#!/usr/bin/env python
"""EXP-62 — a LEARNED selector on held-out train (the third model change).

The hand-rule space is exhausted (exp54's 300-trial study picks its best and
still misses).  exp51 measured the headroom: the deployable rule realizes ~69%
of the perfect-box oracle, and at k=50 the oracle over the *deployable*
CLIP+GEM shortlist reaches r=0.88 — so the binding question is whether a
selector FIT on held-out data can get closer to that oracle than any rule.

Per proposal: the four released scores plus their within-record ranks, mask
area, box coverage and box-bbox IoU.  Label: IoU(mask, GT) >= 0.5 (targeted
records only; no-target records contribute only negatives).  Selection is the
union of proposals above an inclusion threshold; abstention is the VLM `none`
answer or max-probability below an abstention threshold.  Model class and both
thresholds are chosen by Optuna on a FIT/CALIB split *inside* held-out train;
val is touched only once, for the final transfer number.

    python exp/exp57_learned_selector.py --trials 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gref_setselect import rank01  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
STRIDE = 2


def box_path(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def exists_path(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|exists|{fn}|{ph.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def exists_says_no(fn, ph):
    """True when the yes/no probe answered no/none; None when not cached."""
    p = exists_path(fn, ph)
    if not p.is_file():
        return None
    raw = (json.loads(p.read_text())["raw"] or "").strip().lower()
    if raw.startswith("y"):
        return False
    if raw.startswith("n"):
        return True
    return None


def load_record(rec, gd, by_id, mask_cache, gt_cache):
    """Per-proposal feature matrix + labels for one record, or None."""
    rid = rec["ref_id"]
    ref = by_id.get(rid)
    if ref is None or not rec.get("cache_file"):
        return None
    ph = (rec["parsed"].get("sentence") or "").strip()
    bp = box_path(rec["file_name"], ph)
    if not bp.is_file():
        return None
    cf = rec["cache_file"]
    if cf not in mask_cache:
        with np.load(cf, allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                               ).reshape(shape).astype(bool)
        mask_cache[cf] = mm[:, ::STRIDE, ::STRIDE]
    if rid not in gt_cache:
        gt_cache[rid] = gd.ground_truth(ref)[::STRIDE, ::STRIDE]
    masks = mask_cache[cf]
    gt = gt_cache[rid]
    n = len(rec["sc"])
    sc = np.asarray(rec["sc"][:n], np.float32)
    scn = np.asarray(rec["sc_noun"][:n], np.float32)
    scs = np.asarray(rec["sc_strip"][:n], np.float32)
    gem = np.asarray(rec["score_gem"][:n], np.float32)
    H, W = masks.shape[1], masks.shape[2]
    area = masks.sum((1, 2)).astype(np.float32) / (H * W)
    box = json.loads(bp.read_text())["box"]
    inb = np.zeros(n, np.float32)
    bx_iou = np.zeros(n, np.float32)
    has_box = 0.0
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(W, -(-x1 // STRIDE)), min(H, -(-y1 // STRIDE))
        if x1 > x0 and y1 > y0:
            has_box = 1.0
            inb = (masks[:, y0:y1, x0:x1].sum((1, 2))
                   / np.maximum(masks.sum((1, 2)), 1)).astype(np.float32)
            box_area = float((y1 - y0) * (x1 - x0))
            inter = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float32)
            bx_iou = inter / (area * H * W + box_area - inter + 1e-9)
    if rec["no_target"]:
        y = np.zeros(n, np.float32)
    else:
        inter = np.logical_and(masks, gt).sum((1, 2)).astype(np.float32)
        union = np.logical_or(masks, gt).sum((1, 2)).astype(np.float32)
        y = (inter / np.maximum(union, 1) >= 0.5).astype(np.float32)
    X = np.stack([sc, scn, scs, gem,
                  rank01(sc), rank01(scn), rank01(scs), rank01(gem),
                  area, inb, bx_iou,
                  np.full(n, has_box, np.float32)], 1).astype(np.float32)
    # per-noun (clause) grounding features, precomputed by exp61
    cpath = Path("exp/out_final/clause_feats") / (Path(cf).stem + "_clause.npz")
    CX = None
    if cpath.is_file():
        with np.load(cpath, allow_pickle=False) as z:
            CX = z["X"].astype(np.float32)
        if len(CX) != len(X):
            CX = None
    if CX is None:      # no parsed nouns (or size mismatch): pad with zeros
        CX = np.zeros((len(X), 5), np.float32)
    X = np.concatenate([X, CX], 1)
    return dict(X=X, y=y, no_target=bool(rec["no_target"]), rid=rid,
                masks=masks, gt=gt, vlm_none=(box is None),
                exists_no=exists_says_no(rec["file_name"], ph),
                gt_area=float(rec["gt_area"]) * gt.size)


def make_clf(kind, params):
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=params["C"], max_iter=2000,
                                  class_weight="balanced")
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=params["n_estimators"],
                                      max_depth=params["max_depth"],
                                      learning_rate=params["learning_rate"],
                                      random_state=0)


def score_rows(rows, clf):
    for r in rows:
        p = clf.predict_proba(r["X"])[:, 1]
        r["scores"] = p
        r["max_p"] = float(p.max())


def evaluate(rows, p_inc, p_abs, vlm_abstain=True, use_exists=False):
    total = len(rows)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    for r in rows:
        gt_nt = r["no_target"]
        abstained = ((r["max_p"] < p_abs) or (vlm_abstain and r["vlm_none"])
                     or (use_exists and r.get("exists_no") is True))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--train-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--val-records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_val2000.jsonl"))
    ap.add_argument("--calib-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("exp/out_final/optuna_study_gref5.json"))
    args = ap.parse_args()

    t0 = time.time()
    gd_tr = GreferDataset(REPRO / "data/refer", "train")
    gd_va = GreferDataset(REPRO / "data/refer", "val")
    by_tr = {r["ref_id"]: r for r in gd_tr.refs}
    by_va = {r["ref_id"]: r for r in gd_va.refs}
    mc, gc = {}, {}
    train_rows, val_rows = [], []
    for rec in map(json.loads, open(args.train_records)):
        row = load_record(rec, gd_tr, by_tr, mc, gc)
        if row is not None:
            train_rows.append(row)
    for rec in map(json.loads, open(args.val_records)):
        row = load_record(rec, gd_va, by_va, mc, gc)
        if row is not None:
            val_rows.append(row)
    print(f"train rows {len(train_rows)}, val rows {len(val_rows)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # fit/calib split by ref (records of one ref stay together)
    rng = random.Random(args.seed)
    refs = sorted({r["rid"] for r in train_rows})
    rng.shuffle(refs)
    n_fit = int((1 - args.calib_frac) * len(refs))
    fit_refs = set(refs[:n_fit])
    for r in train_rows:
        r["split"] = "fit" if r["rid"] in fit_refs else "calib"
    calib_rows = [r for r in train_rows if r["split"] == "calib"]
    fit_rows = [r for r in train_rows if r["split"] == "fit"]
    print(f"fit refs {len(fit_refs)} rows {len(fit_rows)} | calib rows {len(calib_rows)}",
          flush=True)

    def params_of(trial, kind):
        if kind == "logreg":
            return dict(C=trial.suggest_float("C", 1e-3, 10.0, log=True))
        return dict(n_estimators=trial.suggest_int("n_estimators", 50, 150, step=50),
                    max_depth=trial.suggest_int("max_depth", 2, 4),
                    learning_rate=trial.suggest_float("learning_rate", 0.05, 0.3, log=True))

    def objective(trial):
        t1 = time.time()
        kind = trial.suggest_categorical("clf", ["logreg", "gbm"])
        params = params_of(trial, kind)
        clf = make_clf(kind, params)
        X = np.concatenate([r["X"] for r in fit_rows])
        y = np.concatenate([r["y"] for r in fit_rows])
        clf.fit(X, y)
        score_rows(calib_rows, clf)
        p_inc = trial.suggest_float("p_inc", 0.05, 0.9)
        p_abs = trial.suggest_float("p_abs", 0.0, 0.6)
        use_exists = trial.suggest_categorical("use_exists", [True, False])
        m = evaluate(calib_rows, p_inc, p_abs, use_exists=use_exists)
        trial.set_user_attr("metrics", m)
        print(f"  trial {trial.number} {kind} {time.time()-t1:.1f}s "
              f"cIoU={m['cIoU']:.1f} gIoU={m['gIoU']:.1f} "
              f"v={min(m['cIoU']/68.94, m['gIoU']/74.34):.3f}", flush=True)
        return min(m["cIoU"] / 68.94, m["gIoU"] / 74.34)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=args.seed))
    st.optimize(objective, n_trials=args.trials)
    best = st.best_trial
    print(f"{args.trials} trials in {time.time()-t0:.0f}s", flush=True)
    print(f"best calib value {best.value:.4f}  params {best.params}")
    print(f"calib metrics {best.user_attrs['metrics']}")

    # final: refit on ALL held-out train rows, thresholds frozen, one val pass
    bp = best.params
    kind = bp["clf"]
    params = (dict(C=bp["C"]) if kind == "logreg" else
              dict(n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
                   learning_rate=bp["learning_rate"]))
    clf = make_clf(kind, params)
    X = np.concatenate([r["X"] for r in train_rows])
    y = np.concatenate([r["y"] for r in train_rows])
    clf.fit(X, y)
    score_rows(val_rows, clf)
    vm = evaluate(val_rows, bp["p_inc"], bp["p_abs"],
                  use_exists=bool(bp.get("use_exists", False)))
    print(f"VAL transfer (n={vm['n']}): cIoU={vm['cIoU']:.2f} gIoU={vm['gIoU']:.2f} "
          f"N_acc={vm['N_acc']:.1f} T_acc={vm['T_acc']:.1f}")

    out = dict(n_trials=args.trials, best_calib_value=best.value,
               best_params=bp, calib_metrics=best.user_attrs["metrics"],
               val_transfer=vm, n_train_rows=len(train_rows),
               n_val_rows=len(val_rows),
               split="gRefCOCO train fit/calib; val2000 touched once for transfer",
               generated="exp62_clause_selector.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
