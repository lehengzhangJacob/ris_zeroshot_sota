#!/usr/bin/env python
"""EXP-172 — gRefCOCO evaluation with per-record term alignment.

The earlier gRefCOCO term-path evaluations (exp124/152/163/169) looked terms
up by ``ref_id``; gRefCOCO records share a ref_id across several sentences
(14,229 val records over 4,237 ref_ids), so all but one sentence per ref used
another sentence's noun terms (84.7% of targeted val rows differ).  This
script is exp124's frozen training-free GDINO+SAM rule re-evaluated with each
row's OWN parsed sentence and file name.  Caches are read directly (all term
pairs are cached); no detector model is loaded.

    python exp/exp172_gref_aligned.py --shard 0 --nshards 4
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
from exp67_ensemble_eval import probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import spath, terms_of  # noqa: E402
from exp124_fp_shrink_val import spath1  # noqa: E402

OUTF = Path("exp/out_final")
GDINO_CACHE = Path("exp/out_gdino")
GDINO_MODEL = "IDEA-Research/grounding-dino-base"
GDINO_THRESH = 0.25


def detect_cached(fn, phrase):
    key = hashlib.sha1(f"{GDINO_MODEL}|{GDINO_THRESH}|{fn}|{phrase}".encode()).hexdigest()
    p = GDINO_CACHE / (key + ".json")
    if not p.is_file():
        return [], []
    d = json.loads(p.read_text())
    return d["boxes"], d["scores"]


def llava_box_of(fn, term):
    p = lpath(fn, term)
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("box")


def sam_masks(fn, term, path_fn):
    p = path_fn(fn, term)
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with np.load(p) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
    return [bits[i].astype(bool) for i in range(shape[0])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_14229.jsonl"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref14.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_tf_aligned.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--params", type=Path, default=None,
                    help="optional JSON with alpha/beta/topk/margin/p_sel overrides (held-out re-tune)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    study = json.loads(args.study.read_text())
    cfg = dict(study["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    cfg.setdefault("gd_thresh", 0.25)
    cfg.setdefault("box_src", "llava")
    cfg.setdefault("pool_rule", "sxi")
    if args.params is not None:
        override = json.loads(args.params.read_text())
        override = override.get("best", override)
        for k in ("alpha", "beta", "topk", "margin", "p_sel", "src", "pick_box",
                  "sam", "cap", "sam_area_cap"):
            if k in override:
                cfg[k] = override[k]
        if "cap" in override and "sam_area_cap" not in override:
            cfg["sam_area_cap"] = override["cap"]
        print(f"held-out re-tune override applied: {override}", flush=True)
    print(f"frozen training-free GDINO+SAM rule: {cfg}", flush=True)

    t0 = time.time()
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
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
    print(f"[phase] base+pair clf fitted ({time.time()-t0:.0f}s)", flush=True)

    print("[phase] loading val rows", flush=True)
    val = load_rows(args.records, args.split, clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    if args.nshards > 1:
        val = val[args.shard::args.nshards]
    if args.limit:
        val = val[: args.limit]
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    p_sel = float(cfg.get("p_sel", g8["p_sel"]))
    acc_iou = acc_I = acc_U = 0.0
    tp = fn_c = tn = fp = 0
    n_terms = n_fallback = 0
    per_row = []
    t1 = time.time()
    for k, r in enumerate(val):
        rec = r["rec"]
        fn_ = rec["file_name"]
        ts = terms_of(rec["parsed"])
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        abstain = (pp.max() < p_sel) or any(r["ens"])
        if abstain:
            per_row.append((r["no_target"], True, 0.0, 0.0, r["gt_area"]))
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                acc_U += r["gt_area"]
            continue
        parts = []
        src = cfg.get("src", "bs")
        base = (r["_bs"] if src == "bs" else
                (r["_sc"] if src == "sc" else r["_gem"]))
        pick_box = cfg.get("pick_box", "ll+d0")
        sam_choice = cfg.get("sam", "d0+d1")
        for term in ts:
            n_terms += 1
            boxes, scores = detect_cached(fn_, term)
            det0 = boxes[0] if boxes else None
            b = llava_box_of(fn_, term)
            if cfg.get("box_src") == "gdino" and det0 is not None and \
                    scores and scores[0] >= cfg["gd_thresh"]:
                b = det0
            box_list = []
            if pick_box in ("ll", "ll+d0") and b is not None:
                box_list.append(b)
            if pick_box in ("d0", "ll+d0") and det0 is not None:
                box_list.append(det0)
            for bb in box_list:
                if bb is None:
                    continue
                Fb, _ = box_feats(r["masks"], r["_bs"], bb)
                if Fb is None:
                    continue
                if cfg.get("pool_rule") == "sxi":
                    score = np.asarray(base, np.float32) * (cfg["alpha"] + cfg["beta"] * Fb[:, 1])
                else:  # not used by the frozen rule
                    score = Fb[:, 0]
                order = np.argsort(score)[::-1]
                js = [int(order[0])]
                if cfg["topk"] == 2 and len(order) > 1 and \
                        score[order[1]] >= cfg["margin"] * score[order[0]]:
                    js.append(int(order[1]))
                parts.extend(r["masks"][j] for j in js)
            cap = cfg["sam_area_cap"]
            path_list = {"d0": (spath,), "d0+d1": (spath, spath1), "none": ()}[sam_choice]
            for path_fn in path_list:
                for sm in (sam_masks(fn_, term, path_fn) or []):
                    if sm.sum() / sm.size <= cap:
                        parts.append(sm)
        pred = None
        if parts:
            pred = parts[0].copy()
            for x in parts[1:]:
                pred |= x
        if pred is None:
            n_fallback += 1
            if r["no_target"]:
                single = [kk for kk, c in enumerate(ids) if c[1] < 0]
                kk = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
            else:
                kk = int(np.argmax(pp))
            pred = candidate_mask(r["masks"], ids[kk])
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        per_row.append((r["no_target"], False, I, U, r["gt_area"]))
        if r["no_target"]:
            fn_c += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(val)} ({time.time()-t1:.0f}s)", flush=True)

    n = len(val)
    m = dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
             gIoU=100.0 * acc_iou / n,
             N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
             T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
             TP=tp, FN=fn_c, TN=tn, FP=fp,
             term_lookups=n_terms, fallback_used=n_fallback,
             shard=args.shard, nshards=args.nshards,
             alignment="per-record sentence (r['rec']), not ref_id-keyed",
             generated="exp172_gref_aligned.py")
    print(f"ALIGNED training-free val: cIoU {m['cIoU']:.4f} gIoU {m['gIoU']:.4f} "
          f"N {m['N_acc']:.2f} T {m['T_acc']:.2f} (terms {n_terms}, "
          f"fallback {n_fallback})", flush=True)
    args.out.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
