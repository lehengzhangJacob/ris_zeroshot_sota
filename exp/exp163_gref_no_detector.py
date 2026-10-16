#!/usr/bin/env python
"""EXP-163 — gRefCOCO val pass WITHOUT the GroundingDINO detector (backbone audit arm d).

Fork of exp124 (change-14 joint config) with the open-vocabulary detector
removed entirely: no GDINO model is loaded, term boxes come only from the
cached LLaVA nounboxes, per-term picks come only from the LLaVA box, and the
detector-box-prompted SAM mask caches are not read (they are keyed by det0/det1
boxes).  Abstention, the no-target singles fallback, the pair selector and the
metrics are byte-identical to exp124; the training-free realization serves
directly (no refit needed).  One model load fewer than the shipped path.

    CUDA_VISIBLE_DEVICES=0 python exp/exp163_gref_no_detector.py \
        --out exp/out_final/gref_no_detector_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO, probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco.jsonl"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref14.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_no_detector_val.json"))
    args = ap.parse_args()

    study = json.loads(args.study.read_text())
    cfg = dict(study["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    cfg.setdefault("gd_thresh", 0.25)
    cfg.setdefault("box_src", "llava")
    cfg.setdefault("pool_rule", "sxi")
    print(f"frozen rule (detector removed): {cfg}", flush=True)

    t0 = time.time()
    print("[phase] start", flush=True)
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    print("[phase] loading tune rows", flush=True)
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    print(f"[phase] tune rows {len(tr)}; fitting base clf", flush=True)
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
    print("[phase] base+pair clf fitted", flush=True)

    def llava_box_of(fn, term):
        p = lpath(fn, term)
        if not p.is_file():
            return None
        return json.loads(p.read_text()).get("box")

    def pick_masks(r, F, topk, margin):
        score = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
        order = np.argsort(score)[::-1]
        js = [int(order[0])]
        if topk == 2 and len(order) > 1 and score[order[1]] >= margin * score[order[0]]:
            js.append(int(order[1]))
        return [r["masks"][j] for j in js]

    # ---- full val pass (no detector anywhere) ------------------------------
    print("[phase] loading val rows", flush=True)
    val = load_rows(args.records, args.split, clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_by_rid = {}
    rid_fn = {}
    for line in open(args.records):
        rec = json.loads(line)
        rid_fn[int(rec["ref_id"])] = rec["file_name"]
        if rec.get("no_target"):
            continue
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], terms_of(rec["parsed"]))
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    p_sel = g8["p_sel"]
    masses = dict(S_U=0.0, I_tot=0.0, G_U=0.0, FP_U=0.0, A_U=0.0)
    acc_iou = acc_I = acc_U = 0.0
    tp = fn_c = tn = fp = 0
    n_fallback = n_ll = 0
    per_row = []
    t1 = time.time()
    for k, r in enumerate(val):
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
                masses["A_U"] += r["gt_area"]
            continue
        pred_parts = []
        fn_, ts = terms_by_rid.get(r["rid"], (None, []))
        if fn_ is not None and ts:
            for term in ts:
                b = llava_box_of(fn_, term)     # LLaVA nounbox only
                if b is None:
                    continue
                n_ll += 1
                Fb, _ = box_feats(r["masks"], r["_bs"], b)
                if Fb is not None:
                    pred_parts.extend(pick_masks(r, Fb, cfg["topk"], cfg["margin"]))
        pred = None
        if pred_parts:
            pred = pred_parts[0].copy()
            for x in pred_parts[1:]:
                pred |= x
        if pred is None:
            if r["no_target"]:
                single = [kk for kk, c in enumerate(ids) if c[1] < 0]
                kk = max(single, key=lambda kk: pp[kk]) if single else int(np.argmax(pp))
            else:
                kk = int(np.argmax(pp))
            pred = candidate_mask(r["masks"], ids[kk])
            n_fallback += 1
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        masses["S_U"] += float(pred.sum())
        masses["I_tot"] += I
        if not r["no_target"]:
            masses["G_U"] += r["gt_area"]
        else:
            masses["FP_U"] += float(pred.sum())
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
             llava_picks=n_ll, fallback_used=n_fallback)
    m["masses"] = dict(masses)
    n_nt = sum(1 for row in per_row if row[0])
    n_t = len(per_row) - n_nt
    w_nt = (0.6256 / 0.3744) * (n_t / max(1, n_nt))
    wI = wU = wiou = wsum = 0.0
    for nt, ab, I, U, gta in per_row:
        w = w_nt if nt else 1.0
        wsum += w
        if ab:
            if nt:
                wiou += w
            else:
                wU += w * gta
            continue
        if nt:
            wU += w * U
        else:
            wiou += w * (I / U if U else 0.0)
            wI += w * I
            wU += w * U
    m["cIoU_val_prevalence"] = 100.0 * wI / wU if wU else 0.0
    m["gIoU_val_prevalence"] = 100.0 * wiou / wsum if wsum else 0.0
    print(f"DETECTOR-FREE val: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} "
          f"N {m['N_acc']:.1f} T {m['T_acc']:.1f} "
          f"(llava picks {n_ll}, fallback {n_fallback})", flush=True)
    out = dict(model=dict(kind="termbox union rule WITHOUT GroundingDINO "
                               "(backbone audit arm d)",
                          study_params=cfg, source_study=str(args.study)),
               variant="no_detector",
               metrics=m,
               generated="exp163_gref_no_detector.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
