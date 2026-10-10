#!/usr/bin/env python
"""EXP-103 — SAM box-prompted masks vs the AMG pool (pilot, 262-row sample).

Every pool-pick measurement (exp94/102) conflates two constraints: the VLM
box quality AND the AMG pool's mask quality inside that box.  With GT boxes
the pool pick reaches only mean targeted IoU 0.6552 against the 0.6212 the
gIoU floor demands — but "0.6552 with GT boxes" is a statement about POOL
masks, not about the best mask a box can induce.  SAM's box-prompted decoder
(the Grounded-SAM primitive) is a different mask source the pipeline has
never used: box in, three tight masks out.

This pilot measures, on the same 262 val-matched held-out rows as exp102:

  (a) per-term SAM mask vs best-matching GT object: mean IoU and %>=0.50,
      for each of the 3 multimask outputs (LLaVA-OneVision pool reference on
      boxes: 0.580 / 63.5%);
  (b) record-level union m (mean targeted IoU vs r["gt"]) for:
        v1  per-term argmax-SAM-score mask -> union          (deployable)
        v2  union of ALL 3 masks per term -> union           (deployable)
        v3  per-term largest-area mask -> union              (deployable)
        orc best combination of the 3 options per term       (oracle)
      with LLaVA term boxes, and the same with per-object GT boxes
      (exp102 pool reference under GT boxes: m 0.6552).

If the LLaVA-box oracle clears 0.6212 the gIoU bind is mask quality too, and
a picker over SAM candidates becomes the 13th model change; if the GT-box
oracle stays near the pool's 0.6552, the AMG pool was never the constraint.
Held-out only; no val, no tuning here.

    CUDA_VISIBLE_DEVICES=0 python exp/exp103_sam_boxp.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp67_ensemble_eval import REPRO  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp83_two_box_oracle import object_masks  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, IMAGES, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def npath(file_name, term):
    key = hashlib.sha1(
        f"{MODEL_TAG}|nounbox|{file_name}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR / (key + ".json")


def downsample(m):
    return m[::STRIDE, ::STRIDE]


def best_object_iou(mask, objs):
    best = 0.0
    for om in objs:
        inter = float(np.logical_and(mask, om).sum())
        uni = float(np.logical_or(mask, om).sum())
        best = max(best, inter / max(1.0, uni))
    return best


def union_m(masks_sel, gt):
    if not masks_sel:
        return None
    m = masks_sel[0].copy()
    for x in masks_sel[1:]:
        m |= x
    I = float(np.logical_and(m, gt).sum())
    U = float(np.logical_or(m, gt).sum())
    return I / U if U else 0.0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gref_sam_boxp_sample.json")
    args = ap.parse_args()
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([x["X"] for x in tr]),
                 np.concatenate([x["y"] for x in tr]))
    for x in tr:
        x["_bs"] = clf_base.predict_proba(x["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]
    if args.limit:
        rows = rows[:args.limit]
    print(f"rows {len(rows)}", flush=True)

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    from exp89_termbox_full_val import term_features

    # shipped picker (exp102 construction), fit on the held-out sample rows
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

    def pool_picks(r, rec):
        picks = []
        for F_t in term_features(r, rec):
            pt = pick_clf.predict_proba(F_t)[:, 1]
            order = np.argsort(pt)[::-1]
            picks.append(int(order[0]))
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                picks.append(int(order[1]))
        return picks

    acc = {}
    def bump(k, v):
        acc[k] = acc.get(k, 0.0) + v

    n_rows = 0
    n_terms = 0
    rank_iou = [0.0, 0.0, 0.0]
    rank_sel = [0.0, 0.0, 0.0]
    best_any = 0.0
    gt_rank_iou = [0.0, 0.0, 0.0]
    gt_best_any = 0.0
    gt_objs_n = 0
    missing_terms = 0
    t0 = time.time()
    last_img = None    # SAM predictor holds exactly one image at a time
    term_cache = {}    # (file_name, term) -> (3,H/2,W/2) bool

    for k, r in enumerate(rows):
        rec = recs_by_rid[r["rid"]]
        gt = r["gt"]
        fn = rec["file_name"]
        if fn != last_img:
            rgb = np.asarray(Image.open(IMAGES / fn).convert("RGB"))
            predictor.set_image(rgb)
            last_img = fn

        term_masks = []  # list of (3,H/2,W/2) per term with a box
        for term in rec["terms"]:
            key = (fn, term)
            if key in term_cache:
                term_masks.append(term_cache[key])
                n_terms += 1
                continue
            p = npath(fn, term)
            box = None
            if p.is_file():
                box = json.loads(p.read_text()).get("box")
            if box is None:
                missing_terms += 1
                continue
            b = np.asarray([float(v) for v in box], np.float32)
            ms, scores, _ = predictor.predict(box=b, multimask_output=True)
            ms = np.stack([downsample(m.astype(bool)) for m in ms])
            term_cache[key] = ms
            term_masks.append(ms)
            n_terms += 1
        if not term_masks:
            continue

        # (a) per-term mask quality vs GT objects
        objs = object_masks(gd, ref_by_id[r["rid"]]) if r["rid"] in ref_by_id else []
        if objs:
            for ms in term_masks:
                ious = [best_object_iou(ms[i], objs) for i in range(3)]
                for i in range(3):
                    rank_iou[i] += ious[i]
                    rank_sel[i] += 1.0 if ious[i] >= 0.5 else 0.0
                best_any += max(ious)

        # (b) record-level unions
        v1 = union_m([ms[0] for ms in term_masks], gt)
        v2 = union_m([m for ms in term_masks for m in ms], gt)
        v3 = union_m(
            [ms[int(np.argmax(ms.sum((1, 2))))] for ms in term_masks], gt)
        orc = max(union_m(list(combo), gt)
                  for combo in itertools.product(*[list(ms) for ms in term_masks]))
        for name, v in (("v1", v1), ("v2", v2), ("v3", v3), ("orc", orc)):
            if v is not None:
                bump("m_llava_" + name, v)

        # shipped pool rule on the same rows + pool-union-SAM hybrid
        pp = pool_picks(r, rec)
        pool_u = union_m([r["masks"][i] for i in pp], gt) if pp else None
        if pool_u is not None:
            bump("m_pool", pool_u)
            v4 = union_m([r["masks"][i] for i in pp]
                         + [ms[int(np.argmax(ms.sum((1, 2))))] for ms in term_masks], gt)
            if v4 is not None:
                bump("m_hybrid_v4", v4)

        # GT-box variant: one GT object box per object, same SAM call
        gms_list = []
        for om in objs:
            ys, xs = np.nonzero(om)
            if not len(ys):
                continue
            x0, y0 = int(xs.min()) * STRIDE, int(ys.min()) * STRIDE
            x1, y1 = int(xs.max() + 1) * STRIDE, int(ys.max() + 1) * STRIDE
            b = np.asarray([x0, y0, x1, y1], np.float32)
            ms, _, _ = predictor.predict(box=b, multimask_output=True)
            ms = np.stack([downsample(m.astype(bool)) for m in ms])
            gms_list.append(ms)
            ious = [best_object_iou(ms[i], [om]) for i in range(3)]
            for i in range(3):
                gt_rank_iou[i] += ious[i]
            gt_best_any += max(ious)
            gt_objs_n += 1
        if gms_list:
            gorc = max(union_m(list(combo), gt)
                       for combo in itertools.product(*[list(ms) for ms in gms_list]))
            b1 = union_m([ms[0] for ms in gms_list], gt)
            if gorc is not None:
                bump("m_gtbox_orc", gorc)
            if b1 is not None:
                bump("m_gtbox_v1", b1)

        n_rows += 1
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    out = dict(
        rows=n_rows, terms_with_box=n_terms, terms_missing_box=missing_terms,
        m_llava_sam_v1_argmax_score=acc.get("m_llava_v1", 0) / max(1, n_rows),
        m_llava_sam_v2_all3=acc.get("m_llava_v2", 0) / max(1, n_rows),
        m_llava_sam_v3_largest=acc.get("m_llava_v3", 0) / max(1, n_rows),
        m_llava_sam_oracle=acc.get("m_llava_orc", 0) / max(1, n_rows),
        m_pool_rule_same_rows=acc.get("m_pool", 0) / max(1, n_rows),
        m_pool_union_sam_v4=acc.get("m_hybrid_v4", 0) / max(1, n_rows),
        m_gtbox_sam_v1=acc.get("m_gtbox_v1", 0) / max(1, n_rows),
        m_gtbox_sam_oracle=acc.get("m_gtbox_orc", 0) / max(1, n_rows),
        sam_mask_rank_iou_mean=[v / max(1, n_terms) for v in rank_iou],
        sam_mask_rank_select_ge50_pct=[100.0 * v / max(1, n_terms) for v in rank_sel],
        sam_mask_best_of_3_iou_mean=best_any / max(1, n_terms),
        gtbox_sam_mask_rank_iou_mean=[v / max(1, gt_objs_n) for v in gt_rank_iou],
        gtbox_sam_best_of_3_iou_mean=gt_best_any / max(1, gt_objs_n),
        required_m_at_shipped_operating_point=0.6212,
        references=dict(
            pool_delivered_m_262rows=0.5335,
            pool_gtbox_m_262rows=0.6552,
            llava_pool_box_iou_mean=0.580, llava_pool_select_ge50_pct=63.5),
        note=("pilot on the val-matched held-out sample; unions use per-term masks "
              "with LLaVA nounbox boxes (full-res SAM box prompts, stride-2 decode)"),
        generated="exp103_sam_boxp.py")
    (OUTF / args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
