#!/usr/bin/env python
"""EXP-107 — deployable selection over GDINO detections (+ SAM masks).

exp106: GDINO top-1 per-term boxes score IoU 0.7046 / hit 71.6% vs LLaVA's
0.580 / 63.5%; oracle over its detections 0.8419 / 86.2%; 56.6% of terms get
>1 detection.  The naive rule (pool pick inside the top-1 GDINO box) already
delivers mean targeted IoU m 0.5861 vs the LLaVA-box rule's 0.5335 — the gIoU
requirement is 0.6212.  This measures the deployable moves that could close
the remaining 0.035:

  V1  pool pick in top-1 detection                        (naive; reference)
  V2  LEARNED pick over detections (score, rank, area, in-box evidence)
      -> pool pick in the selected box
  V3  SAM(top-1 box) largest mask -> union
  V4  SAM(selected detection box) -> union
  V5  union of V1 pool picks and V3 SAM masks
  V6  multi-detection coverage: for terms with >=2 detections, pick a mask in
      each of the top-2 boxes and union (targets 2-object terms)

All on the 262-row val-matched held-out sample; detections are cache reads
(exp106 already populated them).  Picker labels are GT-box hits - held-out
only, no val.

    CUDA_VISIBLE_DEVICES=1 python exp/exp107_gdino_select.py
"""
from __future__ import annotations

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
from exp103_sam_boxp import downsample, union_m  # noqa: E402
from exp105_box_ensemble import term_features_box, best_object_iou_box  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def det_features(dets, masks, bs, term_index, n_terms, llava_box):
    """Per-detection feature rows (one row per detection of one term)."""
    areas = masks.sum((1, 2)).astype(np.float64)
    n = len(areas)
    scores = np.asarray([d[1] for d in dets], np.float32)
    rank = np.argsort(np.argsort(-scores)) / max(1, len(scores) - 1)
    rows = []
    for k, (b, s) in enumerate(dets):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        x0, y0 = max(0, x0 // STRIDE), max(0, y0 // STRIDE)
        x1, y1 = min(masks.shape[2], -(-x1 // STRIDE)), min(masks.shape[1], -(-y1 // STRIDE))
        if x1 <= x0 or y1 <= y0:
            rows.append(None)
            continue
        area_box = float((x1 - x0) * (y1 - y0))
        inb = masks[:, y0:y1, x0:x1].sum((1, 2)).astype(np.float64)
        bxi = inb / (areas + area_box - inb + 1e-9)
        inbf = inb / np.maximum(areas, 1.0)
        j = int(np.argmax(bxi))
        boxh = float(y1 - y0) / masks.shape[1]
        boxw = float(x1 - x0) / masks.shape[2]
        l_ov = 0.0
        if llava_box is not None:
            lx0, ly0, lx1, ly1 = llava_box
            ix = max(0.0, min(x1 * STRIDE, lx1) - max(x0 * STRIDE, lx0))
            iy = max(0.0, min(y1 * STRIDE, ly1) - max(y0 * STRIDE, ly0))
            inter = ix * iy
            uni = area_box * 4 + (lx1 - lx0) * (ly1 - ly0) - inter
            l_ov = inter / max(1e-9, uni)
        rows.append(np.asarray([
            float(s), float(rank[k]), area_box / masks[0].size,
            float(bxi.max()), float(bxi[j]), float(bs[j]), float(inbf[j]),
            boxw, boxh, float(term_index), float(n_terms), l_ov,
        ], np.float32))
    return rows


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    import hashlib
    from gdino import Gdino
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    g = Gdino(device="cuda:0")  # cache reader; nothing new to compute

    def lpath(fn, term):
        key = hashlib.sha1(
            f"{MODEL_TAG}|nounbox|{fn}|{term.lower()}".encode()).hexdigest()
        return CACHE_DIR / (key + ".json")

    def llava_box(fn, term):
        p = lpath(fn, term)
        if not p.is_file():
            return None
        return json.loads(p.read_text()).get("box")

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
    print(f"rows {len(rows)}", flush=True)

    # mask picker over pool candidates inside a box (exp102 construction)
    Xf, yf = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                  / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
        for F in term_features_box(
                r, rec, lambda t, fn=rec["file_name"]: llava_box(fn, t)):
            Xf.append(F)
            yf.append((iou_gt >= 0.5).astype(np.float32))
    pick_clf = make_clf("logreg", dict(C=g9["C"]))
    pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))

    # detection picker: label = selected box hits a GT object (IoU>=0.5)
    D, Dy = [], []
    for rid, rec in recs_by_rid.items():
        r = dedup.get(rid)
        if r is None:
            continue
        objs = object_masks(gd, ref_by_id[rid]) if rid in ref_by_id else []
        masks, bs = r["masks"], r["_bs"]
        for ti, term in enumerate(rec["terms"]):
            boxes, scores = g.detect(rec["file_name"], term)
            if not boxes:
                continue
            dets = list(zip(boxes, scores))
            fr = det_features(dets, masks, bs, ti, len(rec["terms"]),
                              llava_box(rec["file_name"], term))
            for (b, s), f in zip(dets, fr):
                if f is None:
                    continue
                iou = best_object_iou_box(b, objs)
                D.append(f)
                Dy.append(1.0 if (iou or 0) >= 0.5 else 0.0)
    det_clf = make_clf("logreg", dict(C=1.0))
    if D:
        det_clf.fit(np.stack(D), np.asarray(Dy))
        print(f"det picker rows {len(D)} pos {np.mean(Dy):.3f}", flush=True)

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    acc = {}
    def bump(k, v):
        acc[k] = acc.get(k, 0.0) + v

    last_img = None
    n_rows = 0
    t0 = time.time()
    for k, r in enumerate(rows):
        rec = recs_by_rid[r["rid"]]
        gt, masks, bs = r["gt"], r["masks"], r["_bs"]
        fn = rec["file_name"]
        if fn != last_img:
            rgb = np.asarray(Image.open(IMAGES / fn).convert("RGB"))
            predictor.set_image(rgb)
            last_img = fn

        picks_v1, picks_v2, sam_v3, sam_v4, picks_v6 = [], [], [], [], []
        n_terms_ok = 0
        for ti, term in enumerate(rec["terms"]):
            boxes, scores = g.detect(fn, term)
            if not boxes:
                continue
            dets = list(zip(boxes, scores))
            # V1: pool pick in top-1 box
            F1 = term_features_box(
                r, rec, lambda t, b=dets[0][0], tt=term: b if t == tt else None)
            if F1:
                pt = pick_clf.predict_proba(F1[0])[:, 1]
                order = np.argsort(pt)[::-1]
                picks_v1.append(int(order[0]))
                if g9["topk"] == 2 and len(order) > 1 and \
                        pt[order[1]] >= g9["margin"] * pt[order[0]]:
                    picks_v1.append(int(order[1]))
            # V2: learned detection pick, then pool pick in it
            if D:
                fr = det_features(dets, masks, bs, ti, len(rec["terms"]),
                                  llava_box(fn, term))
                valid = [(j, f) for j, f in enumerate(fr) if f is not None]
                if valid:
                    probs = det_clf.predict_proba(np.stack([f for _, f in valid]))[:, 1]
                    jbest = valid[int(np.argmax(probs))][0]
                    F2 = term_features_box(
                        r, rec, lambda t, b=dets[jbest][0], tt=term: b if t == tt else None)
                    if F2:
                        pt = pick_clf.predict_proba(F2[0])[:, 1]
                        order = np.argsort(pt)[::-1]
                        picks_v2.append(int(order[0]))
                        if g9["topk"] == 2 and len(order) > 1 and \
                                pt[order[1]] >= g9["margin"] * pt[order[0]]:
                            picks_v2.append(int(order[1]))
            # V3/V4: SAM on top-1 / selected box
            for tag, src in (("v3", 0), ("v4", None)):
                if tag == "v4" and not D:
                    continue
                bi = 0 if src == 0 else None
                if tag == "v4":
                    fr = det_features(dets, masks, bs, ti, len(rec["terms"]),
                                      llava_box(fn, term))
                    valid = [(jj, f) for jj, f in enumerate(fr) if f is not None]
                    if not valid:
                        continue
                    probs = det_clf.predict_proba(np.stack([f for _, f in valid]))[:, 1]
                    bi = valid[int(np.argmax(probs))][0]
                b = np.asarray(dets[bi][0], np.float32)
                ms, _, _ = predictor.predict(box=b, multimask_output=True)
                ms = np.stack([downsample(m.astype(bool)) for m in ms])
                mask = ms[int(np.argmax(ms.sum((1, 2))))]
                (sam_v3 if tag == "v3" else sam_v4).append(mask)
            # V6: coverage picks in top-1 AND top-2 boxes
            for b, s in dets[:2]:
                F6 = term_features_box(
                    r, rec, lambda t, b=b, tt=term: b if t == tt else None)
                if F6:
                    pt = pick_clf.predict_proba(F6[0])[:, 1]
                    picks_v6.append(int(np.argmax(pt)))
            n_terms_ok += 1

        if not n_terms_ok:
            continue
        for tag, pk in (("v1", picks_v1), ("v2", picks_v2), ("v6", picks_v6)):
            v = union_m([masks[i] for i in pk], gt) if pk else None
            if v is not None:
                bump("m_" + tag, v)
        if sam_v3:
            bump("m_v3", union_m(sam_v3, gt))
        if sam_v4:
            bump("m_v4", union_m(sam_v4, gt))
        v5 = union_m([masks[i] for i in (picks_v1 or [])] + sam_v3, gt) \
            if (picks_v1 or sam_v3) else None
        if v5 is not None:
            bump("m_v5", v5)
        n_rows += 1
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    out = dict(
        rows=n_rows,
        m_v1_pool_pick_top1=acc.get("m_v1", 0) / max(1, n_rows),
        m_v2_learned_detection_pick=acc.get("m_v2", 0) / max(1, n_rows),
        m_v3_sam_top1=acc.get("m_v3", 0) / max(1, n_rows),
        m_v4_sam_selected=acc.get("m_v4", 0) / max(1, n_rows),
        m_v5_pool_union_sam=acc.get("m_v5", 0) / max(1, n_rows),
        m_v6_two_detection_coverage=acc.get("m_v6", 0) / max(1, n_rows),
        references=dict(llava_box_rule=0.5335, required_m=0.6212,
                        gdino_box_iou_top1=0.7046, gdino_box_iou_oracle=0.8419),
        note="GDINO detections from cache; held-out sample; pickers fit on held-out labels",
        generated="exp107_gdino_select.py")
    (OUTF / "gref_gdino_select.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
