#!/usr/bin/env python
"""EXP-106 — GroundingDINO as the per-term box source (last untested detector).

exp105 closed prompt paraphrasing by scale (paraphrase agreement 0.838; no
prompt-level move adds +0.30 box IoU).  The remaining untested box source is a
DIFFERENT FAMILY: the repo has a GroundingDINO-base wrapper (exp/gdino.py,
COCO-trained open-vocabulary detector, weights cached locally) whose cache
holds 3 detections from an early probe - it was never measured as the
per-term box source for the termbox construction.

Measures on the 262-row val-matched held-out sample (487 unique (file, term)
pairs, ~125 images; GDINO ~0.3-1 s/query, so minutes not hours):

  per-term box quality   best GDINO detection vs best GT object: IoU mean,
                         hit@0.50   (LLaVA-OneVision reference: 0.580 / 63.5%)
  top-1 vs oracle        top-score detection vs best-IoU-over-detections
  multi-box check        #detections>1 share (the multi-instance signal)

If GDINO clears the LLaVA reference substantially, the termbox rule is re-run
with GDINO boxes (same learned picker) and its mean targeted IoU m is compared
against delivered 0.5335 / required 0.6212; if not, the box-source axis is
closed with an independent detector family.

    CUDA_VISIBLE_DEVICES=1 python exp/exp106_gdino_terms.py
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
from exp105_box_ensemble import term_features_box, best_object_iou_box  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")
    ref_by_id = {x["ref_id"]: x for x in gd.refs}

    pairs = list(dict.fromkeys((r["file_name"], t) for r in sample for t in r["terms"]))
    rid_of = {(r["file_name"], t): r["ref_id"] for r in sample for t in r["terms"]}
    print(f"unique (file, term) pairs: {len(pairs)}", flush=True)

    from gdino import Gdino
    g = Gdino(device="cuda:0")  # CUDA_VISIBLE_DEVICES pins the physical GPU

    det_cache = {}
    t0 = time.time()
    n_det = 0
    for fn, term in pairs:
        boxes, scores = g.detect(fn, term)
        det_cache[(fn, term)] = (boxes, scores)
        n_det += 1
        if n_det % 100 == 0:
            print(f"  {n_det}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"detections done in {time.time()-t0:.0f}s", flush=True)

    ious_top1, ious_orc, hits1, hits_orc, nmulti = [], [], [], [], []
    for fn, term in pairs:
        boxes, scores = det_cache[(fn, term)]
        objs = object_masks(gd, ref_by_id[rid_of[(fn, term)]]) \
            if rid_of[(fn, term)] in ref_by_id else []
        if boxes:
            nmulti.append(1.0 if len(boxes) > 1 else 0.0)
            top1 = best_object_iou_box(boxes[0], objs)
            orc = max(best_object_iou_box(b, objs) for b in boxes)
            ious_top1.append(top1 or 0.0)
            ious_orc.append(orc or 0.0)
            hits1.append(1.0 if (top1 or 0) >= 0.5 else 0.0)
            hits_orc.append(1.0 if (orc or 0) >= 0.5 else 0.0)

    out = dict(
        unique_pairs=len(pairs),
        pairs_with_detections=len(ious_top1),
        gdino_top1_box_iou_mean=float(np.mean(ious_top1)) if ious_top1 else 0.0,
        gdino_oracle_box_iou_mean=float(np.mean(ious_orc)) if ious_orc else 0.0,
        gdino_top1_hit50_pct=100.0 * float(np.mean(hits1)) if hits1 else 0.0,
        gdino_oracle_hit50_pct=100.0 * float(np.mean(hits_orc)) if hits_orc else 0.0,
        multi_detection_share=float(np.mean(nmulti)) if nmulti else 0.0,
        llava_reference=dict(box_iou_mean=0.580, hit50_pct=63.5),
        required_box_iou_for_gIoU="~0.73-0.88 depending on mask source (exp102/103 interpolation)",
        generated="exp106_gdino_terms.py")
    (OUTF / "gref_gdino_terms.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))

    # rule-level m with GDINO boxes (top-1) if quality is in range
    if out["gdino_top1_hit50_pct"] >= 50.0:
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

        Xf, yf = [], []
        for rid, rec in recs_by_rid.items():
            r = dedup.get(rid)
            if r is None:
                continue
            iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                      / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
            for F in term_features_box(
                    r, rec, lambda t, fn=rec["file_name"]:
                    (det_cache.get((fn, t), (None, None))[0] or [None])[0]):
                Xf.append(F)
                yf.append((iou_gt >= 0.5).astype(np.float32))
        pick_clf = make_clf("logreg", dict(C=g9["C"]))
        pick_clf.fit(np.concatenate(Xf), np.concatenate(yf))

        acc, n = 0.0, 0
        for r in rows:
            rec = recs_by_rid[r["rid"]]
            fn = rec["file_name"]
            picks = []
            for F_t in term_features_box(
                    r, rec, lambda t, fn=fn:
                    (det_cache.get((fn, t), (None, None))[0] or [None])[0]):
                pt = pick_clf.predict_proba(F_t)[:, 1]
                order = np.argsort(pt)[::-1]
                picks.append(int(order[0]))
                if g9["topk"] == 2 and len(order) > 1 and \
                        pt[order[1]] >= g9["margin"] * pt[order[0]]:
                    picks.append(int(order[1]))
            if not picks:
                continue
            m = r["masks"][picks[0]].copy()
            for i in picks[1:]:
                m |= r["masks"][i]
            I = float(np.logical_and(m, r["gt"]).sum())
            U = float(np.logical_or(m, r["gt"]).sum())
            acc += I / U if U else 0.0
            n += 1
        out["rule_m_with_gdino_boxes"] = acc / max(1, n)
        out["rule_m_rows"] = n
        (OUTF / "gref_gdino_terms.json").write_text(json.dumps(out, indent=2))
        print(f"rule m with GDINO boxes: {out['rule_m_with_gdino_boxes']:.4f} on {n} rows "
              f"(delivered 0.5335, required 0.6212)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
