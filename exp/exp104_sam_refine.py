#!/usr/bin/env python
"""EXP-104 — SAM as a mask REFINER of pool picks (box from the picked mask).

exp103 measured the two ends: SAM with LLaVA nounbox boxes is worse than the
pool rule (oracle m 0.477 vs 0.5335), while SAM with GT boxes is far better
than any pool pick (0.8845 vs 0.6552) — the pool was the mask-quality
constraint under good boxes, and box imprecision is what breaks SAM under
LLaVA boxes.  The untouched middle: use the POOL PICK to localize (it is
trained on mask-IoU labels and realizes 86% of the in-box oracle) and SAM to
segment (bbox of the picked mask as the box prompt).  Detector = pool+CLIP,
segmenter = SAM; the Grounded-SAM decomposition with the pool as the detector.

Variants on the 262-row val-matched held-out sample:
  ref_global    bbox(union of shipped per-term pool picks) -> SAM largest
  ref_perterm   per term: bbox(term's top pool pick) -> SAM largest -> union
  ref_hybrid    shipped pool-pick union  ∪  ref_perterm SAM masks
  ref_perterm_orc  oracle: per term, over the 3 SAM outputs (ceiling)

References: pool rule 0.5335, LLaVA-box SAM oracle 0.4765, GT-box SAM oracle
0.8845, required m 0.6212.

    CUDA_VISIBLE_DEVICES=1 python exp/exp104_sam_refine.py
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
from exp89_termbox_full_val import term_features  # noqa: E402
from exp103_sam_boxp import downsample, union_m, npath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")
STRIDE = 2


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def main() -> int:
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g9 = json.loads((OUTF / "optuna_study_gref9.json").read_text())["best_params"]
    gd = GreferDataset(REPRO / "data/refer", "train")

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

    # picker fit (exp102 construction)
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
        gt = r["gt"]
        fn = rec["file_name"]
        if fn != last_img:
            rgb = np.asarray(Image.open(IMAGES / fn).convert("RGB"))
            predictor.set_image(rgb)
            last_img = fn

        # per-term pool picks (term_features skips box-less terms in order)
        per_term_picks = []
        for F_t in term_features(r, rec):
            pt = pick_clf.predict_proba(F_t)[:, 1]
            order = np.argsort(pt)[::-1]
            pk = [int(order[0])]
            if g9["topk"] == 2 and len(order) > 1 and pt[order[1]] >= g9["margin"] * pt[order[0]]:
                pk.append(int(order[1]))
            per_term_picks.append(pk)
        all_picks = [i for pk in per_term_picks for i in pk]
        if not all_picks:
            continue
        # dedupe preserving order
        seen, picks_u = set(), []
        for i in all_picks:
            if i not in seen:
                seen.add(i)
                picks_u.append(i)

        pool_m = r["masks"][picks_u[0]].copy()
        for i in picks_u[1:]:
            pool_m |= r["masks"][i]
        bump("m_pool", union_m([r["masks"][i] for i in picks_u], gt))

        # ref_global: one SAM call on the bbox of the whole pool union
        bb = bbox_of(pool_m)
        gmasks = None
        if bb is not None:
            b = np.asarray([v * STRIDE for v in bb], np.float32)
            gms, _, _ = predictor.predict(box=b, multimask_output=True)
            gms = np.stack([downsample(m.astype(bool)) for m in gms])
            gmasks = gms
            bump("m_ref_global", union_m([gms[int(np.argmax(gms.sum((1, 2))))]], gt))

        # ref_perterm: per term, SAM on the bbox of the term's top pool pick
        term_sam, term_sam_orc = [], []
        for pk in per_term_picks:
            top = r["masks"][pk[0]]
            bb = bbox_of(top)
            if bb is None:
                continue
            b = np.asarray([v * STRIDE for v in bb], np.float32)
            ms, _, _ = predictor.predict(box=b, multimask_output=True)
            ms = np.stack([downsample(m.astype(bool)) for m in ms])
            term_sam.append(ms[int(np.argmax(ms.sum((1, 2))))])
            term_sam_orc.append(ms)
        if term_sam:
            bump("m_ref_perterm", union_m(term_sam, gt))
            bump("m_ref_hybrid", union_m([r["masks"][i] for i in picks_u] + term_sam, gt))
        if term_sam_orc:
            import itertools
            orc = max(union_m(list(combo), gt)
                      for combo in itertools.product(*[list(ms) for ms in term_sam_orc]))
            if orc is not None:
                bump("m_ref_perterm_orc", orc)

        n_rows += 1
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    out = dict(
        rows=n_rows,
        m_pool_rule=acc.get("m_pool", 0) / max(1, n_rows),
        m_ref_global=acc.get("m_ref_global", 0) / max(1, n_rows),
        m_ref_perterm=acc.get("m_ref_perterm", 0) / max(1, n_rows),
        m_ref_hybrid=acc.get("m_ref_hybrid", 0) / max(1, n_rows),
        m_ref_perterm_oracle=acc.get("m_ref_perterm_orc", 0) / max(1, n_rows),
        references=dict(pool_rule_exp103=0.5335, sam_llava_box_oracle=0.4765,
                        sam_gtbox_oracle=0.8845, required_m=0.6212),
        note="SAM box prompts from pool-pick bboxes (stride-2 boxes x2); held-out sample",
        generated="exp104_sam_refine.py")
    (OUTF / "gref_sam_refine_sample.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
