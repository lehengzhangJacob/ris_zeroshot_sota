#!/usr/bin/env python
"""EXP-112 — extra mask sources for the frozen rule (held-out sample probe).

The shipped rule adds SAM masks prompted by the top-1 GDINO detection only.
Two untested sources, both cheap: SAM prompted by the LLaVA nounbox (the box
the pool pick already uses), and SAM prompted by the 2nd GDINO detection.
Measured on the full 262-row val-matched sample against the shipped rule's
m 0.6433, plus the 1-term held-out rows (exp111: det1 sources +0.032 there).

Variants (all keep the shipped pool pick + SAM(det0)):
  base        the shipped rule
  +sam_llava  union with SAM(LLaVA nounbox)
  +sam_det1   union with SAM(2nd GDINO detection)
  +sam_both   union with both
  +sam_llava_only_on_1term  the 1-term-record extension from exp111

    CUDA_VISIBLE_DEVICES=1 python exp/exp112_masksources_probe.py
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
from exp103_sam_boxp import downsample  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from vlm_boxes import IMAGES  # noqa: E402

OUTF = Path("exp/out_final")


def main() -> int:
    study = json.loads((OUTF / "optuna_study_gref12.json").read_text())
    cfg = dict(study["best_params"])
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([x["X"] for x in tr]),
                 np.concatenate([x["y"] for x in tr]))
    for x in tr:
        x["_bs"] = clf_base.predict_proba(x["X"])[:, 1]
    dedup = {r["rid"]: r for r in tr}
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    recs_by_rid = {rec["ref_id"]: rec for rec in sample}
    rows = [dedup[rec["ref_id"]] for rec in sample if rec["ref_id"] in dedup]

    # 1-term multi-target held-out rows (no sample filter)
    from exp109a_precompute_val_gdino_sam import terms_of
    t1 = {}
    for line in open('exp/out_extreme/records_grefcoco_train_tune.jsonl'):
        rec = json.loads(line)
        if rec.get("no_target") or int(rec.get("n_targets") or 0) < 2:
            continue
        ts = terms_of(rec["parsed"])
        if len(ts) == 1:
            t1[rec["ref_id"]] = (rec["file_name"], ts)
    rows1 = [dedup[rid] for rid in t1 if rid in dedup]

    from gdino import Gdino
    g = Gdino(device="cuda:0")
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    def prep(rows_):
        last = None
        for r in rows_:
            rec = recs_by_rid.get(r["rid"]) or {"file_name": t1[r["rid"]][0],
                                                "terms": t1[r["rid"]][1]}
            fn = rec["file_name"]
            if fn != last:
                predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
                last = fn
            terms = []
            for term in rec["terms"]:
                boxes, scores = g.detect(fn, term)
                tb = dict(det0=boxes[0] if boxes else None,
                          det1=boxes[1] if len(boxes) > 1 else None,
                          llava=None, sam0=None, samL=None, sam1=None)
                p = lpath(fn, term)
                if p.is_file():
                    tb["llava"] = json.loads(p.read_text()).get("box")
                if tb["det0"] is not None:
                    ms, _, _ = predictor.predict(
                        box=np.asarray(tb["det0"], np.float32), multimask_output=True)
                    tb["sam0"] = [downsample(m.astype(bool)) for m in ms]
                if tb["llava"] is not None:
                    ms, _, _ = predictor.predict(
                        box=np.asarray(tb["llava"], np.float32), multimask_output=True)
                    tb["samL"] = [downsample(m.astype(bool)) for m in ms]
                if tb["det1"] is not None:
                    ms, _, _ = predictor.predict(
                        box=np.asarray(tb["det1"], np.float32), multimask_output=True)
                    tb["sam1"] = [downsample(m.astype(bool)) for m in ms]
                terms.append(tb)
            r["_terms"] = terms
    prep(rows)
    prep(rows1)
    print(f"precompute done ({time.time()-t0:.0f}s)", flush=True)

    cap = cfg["sam_area_cap"]
    def ok(ms):
        return [m for m in (ms or []) if m.sum() / m.size <= cap]

    def predict(r, variant, one_term=False):
        parts = []
        for tb in r["_terms"]:
            box = tb["llava"] if tb["llava"] is not None else tb["det0"]
            if box is not None:
                F, _ = box_feats(r["masks"], r["_bs"], box)
                if F is not None:
                    sc = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
                    order = np.argsort(sc)[::-1]
                    js = [int(order[0])]
                    if cfg["topk"] == 2 and len(order) > 1 and \
                            sc[order[1]] >= cfg["margin"] * sc[order[0]]:
                        js.append(int(order[1]))
                    parts.extend(r["masks"][j] for j in js)
            parts.extend(ok(tb["sam0"]))
            if variant in ("sam_llava", "sam_both"):
                parts.extend(ok(tb["samL"]))
            if one_term and variant in ("sam_det1", "sam_both", "sam_llava_only_on_1term"):
                parts.extend(ok(tb["sam1"]))
                if variant == "sam_llava_only_on_1term" and tb["det1"] is not None:
                    F2, _ = box_feats(r["masks"], r["_bs"], tb["det1"])
                    if F2 is not None:
                        sc = F2[:, 0] * (cfg["alpha"] + cfg["beta"] * F2[:, 1])
                        order = np.argsort(sc)[::-1]
                        js = [int(order[0])]
                        if cfg["topk"] == 2 and len(order) > 1 and \
                                sc[order[1]] >= cfg["margin"] * sc[order[0]]:
                            js.append(int(order[1]))
                        parts.extend(r["masks"][j] for j in js)
        if not parts:
            return None
        m = parts[0].copy()
        for x in parts[1:]:
            m |= x
        I = float(np.logical_and(m, r["gt"]).sum())
        U = float(np.logical_or(m, r["gt"]).sum())
        return I / U if U else 0.0

    results = {}
    for name, variant in (("base", "base"), ("sam_llava", "sam_llava"),
                          ("sam_det1", "sam_det1"), ("sam_both", "sam_both")):
        vals = [v for v in (predict(r, variant, one_term=True) for r in rows) if v is not None]
        vals1 = [v for v in (predict(r, "sam_llava_only_on_1term", one_term=True) for r in rows1)
                 if v is not None]
        results[name] = dict(sample_m=float(np.mean(vals)), sample_n=len(vals),
                             oneterm_m=float(np.mean(vals1)) if vals1 else 0.0,
                             oneterm_n=len(vals1))
        print(f"{name:12s} sample m {np.mean(vals):.4f} (n={len(vals)})   "
              f"1-term m {np.mean(vals1) if vals1 else 0:.4f} (n={len(vals1)})", flush=True)

    out = dict(results=results,
               reference=dict(sample_base_m=0.6433, val_gIoU_floor_gap=0.39,
                              val_m=0.609),
               generated="exp112_masksources_probe.py")
    (OUTF / "gref_masksources_probe.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_masksources_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
