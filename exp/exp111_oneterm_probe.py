#!/usr/bin/env python
"""EXP-111 — the 1-term subpopulation probe (the val transfer gap's home).

Val targeted records by parsed-term count: 1 term 664 (12.5%), 2 terms 2,346,
3 terms 2,314.  The held-out sample behind the shipped rule was FILTERED to
>=2-term records, so the rule's handling of 1-term multi-target records ("the
two men" parses to a single noun) was never measured — and back-computing from
the val pass (m_val 0.609 against the sample's 0.6433 over the >=2-term 87.5%)
puts their implied mean IoU at ~0.37.

Held-out probe on train tune rows with n_targets>=2 and exactly 1 parsed term
(75+ records).  Variants, all with the frozen rule's pieces (sxi pool pick,
SAM masks from the top-1 detection, area cap 0.25):

  a  current rule (det0 box pick + SAM(det0))          <- shipped behavior
  b  a + pool pick in the 2nd detection box
  c  a + SAM prompt from the 2nd detection box
  d  b + c

A variant ships only if it lifts this subpopulation's mean IoU clearly; the
>=2-term path is untouched, so the 262-sample verification stays valid.

    CUDA_VISIBLE_DEVICES=1 python exp/exp111_oneterm_probe.py
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
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402
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

    # 1-term multi-target records from the tune file (with their terms)
    t1 = {}
    for line in open('exp/out_extreme/records_grefcoco_train_tune.jsonl'):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        if int(rec.get("n_targets") or 0) < 2:
            continue
        ts = terms_of(rec["parsed"])
        if len(ts) == 1:
            t1[rec["ref_id"]] = (rec["file_name"], ts)
    rows = [dedup[rid] for rid in t1 if rid in dedup]
    print(f"1-term multi-target held-out rows: {len(rows)}", flush=True)

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

    def llava_box(fn, term):
        p = lpath(fn, term)
        if not p.is_file():
            return None
        return json.loads(p.read_text()).get("box")

    last = None
    for r in rows:
        fn, ts = t1[r["rid"]]
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        tb = {"fn": fn, "terms": ts, "dets": [], "sam0": None}
        term = ts[0]
        boxes, scores = g.detect(fn, term)
        tb["dets"] = list(zip(boxes, scores))
        box0 = llava_box(fn, term) or (boxes[0] if boxes else None)
        tb["box0"] = box0
        if boxes:
            ms, _, _ = predictor.predict(box=np.asarray(boxes[0], np.float32),
                                         multimask_output=True)
            tb["sam0"] = [downsample(m.astype(bool)) for m in ms]
        if len(boxes) > 1:
            ms, _, _ = predictor.predict(box=np.asarray(boxes[1], np.float32),
                                         multimask_output=True)
            tb["sam1"] = [downsample(m.astype(bool)) for m in ms]
        else:
            tb["sam1"] = None
        r["_tb"] = tb
    print(f"precompute done ({time.time()-t0:.0f}s)", flush=True)

    def pick_in(r, box):
        F, _ = box_feats(r["masks"], r["_bs"], box)
        if F is None:
            return None
        sc = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
        order = np.argsort(sc)[::-1]
        js = [int(order[0])]
        if cfg["topk"] == 2 and len(order) > 1 and sc[order[1]] >= cfg["margin"] * sc[order[0]]:
            js.append(int(order[1]))
        return js

    def sam_ok(masks):
        return [m for m in (masks or []) if m.sum() / m.size <= cfg["sam_area_cap"]]

    acc = {}
    for r in rows:
        gt = r["gt"]
        tb = r["_tb"]
        parts = {}
        p0 = pick_in(r, tb["box0"]) if tb["box0"] is not None else None
        parts["a"] = ([r["masks"][j] for j in p0] if p0 else []) + sam_ok(tb["sam0"])
        d1 = tb["dets"][1][0] if len(tb["dets"]) > 1 else None
        p1 = pick_in(r, d1) if d1 is not None else None
        parts["b"] = parts["a"] + ([r["masks"][j] for j in p1] if p1 else [])
        parts["c"] = parts["a"] + sam_ok(tb["sam1"])
        parts["d"] = parts["b"] + sam_ok(tb["sam1"])
        for tag, ps in parts.items():
            if not ps:
                continue
            m = ps[0].copy()
            for x in ps[1:]:
                m |= x
            I = float(np.logical_and(m, gt).sum())
            U = float(np.logical_or(m, gt).sum())
            acc.setdefault(tag, []).append(I / U if U else 0.0)

    out = dict(rows=len(rows),
               reference=dict(implied_val_oneterm_m=0.369, sample_ge2term_m=0.6433),
               generated="exp111_oneterm_probe.py")
    for tag, name in (("a", "current_rule"), ("b", "plus_pick_det1"),
                      ("c", "plus_sam_det1"), ("d", "plus_both")):
        v = np.asarray(acc.get(tag, []))
        out[name] = dict(n=len(v), row_mean_m=float(v.mean()) if len(v) else 0.0)
        print(f"{name:16s} n={len(v):3d} m={float(v.mean()) if len(v) else 0.0:.4f}", flush=True)
    (OUTF / "gref_oneterm_probe.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_oneterm_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
