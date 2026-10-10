#!/usr/bin/env python
"""EXP-116 — GroundingDINO at the SENTENCE level (record-level box source).

exp115 decomposed the remaining loss: the box axis carries +0.234 mean
targeted IoU (GT boxes + shipped pick 0.802 vs 0.568 delivered) while mask
choice adds +0.070.  Everything measured so far asks GDINO per PARSED TERM
(0.7046 top-1 box IoU, 56.6% multi-detection).  The untested query is the
full SENTENCE: grounding models are trained on captions, and a 2-object
sentence like "left giraffe and tall giraffe" may ground directly into two
instance boxes — bypassing the term parse (whose single-noun failures feed
the 12.5% 1-term subpopulation) and giving the rule a per-object box set for
the record as a whole.

Variants on the 262-row val-matched held-out sample (sentence detections are
fresh GDINO queries, cache namespace = same scheme, phrase = the sentence):
  s1  top-1 sentence detection -> SAM all-3 (area-capped) -> union
  s2  top-2 sentence detections -> SAM each -> union
  s3  top-3 sentence detections -> SAM each -> union
  s_orc  best over detections (per-detection SAM all-3, oracle subset) - ceil
  compare: shipped term-level rule (0.6433), required m 0.6212.

Held-out only; no val reads.

    CUDA_VISIBLE_DEVICES=1 python exp/exp116_sentence_gdino.py
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
from exp103_sam_boxp import downsample  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
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

    from gdino import Gdino
    g = Gdino(device="cuda:0")

    # phase A: sentence detections (fresh queries for unique sentences)
    sentences = {}
    for rec in sample:
        s = (rec.get("sentence") or "").strip()
        if s:
            sentences[(rec["file_name"], s)] = True
    print(f"sentence queries: {len(sentences)}", flush=True)
    n = 0
    for fn, s in sentences:
        g.detect(fn, s)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{len(sentences)} ({time.time()-t0:.0f}s)", flush=True)

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPRO))
    from glod.vision import load_upstream
    _, registry, _, _ = load_upstream()
    sam = registry["vit_h"](
        checkpoint=str(REPRO / "checkpoints/sam_vit_h_4b8939.pth")).to("cuda").eval()
    from segment_anything import SamPredictor
    predictor = SamPredictor(sam)

    def sam3_ok(box):
        ms, _, _ = predictor.predict(box=np.asarray(box, np.float32), multimask_output=True)
        out = [downsample(m.astype(bool)) for m in ms]
        return [m for m in out if m.sum() / m.size <= cfg["sam_area_cap"]]

    last = None
    sdet = {}
    for r in rows:
        rec = recs_by_rid[r["rid"]]
        fn = rec["file_name"]
        s = (rec.get("sentence") or "").strip()
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        boxes, scores = g.detect(fn, s)
        dets = list(zip(boxes, scores))[:3]
        masks = [sam3_ok(b) for b, _ in dets]
        sdet[r["rid"]] = masks
    print(f"sentence SAM done ({time.time()-t0:.0f}s)", flush=True)

    out = {}
    for tag, take in (("s1", 1), ("s2", 2), ("s3", 3)):
        vals = []
        for r in rows:
            parts = []
            for ms in sdet[r["rid"]][:take]:
                parts.extend(ms)
            if not parts:
                continue
            m = parts[0].copy()
            for x in parts[1:]:
                m |= x
            I = float(np.logical_and(m, r["gt"]).sum())
            U = float(np.logical_or(m, r["gt"]).sum())
            vals.append(I / U if U else 0.0)
        out[tag] = dict(row_mean=float(np.mean(vals)), n=len(vals))
        print(f"{tag}: m {out[tag]['row_mean']:.4f} (n={len(vals)})", flush=True)

    # oracle over detection subsets
    import itertools
    vals = []
    for r in rows:
        ms = sdet[r["rid"]]
        best = 0.0
        for combo in itertools.product(*[range(len(x)) for x in ms]) if ms else []:
            parts = [ms[i][ci] for i, ci in enumerate(combo)]
            if not parts:
                continue
            m = parts[0].copy()
            for x in parts[1:]:
                m |= x
            I = float(np.logical_and(m, r["gt"]).sum())
            U = float(np.logical_or(m, r["gt"]).sum())
            best = max(best, I / U if U else 0.0)
        if ms:
            vals.append(best)
    out["s_orc"] = dict(row_mean=float(np.mean(vals)), n=len(vals))
    print(f"s_orc: m {out['s_orc']['row_mean']:.4f} (n={len(vals)})", flush=True)

    out["references"] = dict(shipped_term_rule=0.6433, required_m=0.6212)
    out["generated"] = "exp116_sentence_gdino.py"
    (OUTF / "gref_sentence_gdino.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
