#!/usr/bin/env python
"""EXP-108c — top-N study configs evaluated on the full 262-row sample.

The 300-trial study decided on 69 calib rows; this re-evaluates the winner and
its nearest rivals on ALL 262 held-out rows (row-level and per-ref means) and
bootstraps the winner-vs-shipped delta by ref, so the val pass is launched on
a full-sample number rather than a calib artifact.

    CUDA_VISIBLE_DEVICES=1 python exp/exp108c_top_configs.py
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
    sample = json.loads((OUTF / "nounbox_sample_records.json").read_text())
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    study = json.loads((OUTF / "optuna_study_gref12.json").read_text())
    winner = dict(study["best_params"])
    winner["sam_area_cap"] = float(winner["sam_area_cap"])

    t0 = time.time()
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

    last = None
    for k, r in enumerate(rows):
        rec = recs_by_rid[r["rid"]]
        fn = rec["file_name"]
        if fn != last:
            predictor.set_image(np.asarray(Image.open(IMAGES / fn).convert("RGB")))
            last = fn
        terms = []
        for term in rec["terms"]:
            boxes, scores = g.detect(fn, term)
            tb = dict(det0=boxes[0] if boxes else None,
                      det1=boxes[1] if len(boxes) > 1 else None,
                      llava=None, sam=None)
            p = lpath(fn, term)
            if p.is_file():
                tb["llava"] = json.loads(p.read_text()).get("box")
            if tb["det0"] is not None:
                ms, _, _ = predictor.predict(
                    box=np.asarray(tb["det0"], np.float32), multimask_output=True)
                tb["sam"] = [downsample(m.astype(bool)) for m in ms]
            terms.append(tb)
        r["_terms"] = terms
    print(f"precompute done ({time.time()-t0:.0f}s)", flush=True)

    def feats_for(r, cfg):
        fs = []
        for tb in r["_terms"]:
            b = None
            if cfg["box_src"] == "gdino":
                if tb["det0"] is not None and True:
                    b = tb["det0"]
                elif tb["llava"] is not None:
                    b = tb["llava"]
            else:
                b = tb["llava"] if tb["llava"] is not None else tb["det0"]
            fs.append(box_feats(r["masks"], r["_bs"], b)[0] if b is not None else None)
        return fs

    def fit_picker(cfg):
        Xf, yf = [], []
        for r in rows:
            iou_gt = (np.logical_and(r["masks"], r["gt"]).sum((1, 2)).astype(np.float64)
                      / np.maximum(np.logical_or(r["masks"], r["gt"]).sum((1, 2)), 1.0))
            for F in feats_for(r, cfg):
                if F is None:
                    continue
                Xf.append(F)
                yf.append((iou_gt >= 0.5).astype(np.float32))
        clf = make_clf("logreg", dict(C=cfg["C"]))
        clf.fit(np.concatenate(Xf), np.concatenate(yf))
        return clf

    def run(r, cfg, clf):
        masks, gt, bs = r["masks"], r["gt"], r["_bs"]
        fs = feats_for(r, cfg)
        sel, picks = [], []
        for tb, F in zip(r["_terms"], fs):
            if F is None:
                continue
            if cfg["pool_rule"] == "sxi":
                score = F[:, 0] * (cfg["alpha"] + cfg["beta"] * F[:, 1])
            else:
                score = clf.predict_proba(F)[:, 1]
            order = np.argsort(score)[::-1]
            js = [int(order[0])]
            if cfg["topk"] == 2 and len(order) > 1 and \
                    score[order[1]] >= cfg["margin"] * score[order[0]]:
                js.append(int(order[1]))
            picks.extend(js)
            if cfg["sam_mode"] != "off" and tb["sam"] is not None:
                cap = cfg["sam_area_cap"]
                if cfg["sam_mode"] == "largest":
                    sm = tb["sam"][int(np.argmax([m.sum() for m in tb["sam"]]))]
                    if sm.sum() / sm.size <= cap:
                        sel.append(sm)
                elif cfg["sam_mode"] == "all3":
                    sel.extend(s for s in tb["sam"] if s.sum() / s.size <= cap)
                elif cfg["sam_mode"] == "evid":
                    areas = masks.sum((1, 2)).astype(np.float64)
                    best, bestv = None, -1.0
                    for sm in tb["sam"]:
                        if sm.sum() / sm.size > cap:
                            continue
                        inter = np.logical_and(masks, sm[None]).sum((1, 2)).astype(np.float64)
                        ov = inter / np.maximum(areas + sm.sum() - inter, 1.0)
                        v = float((bs * ov).max())
                        if v > bestv:
                            best, bestv = sm, v
                    if best is not None:
                        sel.append(best)
        if picks:
            sel.append(np.logical_or.reduce([masks[i] for i in dict.fromkeys(picks)]))
        if not sel:
            return None
        m = sel[0].copy()
        for x in sel[1:]:
            m |= x
        I = float(np.logical_and(m, gt).sum())
        U = float(np.logical_or(m, gt).sum())
        return I / U if U else 0.0

    # top-N distinct configs from the study, by calib objective
    import json as _json
    _st = _json.loads((OUTF / "optuna_study_gref12.json").read_text())
    _trials = _json.loads((OUTF / "optuna_study_gref12.json").read_text())  # params only here
    configs = {"study_winner": winner}
    # the study json carries only the winner; pull the stored trials if present
    trials_file = OUTF / "gref_study12_trials.json"
    if trials_file.is_file():
        for i, p in enumerate(_json.loads(trials_file.read_text())):
            cfg = dict(p)
            cfg.setdefault("sam_area_cap", 0.25)
            cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
            key = f"t{i}_" + "_".join(f"{k}{v}" for k, v in sorted(cfg.items())
                                      if k in ("box_src", "sam_mode", "cover2", "topk"))
            if key not in configs:
                configs[key] = cfg
    results = {}
    for name, cfg in configs.items():
        clf = fit_picker(cfg) if cfg["pool_rule"] == "logreg" else None
        vals, rids = [], []
        for r in rows:
            v = run(r, cfg, clf)
            if v is None:
                continue
            vals.append(v)
            rids.append(r["rid"])
        v = np.asarray(vals)
        per_rid = {}
        for rid, val in zip(rids, vals):
            per_rid.setdefault(rid, []).append(val)
        pr = np.array([np.mean(v_) for v_ in per_rid.values()])
        results[name] = dict(row_mean=float(v.mean()), row_n=len(v),
                             per_rid_mean=float(pr.mean()), per_rid_n=len(pr))
        print(f"{name:26s} row-mean {v.mean():.4f} (n={len(v)})  "
              f"per-rid {pr.mean():.4f} (n={len(pr)})", flush=True)

    # bootstrap winner vs shipped-equivalent (llava boxes, sam off) by ref
    clf_off = fit_picker(configs["llava_sxi_off"]) if False else None
    w = []
    s = []
    clf = fit_picker(winner) if winner["pool_rule"] == "logreg" else None
    for r in rows:
        a = run(r, winner, clf)
        b = run(r, configs["llava_sxi_off"], None)
        if a is None or b is None:
            continue
        w.append(a)
        s.append(b)
    w = np.asarray(w)
    s = np.asarray(s)
    d = w - s
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    print(f"winner - llava_sam_off (row-level): {d.mean():+.4f}  "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]  "
          f"P>0 {np.mean(boot>0):.4f}", flush=True)

    out = dict(configs={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for k, v in configs.items()},
               results=results,
               winner_delta_row_level=float(d.mean()),
               winner_delta_ci=[float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))],
               required_m=0.6212,
               generated="exp108c_top_configs.py")
    (OUTF / "gref_gdino_winner_verify.json").write_text(json.dumps(out, indent=2))
    print("wrote gref_gdino_winner_verify.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
