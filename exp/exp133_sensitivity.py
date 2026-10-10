#!/usr/bin/env python
"""EXP-133 — Parameter Sensitivity Analysis for the shipped gRefCOCO rule.

Deterministic re-analysis of saved records and caches (no model re-runs):
every per-term box, SAM mask, pair probability and GT is loaded once; the
shipped rule's key hyperparameters are then swept by re-composing the same
rule under override values:

  alpha / beta  box-fill weight in the pool score  s = base * (a + b*in-box IoU)
  margin        top-2 acceptance margin
  cap           SAM mask area cap
  topk          1 (single pick) vs 2 (top-2 pick)
  p_sel         pair-selector abstention threshold

Population: the held-out gRefCOCO TRAIN tuning rows at val prevalence (the
same frame the shipped parameters were fitted in), metric formulas identical
to exp124 (raw + val-prevalence-weighted).  Correctness anchor: the shipped
setting must reproduce the frozen exp124 held-out run exactly.

    CUDA_VISIBLE_DEVICES=1 python exp/exp133_sensitivity.py
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
from exp78_pair_selector import build_candidates, candidate_mask, precompute_row, topk_mask  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import terms_of  # noqa: E402

_SAM_DIRS = [Path("exp/out_val_gdino_sam"), Path("exp/out_val_gdino_sam_d1")]


def spath(fn, term):
    import hashlib
    from vlm_boxes import MODEL_TAG
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam|{fn}|{term.lower()}".encode()).hexdigest()
    return _SAM_DIRS[0] / (key + ".npz")

OUTF = Path("exp/out_final")


def spath1(fn, term):
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest()
    return _SAM_DIRS[1] / (key + ".npz")


def sam_dirs(train_side):
    from vlm_boxes import CACHE_DIR
    if train_side:
        return (Path("exp/out_train_gdino_sam"), Path("exp/out_train_gdino_sam_d1"))
    return (CACHE_DIR.parent / "out_val_gdino_sam",
            CACHE_DIR.parent / "out_val_gdino_sam_d1")


def load_masks_npz(p):
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with np.load(p) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
    return [bits[i].astype(bool) for i in range(shape[0])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path,
                    default=Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref14.json"))
    ap.add_argument("--out", type=Path, default=OUTF / "sensitivity.json")
    ap.add_argument("--train-side-sam", action="store_true",
                    help="read the box-prompted SAM caches built for the TRAIN rows (exp137)")
    args = ap.parse_args()

    global _SAM_DIRS
    _SAM_DIRS = sam_dirs(args.train_side_sam)
    study = json.loads(args.study.read_text())
    cfg = dict(study["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    cfg.setdefault("gd_thresh", 0.25)
    cfg.setdefault("box_src", "llava")
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

    from gdino import Gdino
    g = Gdino(device="cuda:0")

    def llava_box_of(fn, term):
        p = lpath(fn, term)
        if not p.is_file():
            return None
        return json.loads(p.read_text()).get("box")

    print("[phase] loading population", flush=True)
    val = load_rows(args.records, args.split, clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_by_rid = {}
    for line in open(args.records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], terms_of(rec["parsed"]))
    print(f"[phase] population {len(val)} rows", flush=True)

    t0 = time.time()
    pre = []
    for r in val:
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        entry = dict(no_target=bool(r["no_target"]), ens=list(r["ens"]),
                     pp=pp.astype(np.float32), ids=ids, gt=r["gt"],
                     gt_area=float(r["gt_area"]),
                     masks=r["masks"], terms=[])
        if not entry["no_target"] and not any(r["ens"]):
            fn_, ts = terms_by_rid.get(r["rid"], (None, []))
            if fn_ is not None and ts:
                for term in ts:
                    boxes, scores = g.detect(fn_, term)
                    det0 = boxes[0] if boxes else None
                    b2 = boxes[1] if len(boxes) > 1 and scores[1] >= cfg["gd_thresh"] else None
                    b = llava_box_of(fn_, term)
                    if cfg["box_src"] == "gdino":
                        if det0 is not None and scores[0] >= cfg["gd_thresh"]:
                            cand_boxes = [det0]
                        else:
                            cand_boxes = ([b] if b is not None
                                          else ([det0] if det0 is not None else []))
                    else:
                        # exp124 change 14: picks from BOTH the LLaVA box (or the
                        # det0 fallback) and the det0 box
                        cand_boxes = ([b] if b is not None
                                      else ([det0] if det0 is not None else []))
                        if det0 is not None:
                            cand_boxes.append(det0)
                    feats = []
                    for bb in cand_boxes:
                        Fb, _ = box_feats(r["masks"], r["_bs"], bb)
                        if Fb is not None:
                            feats.append(np.stack([Fb[:, 0], Fb[:, 1]], 1))
                    sams = []
                    for sm in (load_masks_npz(spath(fn_, term)) or []) + \
                              (load_masks_npz(spath1(fn_, term)) or []):
                        sams.append((sm, float(sm.sum()) / sm.size))
                    entry["terms"].append((feats, sams))
        pre.append(entry)
        if (len(pre)) % 200 == 0:
            print(f"  precomputed {len(pre)}/{len(val)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[phase] precompute done ({time.time()-t0:.0f}s)", flush=True)

    def eval_setting(alpha, beta, margin, cap, topk, p_sel):
        acc_iou = acc_I = acc_U = 0.0
        tp = fn_c = tn = fp = 0
        per_row = []
        for e in pre:
            if (e["pp"].max() < p_sel) or any(e["ens"]):
                pred = None
            else:
                parts = []
                for feats, sams in e["terms"]:
                    for F2 in feats:
                        score = F2[:, 0] * (alpha + beta * F2[:, 1])
                        order = np.argsort(score)[::-1]
                        js = [int(order[0])]
                        if topk == 2 and len(order) > 1 and \
                                score[order[1]] >= margin * score[order[0]]:
                            js.append(int(order[1]))
                        for j in js:
                            parts.append(e["masks"][j])
                    for sm, frac in sams:
                        if frac <= cap:
                            parts.append(sm)
                if parts:
                    pred = parts[0].copy()
                    for x in parts[1:]:
                        pred |= x
                else:
                    pred = None
            if pred is None:
                if e["no_target"]:
                    single = [k for k, c in enumerate(e["ids"]) if c[1] < 0]
                    kk = max(single, key=lambda k: e["pp"][k]) if single else int(np.argmax(e["pp"]))
                else:
                    kk = int(np.argmax(e["pp"]))
                pred = candidate_mask(e["masks"], e["ids"][kk])
            I = float(np.logical_and(pred, e["gt"]).sum())
            U = float(np.logical_or(pred, e["gt"]).sum())
            ab = (e["pp"].max() < p_sel) or any(e["ens"])
            per_row.append((e["no_target"], ab, I, U, e["gt_area"]))
            if e["no_target"]:
                if ab:
                    tp += 1
                    acc_iou += 1.0
                else:
                    fn_c += 1
                    acc_U += U
            else:
                if ab:
                    fp += 1
                    acc_U += e["gt_area"]
                else:
                    tn += 1
                    acc_iou += (I / U if U else 0.0)
                    acc_I += I
                    acc_U += U
        n = len(pre)
        raw = dict(cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                   gIoU=100.0 * acc_iou / n,
                   N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
                   T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)
        n_nt = sum(1 for row in per_row if row[0])
        n_t = n - n_nt
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
        raw["cIoU_val_prev"] = 100.0 * wI / wU if wU else 0.0
        raw["gIoU_val_prev"] = 100.0 * wiou / wsum if wsum else 0.0
        return raw

    ship = dict(alpha=cfg["alpha"], beta=cfg["beta"], margin=cfg["margin"],
                cap=cfg["sam_area_cap"], topk=int(cfg["topk"]), p_sel=float(g8["p_sel"]))
    anchor = eval_setting(ship["alpha"], ship["beta"], ship["margin"], ship["cap"],
                          ship["topk"], ship["p_sel"])
    print(f"[anchor] shipped setting: cIoU {anchor['cIoU']:.2f} gIoU {anchor['gIoU']:.2f} "
          f"N {anchor['N_acc']:.1f} T {anchor['T_acc']:.1f} "
          f"(val-prev cIoU {anchor['cIoU_val_prev']:.2f} gIoU {anchor['gIoU_val_prev']:.2f})",
          flush=True)

    sweeps = []
    plans = [
        ("alpha (box-fill weight, base)", "alpha", [1.2, 1.5, 1.8112, 2.2, 2.5]),
        ("beta (box-fill weight, in-box IoU term)", "beta", [0.0, 0.12, 0.2343, 0.35, 0.47]),
        ("margin (top-2 acceptance)", "margin", [0.3, 0.45, 0.6313, 0.75, 0.9]),
        ("cap (SAM mask area cap)", "cap", [0.2, 0.3, 0.5, 0.65, 0.8]),
        ("topk (pool picks per box)", "topk", [1, 2]),
        ("p_sel (abstention threshold)", "p_sel", [0.2, 0.35, 0.5, 0.65, 0.8]),
    ]
    for label, key, values in plans:
        settings = []
        for v in values:
            kw = dict(ship)
            kw[key] = v
            m = eval_setting(kw["alpha"], kw["beta"], kw["margin"], kw["cap"],
                             kw["topk"], kw["p_sel"])
            m = {k: round(float(x), 4) for k, x in m.items()}
            m["value"] = v
            m["is_shipped"] = bool(abs(v - ship[key]) < 1e-9) or (key == "topk" and v == ship["topk"])
            settings.append(m)
            print(f"  {key}={v}: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} N {m['N_acc']:.1f} T {m['T_acc']:.1f}", flush=True)
        sweeps.append(dict(parameter=label, key=key, shipped_value=ship[key],
                           settings=settings))

    out = dict(
        population=str(args.records),
        population_n=len(pre),
        split=args.split,
        frame=("held-out gRefCOCO train tuning rows; raw metrics plus "
               "val-prevalence-weighted metrics (exp124 frame)"),
        tool="exp/exp133_sensitivity.py (deterministic re-analysis of saved records and caches)",
        shipped_setting=ship,
        shipped_anchor={k: round(float(v), 4) for k, v in anchor.items()},
        note=("no model inference in the sweep: boxes, SAM masks and pair "
              "probabilities come from the frozen caches; the sweep reports "
              "metric values per setting and selects nothing"),
        sweeps=sweeps)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
