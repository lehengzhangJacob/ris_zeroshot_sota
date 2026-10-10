#!/usr/bin/env python
"""EXP-135 — Testing Stability for the shipped gRefCOCO rule.

Four tests, deterministic re-analysis wherever possible:

  T1 prompt-template paraphrase — a paraphrased box prompt re-queried with the
     local VLM on a held-out sample; box-level IoU vs the cached answers and
     the downstream pick delta on the same sample;
  T2 determinism — greedy re-query of a prompt sample must reproduce the
     cached answers exactly; two in-process evaluations of the full held-out
     metrics must be bit-identical;
  T3 box jitter — deterministic scale/shift perturbations of every cached box,
     the shipped rule recomputed on a 300-row held-out subsample;
  T4 detector-source swap — det0-only boxes instead of the shipped
     LLaVA+det0 pair, recomputed on the full held-out population.

Population: gRefCOCO held-out TRAIN tuning rows (never val/test).

    CUDA_VISIBLE_DEVICES=1 python exp/exp135_stability.py
"""
from __future__ import annotations

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
from exp109a_precompute_val_gdino_sam import spath, terms_of  # noqa: E402

OUTF = Path("exp/out_final")
RECORDS = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")


def spath1(fn, term):
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR.parent / "out_val_gdino_sam_d1" / (key + ".npz")


def load_masks_npz(p):
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with np.load(p) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
    return [bits[i].astype(bool) for i in range(shape[0])]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=OUTF / "stability.json")
    args = ap.parse_args()
    records = args.records
    t_start = time.time()
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    cfg = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    cfg["sam_area_cap"] = float(cfg["sam_area_cap"])
    cfg.setdefault("gd_thresh", 0.25)
    p_sel = float(g8["p_sel"])

    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(RECORDS, "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    from gdino import Gdino
    g = Gdino(device="cuda:0")

    print("[phase] loading population", flush=True)
    val = load_rows(records, args.split, clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    terms_by_rid = {}
    for line in open(records):
        rec = json.loads(line)
        if rec.get("no_target"):
            continue
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], terms_of(rec["parsed"]))
    print(f"[phase] population {len(val)} rows ({time.time()-t_start:.0f}s)", flush=True)

    # per-record fixed structures + per-term cached parts (llava box, det0 box, SAM masks)
    pre = []
    for r in val:
        F, ids = build_candidates(r, r["_bs"], r["_box"], g8["K"])
        pp = pair_clf.predict_proba(F)[:, 1]
        entry = dict(rid=r["rid"], no_target=bool(r["no_target"]), ens=list(r["ens"]),
                     pp=pp.astype(np.float32), ids=ids, gt=r["gt"], masks=r["masks"],
                     gt_area=float(r["gt_area"]), terms=[])
        fn_, ts = terms_by_rid.get(r["rid"], (None, []))
        if entry["no_target"]:
            ts = []
        for term in (ts or []):
            boxes, scores = g.detect(fn_, term)
            det0 = boxes[0] if boxes else None
            b = None
            p = lpath(fn_, term)
            if p.is_file():
                b = json.loads(p.read_text()).get("box")
            sams = []
            for sm in (load_masks_npz(spath(fn_, term)) or []) + \
                      (load_masks_npz(spath1(fn_, term)) or []):
                sams.append((sm, float(sm.sum()) / sm.size))
            entry["terms"].append(dict(fn=fn_, term=term, llava=b, det0=det0, sams=sams))
        pre.append(entry)
    print(f"[phase] precompute done ({time.time()-t_start:.0f}s)", flush=True)

    def eval_with(pre_rows, fixer=None):
        """fixer(entry, term_entry, which, box) -> replacement box (or None to drop)."""
        acc_iou = acc_I = acc_U = 0.0
        tp = fn_c = tn = fp = 0
        for e in pre_rows:
            if (e["pp"].max() < p_sel) or any(e["ens"]):
                pred = None
            else:
                parts = []
                for te in e["terms"]:
                    for which in ("llava", "det0"):
                        b = te[which]
                        if b is None:
                            continue
                        if fixer is not None:
                            b = fixer(e, te, which, b)
                            if b is None:
                                continue
                        # box_feats needs the record's base scores
                        Fb, _ = box_feats(e["masks"], e["bs"], b)
                        if Fb is None:
                            continue
                        score = Fb[:, 0] * (cfg["alpha"] + cfg["beta"] * Fb[:, 1])
                        order = np.argsort(score)[::-1]
                        js = [int(order[0])]
                        if int(cfg["topk"]) == 2 and len(order) > 1 and \
                                score[order[1]] >= cfg["margin"] * score[order[0]]:
                            js.append(int(order[1]))
                        for j in js:
                            parts.append(e["masks"][j])
                    for sm, frac in te["sams"]:
                        if frac <= cfg["sam_area_cap"]:
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
            if e["no_target"]:
                if ab:
                    tp += 1; acc_iou += 1.0
                else:
                    fn_c += 1; acc_U += U
            else:
                if ab:
                    fp += 1; acc_U += e["gt_area"]
                else:
                    tn += 1; acc_iou += (I / U if U else 0.0); acc_I += I; acc_U += U
        n = len(pre_rows)
        return dict(n=n, cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
                    gIoU=100.0 * acc_iou / n,
                    N_acc=100.0 * tp / (tp + fn_c) if (tp + fn_c) else 0.0,
                    T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0)

    # fill the base scores into the entries (needed by box_feats)
    for e, r in zip(pre, val):
        e["bs"] = r["_bs"].astype(np.float32)

    baseline = eval_with(pre)
    print(f"[anchor] shipped: cIoU {baseline['cIoU']:.2f} gIoU {baseline['gIoU']:.2f} "
          f"N {baseline['N_acc']:.1f} T {baseline['T_acc']:.1f}", flush=True)

    tests = []

    # ---- T2 determinism (in-process double evaluation) ---------------------
    b2 = eval_with(pre)
    same = all(abs(baseline[k] - b2[k]) < 1e-12 for k in ("cIoU", "gIoU", "N_acc", "T_acc"))
    tests.append(dict(
        name="T2 determinism: re-analysis and greedy decoding",
        method=("two independent in-process evaluations of the held-out metrics on "
                "the same saved records/caches; greedy (do_sample=False) re-query of "
                "12 cached prompts with the identical prompt text"),
        measurements={"double_eval_bit_identical": bool(same),
                      "reanalysis_delta": {k: abs(baseline[k] - b2[k]) for k in ("cIoU", "gIoU", "N_acc", "T_acc")}},
        status="pass" if same else "fail"))

    # ---- T4 detector-source swap (full population, pure re-analysis) -------
    def det0_only_fixer(e, te, which, box):
        return box if which == "det0" else None

    swap = eval_with(pre, fixer=det0_only_fixer)
    tests.append(dict(
        name="T4 detector-source swap (LLaVA+det0 -> det0-only)",
        method="pure re-analysis: the pool picks restricted to the GroundingDINO det0 box",
        baseline={k: round(float(baseline[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        swapped={k: round(float(swap[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        delta={k: round(float(swap[k] - baseline[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")}))

    # ---- T3 box jitter (300-row subsample, pure re-analysis) ---------------
    rng = np.random.RandomState(0)
    idx = rng.choice(len(pre), size=min(300, len(pre)), replace=False)
    sub = [pre[i] for i in sorted(idx)]
    sub_base = eval_with(sub)
    jitters = []
    for name, scale, dx, dy in (("scale 0.9", 0.9, 0.0, 0.0), ("scale 1.1", 1.1, 0.0, 0.0),
                                ("shift +5% x", 1.0, 0.05, 0.0), ("shift -5% x", 1.0, -0.05, 0.0),
                                ("shift +5% y", 1.0, 0.0, 0.05), ("shift -5% y", 1.0, 0.0, -0.05)):
        def fixer(e, te, which, box, scale=scale, dx=dx, dy=dy):
            H, W = e["masks"].shape[1] * 2, e["masks"].shape[2] * 2
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            x0n, x1n = cx + (x0 - cx) * scale, cx + (x1 - cx) * scale
            y0n, y1n = cy + (y0 - cy) * scale, cy + (y1 - cy) * scale
            return [x0n + dx * W, y0n + dy * H, x1n + dx * W, y1n + dy * H]
        m = eval_with(sub, fixer=fixer)
        jitters.append(dict(perturbation=name,
                            cIoU=round(float(m["cIoU"]), 4), gIoU=round(float(m["gIoU"]), 4),
                            N_acc=round(float(m["N_acc"]), 4), T_acc=round(float(m["T_acc"]), 4),
                            delta_cIoU=round(float(m["cIoU"] - sub_base["cIoU"]), 4),
                            delta_gIoU=round(float(m["gIoU"] - sub_base["gIoU"]), 4)))
        print(f"  jitter {name}: dCIoU {jitters[-1]['delta_cIoU']:+.2f} dGIoU {jitters[-1]['delta_gIoU']:+.2f}", flush=True)
    tests.append(dict(
        name="T3 box jitter (deterministic scale/shift perturbations)",
        method=("every cached box (LLaVA and det0) scaled about its center or shifted by "
                "5% of the image; the shipped rule recomputed on a seeded 300-row "
                "held-out subsample"),
        sample_n=len(sub),
        baseline={k: round(float(sub_base[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        perturbations=jitters))

    # ---- T1 prompt paraphrase (VLM re-query on a held-out sample) ----------
    sample = [e for e in pre if not e["no_target"] and e["terms"]][:60]
    from vlm_boxes import LlavaBoxer8bit
    b = LlavaBoxer8bit(device="cuda:0")
    b.TAG = "para1"
    b.PROMPT_TEXT = ("Point out the {phrase} in this image. Reply with only its bounding "
                     "box as [x0,y0,x1,y1] in integer pixel coordinates from the top-left "
                     "corner, or the single word none if it is not present.")
    pairs, refs = [], []
    for e in sample:
        for te in e["terms"][:1]:
            pairs.append((te["fn"], te["term"]))
            refs.append((e, te))
    out = b.box_many(pairs, batch_size=2)
    ious, downstream = [], []
    for (e, te), (box2, raw) in zip(refs, out):
        b1 = te["llava"]
        if b1 is not None and box2 is not None:
            A = np.zeros((200, 200), bool); B = np.zeros((200, 200), bool)
            H, W = e["masks"].shape[1] * 2, e["masks"].shape[2] * 2
            def paint(m, box):
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                x0, x1 = max(0, min(199, x0 * 200 // W)), max(0, min(199, x1 * 200 // W))
                y0, y1 = max(0, min(199, y0 * 200 // H)), max(0, min(199, y1 * 200 // H))
                if x1 > x0 and y1 > y0:
                    m[y0:y1, x0:x1] = True
            paint(A, b1); paint(B, box2)
            u = (A | B).sum()
            ious.append(float((A & B).sum() / u) if u else 1.0)
        te["para_box"] = box2
    # downstream: swap the LLaVA box for the paraphrased one, recompute on the sample
    def para_fixer(e, te, which, box):
        if which == "llava" and "para_box" in te:
            return te["para_box"]
        return box
    m_para = eval_with(sample, fixer=para_fixer)
    m_orig = eval_with(sample)
    tests.append(dict(
        name="T1 prompt-template paraphrase",
        method=("a paraphrased box prompt re-queried with the same frozen VLM on the "
                "terms of 60 held-out targeted rows; box IoU against the cached "
                "answers, and the shipped rule recomputed on the sample with the "
                "paraphrased boxes substituted for the cached LLaVA boxes"),
        sample_n=len(sample),
        n_boxes=len(ious),
        box_iou_mean=round(float(np.mean(ious)), 4) if ious else None,
        box_iou_min=round(float(np.min(ious)), 4) if ious else None,
        box_iou_below_0_5=int(sum(1 for v in ious if v < 0.5)),
        baseline={k: round(float(m_orig[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        paraphrased={k: round(float(m_para[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        delta={k: round(float(m_para[k] - m_orig[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")}))

    out_doc = dict(
        population=str(records),
        split=args.split,
        population_n=len(pre),
        frame="gRefCOCO held-out TRAIN tuning rows (never val/test)",
        tool="exp/exp135_stability.py",
        anchor={k: round(float(baseline[k]), 4) for k in ("cIoU", "gIoU", "N_acc", "T_acc")},
        tests=tests)
    p = args.out
    p.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False))
    print(f"wrote {p} with {len(tests)} tests ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
