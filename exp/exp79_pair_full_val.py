#!/usr/bin/env python
"""EXP-79 — frozen pair/union selector: one full-val pass (protocol-compliant).

exp78's 300-trial held-out study selected the pair/union selector (K=6,
logreg C=2.85e-4, p_sel=0.340, ensemble abstention) over the shipped
threshold-union reference by +0.0356 on the val-prevalence objective
(cIoU 46.63 vs 44.18, gIoU 79.53 vs 78.61, T_acc 90.3 vs 83.8, same N_acc).

This refits exactly that configuration on ALL held-out train rows, freezes
every scalar, and evaluates the official cIoU/gIoU protocol once over the
complete 14,220-record val split.  No val number influences any choice.

    python exp/exp79_pair_full_val.py --out exp/out_final/gref_pair_full_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import load_record, make_clf  # noqa: E402
from exp67_ensemble_eval import probe_no  # noqa: E402
from exp78_pair_selector import (build_candidates, candidate_mask,  # noqa: E402
                                 precompute_row, topk_mask)
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUTF = Path("exp/out_final")


def load_rows(records, split, clf_base, need_gt=True):
    import hashlib
    from vlm_boxes import CACHE_DIR, MODEL_TAG
    gd = GreferDataset(REPRO / "data/refer", split)
    by_id = {r["ref_id"]: r for r in gd.refs}
    mc, gc = {}, {}
    rows = []
    for line in open(records):
        rec = json.loads(line)
        ph = (rec["parsed"].get("sentence") or "").strip()
        if not ph:
            continue
        r = load_record(rec, gd, by_id, mc, gc)
        if r is None:
            continue
        r["ens"] = [probe_no(t, rec["file_name"], ph)
                    for t in ("exists", "exists2", "exists3")]
        if any(v is None for v in r["ens"]):
            continue
        r["_sc"] = np.asarray(rec["sc"][:len(r["X"])], np.float32)
        r["_gem"] = np.asarray(rec["score_gem"][:len(r["X"])], np.float32)
        key = hashlib.sha1(
            f"{MODEL_TAG}|{rec['file_name']}|{ph.lower()}".encode()).hexdigest()
        bp = CACHE_DIR / (key + ".json")
        box = json.loads(bp.read_text())["box"] if bp.is_file() else None
        r["_box"] = None
        if box is not None:
            H, W = r["masks"].shape[1], r["masks"].shape[2]
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0 = max(0, x0 // 2), max(0, y0 // 2)
            x1, y1 = min(W, -(-x1 // 2)), min(H, -(-y1 // 2))
            if x1 > x0 and y1 > y0:
                r["_box"] = (x0, y0, x1, y1)
        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path,
                    default=Path("exp/out_final/optuna_study_gref8.json"))
    ap.add_argument("--classifier", type=Path,
                    default=Path("exp/out_final/optuna_study_gref5.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("exp/out_final/gref_pair_full_val.json"))
    args = ap.parse_args()

    study = json.loads(args.study.read_text())
    bp = study["best_params"]
    st5 = json.loads(args.classifier.read_text())["best_params"]
    K, p_sel, use_ens = bp["K"], bp["p_sel"], bp["use_ens"]
    print(f"frozen pair selector: K={K} C={bp['C']:.2e} p_sel={p_sel:.3f} "
          f"use_ens={use_ens}", flush=True)

    t0 = time.time()
    clf_base = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(Path("exp/out_extreme/records_grefcoco_train_tune.jsonl"),
                   "train", clf_base)
    clf_base.fit(np.concatenate([r["X"] for r in tr]),
                 np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"train rows {len(tr)} ({time.time()-t0:.0f}s)", flush=True)

    # fit the pair classifier on ALL held-out train rows at the frozen K,
    # on the exact candidate subset the study selected (same topk_mask path)
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), K)
        Xs.append(F[km])
        ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=bp["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))
    print(f"pair classifier fit ({time.time()-t0:.0f}s)", flush=True)

    val = load_rows(Path("exp/out_extreme/records_grefcoco.jsonl"), "val", clf_base)
    for r in val:
        r["_bs"] = clf_base.predict_proba(r["X"])[:, 1]
    print(f"val rows {len(val)} ({time.time()-t0:.0f}s)", flush=True)

    acc_iou = acc_I = acc_U = 0.0
    tp = fn = tn = fp = 0
    n_abstain_t = 0
    for r in val:
        F, ids = build_candidates(r, r["_bs"], r["_box"], K)
        if len(F) == 0:
            abstain = True
        else:
            p = pair_clf.predict_proba(F)[:, 1]
            k = int(np.argmax(p))
            abstain = (p[k] < p_sel) or (use_ens and any(r["ens"]))
        if abstain:
            if r["no_target"]:
                tp += 1
                acc_iou += 1.0
            else:
                fp += 1
                n_abstain_t += 1
                acc_U += r["gt_area"]
            continue
        pred = candidate_mask(r["masks"], ids[k])
        I = float(np.logical_and(pred, r["gt"]).sum())
        U = float(np.logical_or(pred, r["gt"]).sum())
        if r["no_target"]:
            fn += 1
            acc_U += U
        else:
            tn += 1
            acc_iou += (I / U if U else 0.0)
            acc_I += I
            acc_U += U
    n = len(val)
    m = dict(n=n,
             cIoU=100.0 * acc_I / acc_U if acc_U else 0.0,
             gIoU=100.0 * acc_iou / n,
             N_acc=100.0 * tp / (tp + fn) if (tp + fn) else 0.0,
             T_acc=100.0 * tn / (tn + fp) if (tn + fp) else 0.0,
             TP=tp, FN=fn, TN=tn, FP=fp, targeted_abstained=n_abstain_t)
    print(f"FULL VAL pair selector: cIoU {m['cIoU']:.2f} gIoU {m['gIoU']:.2f} "
          f"N {m['N_acc']:.1f} T {m['T_acc']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(model=dict(kind="pair/union selector", K=K, C=bp["C"], p_sel=p_sel,
                          use_ens=use_ens,
                          base_classifier="study-5 logreg C=%.6g" % st5["C"],
                          source_study=str(args.study)),
               metrics=m, n_val_records=n,
               history={"ensemble_threshold_union": {"cIoU": 34.9419, "gIoU": 69.7711,
                                                     "N_acc": 89.11, "T_acc": 77.22},
                        "single_probe": {"cIoU": 35.0884, "gIoU": 68.3729}},
               generated="exp79_pair_full_val.py")
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
