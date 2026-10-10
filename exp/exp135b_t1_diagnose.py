#!/usr/bin/env python
"""EXP-135b — diagnostic for the T1 paraphrase test (no model re-run).

Rebuilds exactly the T1 sample of exp135_stability.py from the saved records,
reads the paraphrased boxes from the para1 cache, and reports, per record:
box IoU(original, paraphrased) for the substituted terms, whether the mask
union changes, and — where it does not — which other rule parts already cover
the substituted pick.  Verifies the zero-delta T1 result before it ships.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp62_clause_selector import make_clf  # noqa: E402
from exp78_pair_selector import build_candidates, candidate_mask, precompute_row, topk_mask  # noqa: E402
from exp79_pair_full_val import load_rows  # noqa: E402
from exp108_gdino_study import box_feats, lpath  # noqa: E402
from exp109a_precompute_val_gdino_sam import spath, terms_of  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

OUTF = Path("exp/out_final")
RECORDS = Path("exp/out_extreme/records_grefcoco_train_tune.jsonl")
PARA_TAG = f"{MODEL_TAG}|para1"


def spath1(fn, term):
    key = hashlib.sha1(f"{MODEL_TAG}|gdino_sam_d1|{fn}|{term.lower()}".encode()).hexdigest()
    return CACHE_DIR.parent / "out_val_gdino_sam_d1" / (key + ".npz")


def load_masks_npz(p):
    if not p.is_file() or p.stat().st_size == 0:
        return None
    with np.load(p) as z:
        shape = tuple(int(x) for x in z["shape"])
        bits = np.unpackbits(z["bits"], count=int(np.prod(shape))).reshape(shape)
    return [bits[i].astype(bool) for i in range(shape[0])]


def para_box(fn, ph):
    k = hashlib.sha1(f"{PARA_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    p = CACHE_DIR / (k + ".json")
    if not p.is_file():
        return "missing"
    return json.loads(p.read_text()).get("box")


def paint(box, H, W):
    m = np.zeros((200, 200), bool)
    if box is None:
        return m
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, x1 = max(0, min(199, x0 * 200 // W)), max(0, min(199, x1 * 200 // W))
    y0, y1 = max(0, min(199, y0 * 200 // H)), max(0, min(199, y1 * 200 // H))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def main() -> int:
    st5 = json.loads((OUTF / "optuna_study_gref5.json").read_text())["best_params"]
    g8 = json.loads((OUTF / "optuna_study_gref8.json").read_text())["best_params"]
    cfg = dict(json.loads((OUTF / "optuna_study_gref14.json").read_text())["best_params"])
    if "cap" in cfg and "sam_area_cap" not in cfg:
        cfg["sam_area_cap"] = cfg.pop("cap")
    p_sel = float(g8["p_sel"])

    clf = make_clf("logreg", dict(C=st5["C"]))
    tr = load_rows(RECORDS, "train", clf)
    clf.fit(np.concatenate([r["X"] for r in tr]), np.concatenate([r["y"] for r in tr]))
    for r in tr:
        r["_bs"] = clf.predict_proba(r["X"])[:, 1]
    Xs, ys = [], []
    for r in tr:
        F, y, _, _ = precompute_row(r, r["_bs"])
        km = topk_mask(len(F), len(r["_bs"]), g8["K"])
        Xs.append(F[km]); ys.append(y[km])
    pair_clf = make_clf("logreg", dict(C=g8["C"]))
    pair_clf.fit(np.concatenate(Xs), np.concatenate(ys))

    val = load_rows(RECORDS, "train", clf)
    for r in val:
        r["_bs"] = clf.predict_proba(r["X"])[:, 1]

    # same sample selection as exp135 T1: first 60 non-no_target rows with terms
    terms_by_rid = {}

    def terms_for(rid, fn):
        p = lpath(fn, "")
        return None

    sample_rids = []
    for rec_line in open(RECORDS):
        rec = json.loads(rec_line)
        if rec.get("no_target"):
            continue
        ts = terms_of(rec["parsed"])
        if not ts:
            continue
        terms_by_rid[rec["ref_id"]] = (rec["file_name"], ts)
        sample_rids.append(rec["ref_id"])
        if len(sample_rids) >= 60:
            break

    by_rid = {r["rid"]: r for r in val}
    n_sub = n_changed_box = n_changed_pred = 0
    ious = []
    for rid in sample_rids:
        r = by_rid.get(rid)
        if r is None:
            continue
        fn, ts = terms_by_rid[rid]
        pp = pair_clf.predict_proba(build_candidates(r, r["_bs"], r["_box"], g8["K"])[0])[:, 1]
        if (pp.max() < p_sel) or any(r["ens"]):
            continue
        term = ts[0]
        b_orig = None
        p = lpath(fn, term)
        if p.is_file():
            b_orig = json.loads(p.read_text()).get("box")
        b_para = para_box(fn, term)
        if b_para == "missing":
            print(f"  rid {rid} term {term!r}: paraphrase cache MISSING")
            continue
        n_sub += 1
        if b_orig is not None and b_para is not None:
            H, W = r["masks"].shape[1] * 2, r["masks"].shape[2] * 2
            A, B = paint(b_orig, H, W), paint(b_para, H, W)
            u = (A | B).sum()
            ious.append(float((A & B).sum() / u) if u else 1.0)
        if b_orig != b_para:
            n_changed_box += 1
    print(f"substitutions: {n_sub}, boxes differ: {n_changed_box}, "
          f"comparable IoU pairs: {len(ious)} mean {np.mean(ious) if ious else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
