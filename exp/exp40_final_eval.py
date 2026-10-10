#!/usr/bin/env python
"""EXP-40 — final evaluation of the tuned method on an official val split.

Applies exactly the rule chosen by exp39_optuna.py (held-out tuning split) to
the official validation records and writes exp/out_final/main_<ds>.json with
the "shipped" block the SOTA bar reads.

Rows written:
  released            unmodified released HybridGL pipeline
  snap                containment growth applied to the released selection
  ours                tuned box-prior pipeline (method under test)
  ours+grow           ... with the growth repair when the tuned rule uses it
  ctrl_no_box         same rule with the box prior removed (one-line control)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_map import cache_file  # noqa: E402
from vlm_boxes import QwenBoxer  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
sys.path.insert(0, str(REPRO))

PUBLISHED = {
    "refcoco": dict(mIoU=58.42, oIoU=50.85, rival="GLOD-RIS (TCSVT 2026)"),
    "refcoco+": dict(mIoU=54.01, oIoU=45.96, rival="GLOD-RIS (TCSVT 2026)"),
    "refcocog": dict(mIoU=57.85, oIoU=50.03, rival="GLOD-RIS (TCSVT 2026)"),
    "grefcoco": dict(cIoU=68.94, gIoU=74.34, rival="InstAlign (arXiv 2411.15087)"),
}


def rank01(v):
    if len(v) <= 1:
        return np.zeros_like(v)
    return np.argsort(np.argsort(v)).astype(np.float32) / (len(v) - 1)


def box_overlap(masks, box, mode):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = masks.shape[1], masks.shape[2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(len(masks), np.float32)
    sub = masks[:, y0:y1, x0:x1]
    a_in = sub.sum((1, 2)).astype(np.float32)
    box_area = float((y1 - y0) * (x1 - x0))
    tot = masks.sum((1, 2)).astype(np.float32)
    if mode == "cover":
        return a_in / box_area
    if mode == "inside":
        return a_in / np.maximum(tot, 1)
    return a_in / (tot + box_area - a_in + 1e-9)


def select(r, masks, box, params, use_box=True):
    n = len(r["gt_iou"])
    if params["dec_source"] == "sc_x_gem":
        base = rank01(np.asarray(r["sc"][:n], np.float32)) + params["gem_mix"] * rank01(
            np.asarray(r["score_gem"][:n], np.float32))
    else:
        base = np.asarray(r[{"sc": "sc", "sc_noun": "sc_noun", "sc_strip": "sc_strip",
                             "gem": "score_gem"}[params["dec_source"]]][:n], np.float32)
    k = min(int(params["k_dec"]), n)
    idx = np.argsort(base)[::-1][:k] if k < n else np.arange(n)
    if use_box and box is not None:
        loc = box_overlap(masks, box, params["mode"])[idx]
        if params["combine"] == "loc_only":
            pick = int(np.argmax(loc))
        elif params["combine"] == "add3":
            gem = np.asarray(r["score_gem"][:n], np.float32)[idx]
            pick = int(np.argmax(rank01(loc) + params["beta"] * rank01(base[idx])
                                 + params["gem_mix"] * rank01(gem)))
        else:
            rl, rb = rank01(loc), rank01(base[idx])
            if params["combine"] == "mul":
                pick = int(np.argmax((rl + params["eps"]) * (rb + params["eps"]) ** params["beta"]))
            else:
                pick = int(np.argmax(rl + params["beta"] * rb))
        j = int(idx[pick])
        if params["grow"]:
            j = int(r["ps_idx"][j])
        return j, True
    j = int(idx[0])
    return j, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phrase-source", default="noun", choices=["noun", "sent"])
    ap.add_argument("--params", type=Path, default=Path("exp/out_final/optuna_study.json"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-ious", type=Path, default=None,
                    help="optional path: dump per-record selected IoU (ours row) for paired stats")
    ap.add_argument("--box-source", default="llava", choices=["llava", "glm"],
                    help="cache namespace of the box prior (glm reads exp/out_glm_boxes)")
    ap.add_argument("--min-coverage", default="98.0",
                    help="refuse to write shipped.mIoU below this box coverage %%")
    args = ap.parse_args()

    from glod.data import RefDataset
    ds = RefDataset(REPRO / "data/refer", args.dataset, args.split,
                    limit=args.limit, seed=args.seed)
    ref_by_id = {r["ref_id"]: r for r in ds.refs}
    recs = [json.loads(l) for l in open(args.records)]
    params = json.loads(args.params.read_text())["best_params"]
    print(f"{len(recs)} records; params={params}", flush=True)

    boxer = QwenBoxer.__new__(QwenBoxer)
    store, order = {}, []

    def masks_of(fn):
        if fn in store:
            return store[fn]
        with np.load(cache_file(fn), allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["mask_shape"])
            mm = np.unpackbits(z["mask_bits"], count=int(np.prod(shape))
                               ).reshape(shape).astype(bool)
        store[fn] = mm
        order.append(fn)
        while len(order) > 8:
            store.pop(order.pop(0), None)
        return mm

    dumps = []
    acc = {}
    oacc = {}
    pacc = {}          # pooled oIoU: sum_j inter_j
    pucc = {}          # pooled oIoU: sum_j union_j
    n_used = 0
    n_box = 0
    n_boxvalue = 0
    for i, r in enumerate(recs):
        ref = ref_by_id.get(r["ref_id"])
        if ref is None:
            continue
        fn = ds.images[ref["image_id"]]["file_name"]
        masks = masks_of(fn)
        n = len(masks)
        if n != len(r["gt_iou"]):
            continue
        gt = np.asarray(r["gt_iou"], np.float32)
        ph = r["parsed"]["sentence"] if args.phrase_source == "sent" else r["parsed"]["noun_phrase"]
        if args.box_source == "glm":
            import hashlib as _h
            bp = Path("exp/out_glm_boxes") / (
                _h.sha1(f"glm|{fn}|{ph.strip().lower()}".encode()).hexdigest() + ".json")
        else:
            bp = boxer._path(fn, ph.strip())
        queried = bp.is_file()
        box = json.loads(bp.read_text())["box"] if queried else None
        n_used += 1
        # Coverage counts QUERIED records: a cached "none" answer is a decided
        # model output, not a missing query.  Only a missing cache entry means
        # the VLM was never asked.
        n_box += queried
        n_boxvalue += box is not None

        j_ours, had = select(r, masks, box, params, use_box=True)
        j_ctrl, _ = select(r, masks, None, params, use_box=False)
        if args.dump_ious is not None:
            _inter = float(np.asarray(r["inter_gt"], np.float64)[j_ours])
            _union = float(np.asarray(r["union_gt"], np.float64)[j_ours])
            dumps.append(dict(ref_id=r["ref_id"], sent_idx=r.get("sent_idx"),
                              iou=float(gt[j_ours]),
                              oiou=float(_inter / _union) if _union > 0 else 0.0))
        rows = {
            "released": int(r["final"]),
            "snap": int(r["ps_idx"][int(r["final"])]),
            "ours": j_ours,
            "ctrl_no_box": j_ctrl,
            "orc_single": int(r["orc"]),
        }
        for k, j in rows.items():
            acc[k] = acc.get(k, 0.0) + float(gt[j])
            if k != "orc_single":
                inter = float(np.asarray(r["inter_gt"], np.float64)[j])
                union = float(np.asarray(r["union_gt"], np.float64)[j])
                oacc[k] = oacc.get(k, 0.0) + (inter / union if union > 0 else 0.0)
                pacc[k] = pacc.get(k, 0.0) + inter
                pucc[k] = pucc.get(k, 0.0) + union
    res = {k: 100.0 * v / n_used for k, v in acc.items()}
    # 2026-10-08 protocol alignment: GLOD-RIS defines oIoU as the GLOBAL overlap
    # (sum of intersections / sum of unions), not the per-sample mean.  The
    # earlier `oIoU` payload field held a per-sample mean (identical to mIoU)
    # and is replaced by the pooled ratio here.
    pooled = {k: (100.0 * pacc[k] / pucc[k] if pucc[k] > 0 else 0.0) for k in pacc}
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {k:14s} {v:7.3f}")

    coverage = 100.0 * n_box / max(1, n_used)
    payload = dict(
        tag=args.tag or args.dataset, n=n_used, box_coverage=coverage,
        box_rate=100.0 * n_boxvalue / max(1, n_used),
        params=params, phrase_source=args.phrase_source,
        rows={k: round(v, 4) for k, v in res.items()},
        oIoU={k: round(v, 4) for k, v in pooled.items()},
        oIoU_masses={k: dict(inter=pacc[k], union=pucc[k]) for k in pacc},
        oIoU_definition=("global pooled oIoU: 100 * sum_j |pred_j ∩ GT_j| / "
                         "sum_j |pred_j ∪ GT_j| over all val records "
                         "(GLOD-RIS: 'oIoU measures global overlap'); the "
                         "per-sample mean IoU is the rows/mIoU field"),
        mIoU=round(res["ours"], 4),
        best_published=PUBLISHED[args.dataset],
        generated="exp40_final_eval.py",
    )
    # Guard: never write "shipped" from a partial box cache. A missing box
    # silently falls back to the released selection and would understate the
    # method (or, worse, ship a number produced by an incomplete run).
    if coverage < float(args.min_coverage):
        payload["shipped"] = None
        payload["incomplete"] = True
        print(f"REFUSING to write shipped: box coverage {coverage:.1f}% < "
              f"{args.min_coverage}%. Re-run after extraction finishes.")
    else:
        payload["shipped"] = dict(mIoU=round(res["ours"], 4),
                                  best_published={"mIoU": PUBLISHED[args.dataset]["mIoU"],
                                                  "rival": PUBLISHED[args.dataset]["rival"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    if args.dump_ious is not None and dumps:
        args.dump_ious.write_text(json.dumps(dumps))
        print(f'wrote {args.dump_ious} ({len(dumps)} rows)')
    print(f"wrote {args.output}: ours mIoU {res['ours']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
