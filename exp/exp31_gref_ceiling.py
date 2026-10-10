#!/usr/bin/env python
"""EXP-31 — how high can ANY selector go on gRefCOCO val with this pool?

Two questions, both answerable offline from the stored records + mask cache:

  A. no-target separability: among the scores the pipeline already computes,
     how well can a threshold possibly separate no-target from targeted
     samples?  (oracle threshold = upper bound for any threshold rule)
  B. set-oracle: for targeted samples, what is the best achievable IoU of a
     *union of pool masks* vs the GT union?  (greedy forward selection over
     the top candidates, exact mask arithmetic)

These give the (N_acc, mean-IoU) frontier; combined with the official metric
formulas they say whether cIoU > 68.94 / gIoU > 74.34 is reachable at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grefer_data import GreferDataset  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
RECORDS = Path("exp/out_extreme/records_grefcoco.jsonl")
OUT = Path("exp/out_final/gref_ceiling.json")

N_NO_TARGET = 8896  # samples whose GT is empty
N_TOTAL = 14220


def auc(scores, labels):
    """Rank AUC (labels: True = positive class)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def best_threshold(score, no_target, min_tacc=0.98):
    """Oracle threshold: max N_acc subject to T_acc >= min_tacc.

    no_target: bool array (True = sample has no target).
    Predict no-target when score < thr (low evidence of presence).
    """
    order = np.argsort(score)
    s_sorted = score[order]
    nt_sorted = no_target[order]
    # candidate thresholds: midpoints between distinct values
    total_nt = nt_sorted.sum()
    total_t = len(nt_sorted) - total_nt
    # cumulative: samples with score < s are predicted no-target
    c_nt = np.cumsum(nt_sorted)          # no-target correctly abstained
    c_t = np.cumsum(~nt_sorted)          # targeted wrongly abstained
    tacc = 1.0 - c_t / total_t
    nacc = c_nt / total_nt
    ok = tacc >= min_tacc
    if not ok.any():
        return None
    i = np.flatnonzero(ok)[np.argmax(nacc[ok])]
    return dict(threshold=float(s_sorted[i]), n_acc=float(nacc[i]), t_acc=float(tacc[i]))


def load_masks(cache_file):
    with np.load(cache_file, allow_pickle=False) as d:
        shape = tuple(int(x) for x in d["mask_shape"])
        bits = d["mask_bits"]
    masks = np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)
    return masks


def greedy_set_iou(masks, gt, cand, max_steps=6, stride=2):
    """Best union-of-candidates IoU vs gt by greedy forward selection.

    Returns (iou, chosen_idx, I, U) with I/U measured on the full-resolution
    masks of the chosen set (exact, for the cumulative metric).
    """
    sub = masks[cand][:, ::stride, ::stride]
    g = gt[::stride, ::stride]
    if not g.any():
        return 0.0, [], 0, 0
    cur = np.zeros_like(g)
    best_hist, chosen = [], []
    for _ in range(max_steps):
        best_i, best_iou, best_u = -1, -1.0, None
        for j in range(len(cand)):
            u = cur | sub[j]
            iou = (u & g).sum() / max(1, (u | g).sum())
            if iou > best_iou:
                best_iou, best_i, best_u = iou, j, u
        if best_i < 0 or (best_hist and best_iou <= best_hist[-1] + 1e-6):
            break
        cur = best_u
        chosen.append(int(cand[best_i]))
        best_hist.append(best_iou)
    if not chosen:
        return 0.0, [], 0, 0
    pred = masks[chosen].any(0)
    I = int(np.logical_and(pred, gt).sum())
    U = int(np.logical_or(pred, gt).sum())
    return best_hist[-1], chosen, I, U


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--limit", type=int, default=0, help="limit records processed (0=all)")
    ap.add_argument("--oracle-limit", type=int, default=400,
                    help="how many targeted records to run the mask oracle on")
    ap.add_argument("--candidates", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.records.open()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} records", flush=True)

    # ---------------- Part A: no-target separability -----------------------
    feats = {}
    for name in ("sc", "sc_strip", "sc_noun", "sc_content"):
        feats[name] = np.array([float(np.max(r[name])) for r in recs])
    feats["gem"] = np.array([float(np.max(r["score_gem"])) for r in recs])
    feats["gem_top1"] = np.array([float(np.sort(r["score_gem"])[-1]) for r in recs])
    npool = np.array([len(r["gt_iou"]) for r in recs])
    nt = np.array([bool(r["no_target"]) for r in recs])
    print(f"no-target {int(nt.sum())} / targeted {int((~nt).sum())}", flush=True)

    sep = {}
    for name, v in feats.items():
        a = auc(v, ~nt)  # positive class = targeted (has target)
        bt = best_threshold(v, nt)
        sep[name] = dict(auc_targeted_vs_not=a, oracle_threshold=bt)
        print(f"  {name:10s} AUC(target has target)={a:.3f} "
              f"oracle N_acc@T_acc>=0.98 -> {bt}", flush=True)

    # multi-feature bound: logistic regression on the scalars (fitted on all
    # data => optimistic upper bound, not a deployable rule)
    X = np.stack([feats[k] for k in ("sc", "sc_strip", "sc_noun", "sc_content", "gem")], 1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    y = (~nt).astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        w -= 0.5 * g
    p = 1 / (1 + np.exp(-X @ w))
    a_lr = auc(p, ~nt)
    bt_lr = best_threshold(p, nt)
    print(f"  logistic(5 scores) AUC={a_lr:.3f} oracle N_acc -> {bt_lr}", flush=True)

    # ---------------- Part B: set oracle on targeted samples ---------------
    gd = GreferDataset(REPRO / "data/refer", "val")
    ref_by_id = {r["ref_id"]: r for r in gd.refs}
    gt_cache = {}

    target_idx = [i for i, r in enumerate(recs) if not r["no_target"]]
    rng = np.random.RandomState(0)
    if args.oracle_limit and args.oracle_limit < len(target_idx):
        pick = rng.choice(len(target_idx), args.oracle_limit, replace=False)
        sample = [target_idx[i] for i in sorted(pick)]
    else:
        sample = target_idx

    ious, I_list, U_list, chosen_sizes = [], [], [], []
    import time
    t0 = time.time()
    for n, i in enumerate(sample):
        r = recs[i]
        rid = r["ref_id"]
        if rid not in gt_cache:
            gt_cache[rid] = gd.ground_truth(ref_by_id[rid])
        gt = gt_cache[rid]
        order = np.argsort(r["gt_iou"])[::-1][: args.candidates]
        masks = load_masks(r["cache_file"])
        iou, chosen, I, U = greedy_set_iou(masks, gt, order)
        ious.append(iou)
        I_list.append(I)
        U_list.append(U)
        chosen_sizes.append(len(chosen))
        if (n + 1) % 100 == 0:
            print(f"  oracle {n+1}/{len(sample)} mean_iou_so_far={np.mean(ious):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ious = np.array(ious)
    I_arr, U_arr = np.array(I_list, float), np.array(U_list, float)
    miou = float(ious.mean())
    bs = [np.mean(rng.choice(ious, len(ious))) for _ in range(2000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    # Ideal metrics: perfect abstention on no-target + oracle sets on targeted.
    scale = (N_TOTAL - N_NO_TARGET) / len(sample)
    ideal_giou = 100.0 * (N_NO_TARGET * 1.0 + ious.sum() * scale) / N_TOTAL
    ideal_ciou = 100.0 * I_arr.sum() / U_arr.sum()   # cumulative, no-target FPs = 0
    # FP sensitivity: if x% of no-target samples are wrongly non-empty, their
    # masks add to the cIoU denominator.  Estimate the average FP union size
    # from the targeted oracle U's (same object scale), and report the cIoU at
    # several N_acc levels.
    mean_U = float(U_arr.mean())
    fp_rows = {}
    for n_acc in (0.5, 0.7, 0.8, 0.9, 1.0):
        n_fp = (1 - n_acc) * N_NO_TARGET
        ciou = 100.0 * (I_arr.sum() * scale) / (U_arr.sum() * scale + n_fp * mean_U)
        giou = 100.0 * (N_NO_TARGET * n_acc + ious.sum() * scale) / N_TOTAL
        fp_rows[f"n_acc={n_acc:.1f}"] = dict(cIoU=ciou, gIoU=giou)

    report = dict(
        n_records=len(recs),
        n_no_target=int(nt.sum()),
        separability=sep,
        logistic_bound=dict(auc=a_lr, oracle_threshold=bt_lr),
        oracle=dict(
            n_sampled=len(sample),
            mean_set_iou=miou,
            ci=[lo, hi],
            mean_chosen_masks=float(np.mean(chosen_sizes)),
            frac_iou_ge_0_5=float((ious >= 0.5).mean()),
            frac_iou_ge_0_7=float((ious >= 0.7).mean()),
            ideal_gIoU_perfect_abstention=ideal_giou,
            ideal_cIoU_perfect_abstention=ideal_ciou,
            scaling=dict(scale=scale, mean_U_toward_cumulative=mean_U),
            n_acc_sensitivity=fp_rows,
        ),
        generated="exp31_gref_ceiling.py",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
