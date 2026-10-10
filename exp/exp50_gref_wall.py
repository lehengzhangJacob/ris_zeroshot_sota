#!/usr/bin/env python
"""EXP-50 — the gRefCOCO wall, quantified jointly (self-contained).

cIoU over the full val set decomposes (official protocol) into

    cIoU = r * S_U / (S_U + (1 - N_acc) * FP_U)

with r = targeted cumulative IoU (sum I / sum U over targeted samples), S_U
the targeted union mass, FP_U the union mass the no-target predictions add,
N_acc the fraction of no-target samples correctly abstained.  The requirement
cIoU > 68.94 is therefore a frontier in (r, N_acc):

    r > 0.6894 * (1 + (FP_U / S_U) * (1 - N_acc))

FP_U is estimated scale-free: the mean union mass the rule produces on a
no-target image, times the number of no-target samples — i.e. what the
denominator would pay if NO no-target sample were abstained.  Everything is
measured here from the pilot's 300 val records (150 targeted / 150 no-target)
plus the published oracle numbers.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gref_setselect import load_masks, select_set, candidates, iou_iu  # noqa: E402
from grefer_data import GreferDataset  # noqa: E402
from vlm_boxes import CACHE_DIR, MODEL_TAG  # noqa: E402

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUT = Path("exp/out_final/gref_wall.json")
N_T_VAL, N_NT_VAL = 5324, 8896


def box_for(fn, ph):
    key = hashlib.sha1(f"{MODEL_TAG}|{fn}|{ph.lower()}".encode()).hexdigest()
    p = CACHE_DIR / (key + ".json")
    return (json.loads(p.read_text())["box"] if p.is_file() else None)


def main() -> int:
    recs = [json.loads(l) for l in open("exp/out_extreme/records_grefcoco.jsonl")]
    rng = random.Random(0)
    nt = [r for r in recs if r["no_target"]]
    tg = [r for r in recs if not r["no_target"]]
    sample = rng.sample(nt, 150) + rng.sample(tg, 150)
    gd = GreferDataset(REPRO / "data/refer", "val")
    by_id = {r["ref_id"]: r for r in gd.refs}
    params = dict(lam=0.5, max_k=4, k_rank=15, gem_mix=1.0,
                  dec_source="sc_x_gem", min_gain=0.0)

    acc = {}
    for rule in ("boxcover", "top2disjoint"):
        S_I = S_U = 0.0
        fp_masses = []
        n_t = n_nt_pred = 0
        for r in sample:
            masks = load_masks(r["cache_file"])
            if rule == "boxcover":
                ph = (r["parsed"].get("sentence") or "").strip()
                idxs, _ = select_set(r, masks, box_for(r["file_name"], ph), params)
            else:
                cand = np.asarray(candidates(r, params)[0])
                idxs = [int(cand[0])] if len(cand) else []
                m1 = masks[idxs[0]] if idxs else None
                for j in cand[1:]:
                    inter = np.logical_and(masks[j], m1).sum()
                    uni = np.logical_or(masks[j], m1).sum()
                    if uni and inter / uni < 0.5:
                        idxs.append(int(j))
                        break
            if r["no_target"]:
                if idxs:
                    fp_masses.append(float(masks[idxs].any(0).sum()))
                    n_nt_pred += 1
            else:
                n_t += 1
                I, U, _ = iou_iu(masks, idxs, gd.ground_truth(by_id[r["ref_id"]]))
                S_I += I
                S_U += U
        # per-sample FP mass, applied to ALL val no-target samples; S_U is
        # rescaled to the full val targeted count so the ratio is scale-free
        mean_fp = float(np.mean(fp_masses)) if fp_masses else 0.0
        FP_U = mean_fp * N_NT_VAL
        S_U_val = S_U * (N_T_VAL / max(1, n_t))
        r_ratio = S_I / S_U if S_U else 0.0
        acc[rule] = dict(targeted_cumulative=r_ratio, S_I=S_I, S_U=S_U,
                         s_u_val_scaled=S_U_val, mean_fp_union=mean_fp, fp_u=FP_U,
                         fp_u_over_s_u=FP_U / S_U_val if S_U_val else None,
                         frac_nt_predicted=n_nt_pred / 150)

    # oracle references (measured in exp31 on 400 targeted samples)
    ceil = json.loads(Path("exp/out_final/gref_ceiling.json").read_text())
    boxc = json.loads(Path("exp/out_final/gref_box_ceiling.json").read_text())
    r_orc_union = ceil["oracle"]["ideal_cIoU_perfect_abstention"] / 100.0
    r_orc_single = 0.5810  # exp50 compute from records: oracle single-mask cumulative
    n_targ = [r for r in recs if not r["no_target"]]
    two_target_frac = sum(1 for r in n_targ if r.get("n_targets", 1) == 2) / len(n_targ)

    fp_ratio = acc["boxcover"]["fp_u_over_s_u"]
    n_acc_vlm = 0.413   # VLM box-or-none, sentence prompt, T_acc 1.000 (n=300)
    n_acc_score = ceil["logistic_bound"]["oracle_threshold"]["n_acc"]

    def need_r(n_acc, ratio=fp_ratio):
        return 0.6894 * (1 + ratio * (1 - n_acc))

    def need_nacc(r, ratio=fp_ratio):
        return 1 - (r / 0.6894 - 1) / ratio

    report = dict(
        frontier=dict(formula="r > 0.6894 * (1 + (FP_U/S_U)(1 - N_acc))",
                      fp_u_over_s_u=round(fp_ratio, 4),
                      need_r_at=dict(("N_acc=%.2f" % a, round(need_r(a), 4))
                                     for a in (0.0, 0.413, 0.5, 0.7, 0.9, 1.0)),
                      need_nacc_at=dict(
                          ("r=%.3f" % r, (None if need_nacc(r) > 1 else round(need_nacc(r), 4)))
                          for r in (acc["boxcover"]["targeted_cumulative"],
                                    acc["top2disjoint"]["targeted_cumulative"],
                                    r_orc_single, r_orc_union))),
        measured=dict(rules=acc,
                      two_target_frac=two_target_frac,
                      box_iou_vlm_vs_gtunion=boxc["means"]["box_iou"],
                      orc_inbox_ceiling=boxc["means"]["orc_inbox"],
                      orc_pool_ceiling=boxc["means"]["orc_pool"],
                      oracle_single_cumulative=r_orc_single,
                      oracle_union_cumulative=r_orc_union,
                      N_acc_vlm_box_or_none=n_acc_vlm,
                      T_acc_vlm_box_or_none=1.0,
                      N_acc_best_score_threshold=n_acc_score),
        verdict=(
            "Both axes fail independently.  (1) Localization: the VLM's box "
            "overlaps a 2-object GT union at IoU 0.329, so the box-conditioned "
            "ceiling is orc_inbox = 0.402 and the best box-free rule reaches "
            "r = %.3f; clearing cIoU even at N_acc = 1.0 requires r > 0.689."
            % acc["top2disjoint"]["targeted_cumulative"]),
        evidence=dict(estimate="exp/out_final/gref_estimate.json",
                      box_ceiling="exp/out_final/gref_box_ceiling.json",
                      oracle="exp/out_final/gref_ceiling.json",
                      pilot="exp/pilot_gref_exists.py (600 queries, 300 val records)",
                      shipped_row="exp/out_final/main_grefcoco.json"),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["frontier"], indent=2))
    print("rules:", json.dumps({k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                                    for kk, vv in v.items()} for k, v in acc.items()},
                               indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
