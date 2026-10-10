#!/usr/bin/env python
"""EXP-52 — fold the rule-realization grid into the wall artifact.

Aggregates exp50 (frontier + measured operating points) and exp51 (deployable
rules x shortlist depth x box source) into exp/out_final/gref_wall.json, and
emits the three-axis verdict:

  axis 1  grounder:   VLM box IoU 0.329 vs a perfect box
  axis 2  selector:   deployable rules realize ~69% of the perfect-box oracle
  axis 3  abstention: N_acc 0.413 vs the >= 0.91 (cIoU) / >= 0.74 (gIoU) the
                      oracle-with-perfect-box ceiling still requires

Each axis is measured; the conjunction fails on every axis simultaneously.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("exp/out_final/gref_wall.json")


def main() -> int:
    wall = json.loads(OUT.read_text())
    rules = json.loads(Path("exp/out_final/gref_rules.json").read_text())
    boxc = json.loads(Path("exp/out_final/gref_box_ceiling.json").read_text())
    ceil = json.loads(Path("exp/out_final/gref_ceiling.json").read_text())

    c = rules["cells"]
    fp_ratio = wall["frontier"]["fp_u_over_s_u"]

    def need_nacc(r):
        return 1 - (r / 0.6894 - 1) / fp_ratio

    def need_nacc_giou(mean_iou, n_t=5324, n_nt=8896, total=14220, floor=0.7434):
        return (floor * total - mean_iou * n_t) / n_nt

    decomp = {
        "axis1_grounder": dict(
            vlm_box_iou=boxc["means"]["box_iou"],
            perfect_box_iou=1.0,
            note="VLM box vs 2-object GT union"),
        "axis2_selector": {
            f"{k}": dict(mean_iou=v["mean_iou"], r=v["r"],
                         frac_of_perfect_box_oracle=round(v["r"] / 0.7399, 3))
            for k, v in c.items() if k.endswith("|gt")},
        "axis2_ceilings": dict(
            oracle_perfect_box_k15=0.7399, oracle_perfect_box_k50=0.8832,
            oracle_no_box_k50=0.8813,
            note="oracle = GT-guided greedy union over the same shortlist"),
        "axis3_abstention": dict(
            direct_probe_train=dict(N_acc=0.875, T_acc=0.920),
            direct_probe_val=dict(N_acc=0.804, T_acc=0.932),
            shipped_combination_val=dict(N_acc=0.867, T_acc=0.797),
            achieved_N_acc=wall["measured"]["N_acc_vlm_box_or_none"],
            best_score_threshold=wall["measured"]["N_acc_best_score_threshold"],
            trained_sota_N_acc=0.797,
            note="InstAlign N-acc on gRefCOCO val (published)"),
        "requirements": {
            "cIoU_at_realizable_stack": dict(r=c["box_cover|k15|vlm"]["r"],
                                             need_N_acc=None),
            "cIoU_at_perfect_box_realizable_rule": dict(
                r=c["box_cover|k15|gt"]["r"],
                need_N_acc=None),
            "cIoU_at_oracle_perfect_box": dict(
                r=0.7399, need_N_acc=round(need_nacc(0.7399), 3)),
            "gIoU_at_oracle_perfect_box": dict(
                mean_iou=boxc["means"]["orc_in_gtbox"],
                need_N_acc=round(need_nacc_giou(boxc["means"]["orc_in_gtbox"]), 3)),
        },
        "verdict": (
            "Three independent axes, all measured, all short. (1) Grounder: the "
            "VLM box overlaps the 2-object union at 0.329. (2) Selector: even "
            "given a PERFECT box the best deployable rule reaches r=0.507 vs the "
            "oracle's 0.740 - and r=0.507 cannot clear cIoU at ANY N_acc. "
            "(3) Abstention: the cIoU floor at the oracle-with-perfect-box "
            "ceiling still needs N_acc>=0.91, and gIoU needs >=0.74; the best "
            "training-free existence signal measures 0.413 and the trained "
            "state of the art is 0.797. No realizable combination clears either "
            "line. UPDATE: the abstention axis was subsequently SOLVED to a "
            "useful point (direct yes/no probe 0.875@0.920 train, 0.804@0.932 "
            "val; shipped combination 0.867@0.797, above the trained SOTA "
            "0.797). What remains is selector-bound: both floors need the "
            "selector at >=78% of the k=50 oracle; shipped realization 55.4%; "
            "best ever measured (perfect box + current rule) 62.4%."),
    }
    wall["decomposition"] = decomp
    OUT.write_text(json.dumps(wall, indent=2))
    print(json.dumps(decomp["requirements"], indent=2))
    print(json.dumps(decomp["axis2_selector"], indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
