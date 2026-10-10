#!/usr/bin/env python
"""EXP-7 — Why does the spatial selector stall at the guided operating point?

Diagnostics that separate the two failure locations:
  set    the oracle proposal is not in the candidate set the rule is allowed to
         choose from
  pick   the oracle IS in the set but the resolver chooses something else

Also measures the structural limit of the released guidance: `maxidxs =
topk(score_clip, k1=3)` means the final answer is always one of the three
highest-scoring proposals under the FULL expression score. If the reference is
not in that set, no amount of relation or coherence guidance can recover it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, confusion, resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=Path("exp/out_exp4/records_guided_val.jsonl"))
    ap.add_argument("--pairs", type=Path, default=Path("exp/out_exp4/pairs_guided_val.npz"))
    ap.add_argument("--select-refs", type=Path, default=Path("exp/select_refs_s101.json"))
    ap.add_argument("--output", type=Path, default=Path("exp/out_exp7"))
    ap.add_argument("--tag", default="refcoco_guided")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    recs = [Rec(json.loads(l)) for l in args.records.open()]
    pairs = np.load(args.pairs)
    for r in recs:
        r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
    print(f"{len(recs)} records", flush=True)

    out = {}
    n = len(recs)

    # --- structural limit of the released guidance
    in_top3 = np.array([r.orc in set(r.top.tolist()) for r in recs])
    out["oracle_in_released_top3"] = dict(n=int(in_top3.sum()), frac=float(in_top3.mean()))
    # what the released guidance would score with a perfect chooser restricted
    # to its own top-3 set
    perfect_top3 = np.array([r.gt[r.top[np.argmax(r.S["sc"][r.top])]] if False else
                             max(r.gt[i] for i in r.top) for r in recs])
    out["ceiling_top3_chooser_mIoU"] = float(perfect_top3.mean())
    out["released_mIoU"] = float(np.mean([r.gt[r.final] for r in recs]))

    # --- set/pick decomposition for each candidate-set definition
    rows = {}
    for key in ("sc", "sc_strip", "sc_content", "sc_noun"):
        for k in (3, 5, 8, 12):
            for delta in (0.20, 0.35, 1.01):
                oracle_in = 0
                pick_oracle = 0
                fired = 0
                gain = 0.0
                base_correct_fired = 0
                for r in recs:
                    cand = confusion(r, key, k, delta, 0.20)
                    if r.rel is None or len(cand) < 2:
                        continue
                    fired += 1
                    if r.orc in cand:
                        oracle_in += 1
                    pick = resolve(r, cand, r.rel)
                    if pick is not None and pick == r.orc:
                        pick_oracle += 1
                    if r.gt[r.final] >= 0.5:
                        base_correct_fired += 1
                    if pick is not None:
                        gain += r.gt[pick] - r.gt[r.final]
                rows[f"{key}_k{k}_d{delta}"] = dict(
                    fired=fired, oracle_in_set=oracle_in,
                    oracle_in_set_rate=oracle_in / max(fired, 1),
                    pick_oracle=pick_oracle,
                    pick_given_in=pick_oracle / max(oracle_in, 1),
                    released_already_correct=base_correct_fired,
                    released_already_correct_rate=base_correct_fired / max(fired, 1),
                    total_gain=gain, delta=gain / n)
    out["rules"] = rows

    # --- what does the released module do on the wrong-instance stratum?
    wi = [r for r in recs if r.gt[r.final] < 0.1 and r.gt[r.orc] > 0.6]
    out["wrong_instance_released"] = dict(
        n=len(wi),
        oracle_in_top3=float(np.mean([r.orc in set(r.top.tolist()) for r in wi])),
        mean_rank_orc=float(np.mean([
            int(np.flatnonzero(np.argsort(-r.S["sc"], kind="stable") == r.orc)[0]) for r in wi])),
        rank_le3=float(np.mean([
            int(np.flatnonzero(np.argsort(-r.S["sc"], kind="stable") == r.orc)[0]) <= 2 for r in wi])),
        rank_le8=float(np.mean([
            int(np.flatnonzero(np.argsort(-r.S["sc"], kind="stable") == r.orc)[0]) <= 7 for r in wi])),
    )
    # same, but measured under the category (content) score
    out["wrong_instance_released"]["oracle_in_cat_top3"] = float(np.mean(
        [r.orc in set(np.argsort(-r.S["sc_content"], kind="stable")[:3].tolist()) for r in wi]))
    out["wrong_instance_released"]["oracle_in_cat_top8"] = float(np.mean(
        [r.orc in set(np.argsort(-r.S["sc_content"], kind="stable")[:8].tolist()) for r in wi]))

    with (args.output / f"headroom_{args.tag}.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    print(json.dumps({k: v for k, v in out.items() if k != "rules"}, indent=2))
    print(f"\n{'setdef':<24}{'fired':>7}{'oracle_in':>11}{'pick|in':>9}{'already_ok':>12}{'delta':>9}")
    for name, v in sorted(rows.items(), key=lambda kv: -kv[1]["delta"])[:16]:
        print(f"{name:<24}{v['fired']:>7d}{v['oracle_in_set_rate']:>11.3f}"
              f"{v['pick_given_in']:>9.3f}{v['released_already_correct_rate']:>12.3f}{v['delta']:>+9.4f}")
    print("  ...worst:")
    for name, v in sorted(rows.items(), key=lambda kv: kv[1]["delta"])[:4]:
        print(f"{name:<24}{v['fired']:>7d}{v['oracle_in_set_rate']:>11.3f}"
              f"{v['pick_given_in']:>9.3f}{v['released_already_correct_rate']:>12.3f}{v['delta']:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
