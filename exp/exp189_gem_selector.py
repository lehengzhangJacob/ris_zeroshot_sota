#!/usr/bin/env python
"""EXP-189 -- offline selector probe: GEM coherence alone vs the released semantic score.

Adopts the idea from Bousselham et al. (GEM, arXiv:2312.00878) that self-self
attention coherence is a training-free grounding signal.  We hold the pool and
candidate set fixed (the stored full-val pass) and change only the selection
rule: argmax GEM coherence, against argmax released semantic score and the
released final choice.  No re-encoding, no tuning on val.

    python exp/exp189_gem_selector.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/out_final/gem_selector.json"
RECORDS = {
    "refcoco": ROOT / "exp/out_extreme/records_refcoco.jsonl",
    "refcocoplus": ROOT / "exp/out_extreme/records_refcocoplus.jsonl",
    "refcocog": ROOT / "exp/out_extreme/records_refcocog.jsonl",
}


def score(records, key):
    """Return mIoU, oIoU, oracle, wrong-instance share for argmax of `key`."""
    ious, inters, unions, oracles, gaps, wrong = [], 0.0, 0.0, [], 0.0, 0.0
    for r in records:
        vals = r[key]
        if isinstance(vals, (int, np.integer)):  # stored final pick, not a score list
            pick = int(vals)
        elif vals:
            pick = int(np.argmax(vals))
        else:
            continue
        gt = r["gt_iou"]
        if pick < 0 or pick >= len(gt):
            continue
        a = float(gt[pick])
        o = float(max(gt))
        ious.append(a)
        oracles.append(o)
        inters += float(r["inter_gt"][pick])
        unions += float(r["union_gt"][pick])
        g = max(0.0, o - a)
        gaps += g
        if a < 0.5 <= o:
            wrong += g
    return dict(n=len(ious), mIoU=100.0 * float(np.mean(ious)),
                oIoU=100.0 * inters / unions if unions else 0.0,
                oracle_mIoU=100.0 * float(np.mean(oracles)),
                wrong_instance_share=wrong / gaps if gaps else 0.0)


def main():
    out = dict(
        idea="GEM self-self attention coherence as a training-free selection signal",
        source="Bousselham et al., Grounding Everything, arXiv:2312.00878",
        entry="exp/exp189_gem_selector.py",
        protocol=("full official val; stored single scoring pass per dataset; the pool and "
                  "candidate set are fixed, only the selection rule changes; argmax over "
                  "the stored per-proposal GEM coherence ('score_gem') vs the released "
                  "semantic score ('sc') and the released final pick ('final'); oIoU is "
                  "pooled from the same records"),
        datasets={},
    )
    for name, path in RECORDS.items():
        recs = [json.loads(line) for line in open(path)]
        out["datasets"][name] = {
            "released_final": score(recs, "final"),
            "argmax_semantic": score(recs, "sc"),
            "argmax_gem": score(recs, "score_gem"),
        }
        print(name, json.dumps(out["datasets"][name], indent=1), flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
