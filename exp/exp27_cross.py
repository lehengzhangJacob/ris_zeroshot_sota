#!/usr/bin/env python
"""EXP-27 — Cross-dataset dose-response and residual strata, from this run's records.

Writes exp/out_final/cross_dataset.json: per dataset, the spatial-expression
frequency, how often the operator fires, the overall / spatial-stratum /
on-fired deltas, the growth control's delta, the wrong-instance share of the
oracle gap, and the residual-stratum numbers used in the paper (RefCOCO+ is
the pre-registered control: it is collected without location words).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp5_analyze_guided import Rec, reltype, taxonomy  # noqa: E402
from method import Selector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "exp/out_extreme"
OUT = ROOT / "exp/out_final"

SETS = [("refcoco", "refcoco"), ("refcocog", "refcocog"), ("refcocoplus", "refcocoplus")]


def main() -> int:
    own = Selector()
    out = {}
    for name, tag in SETS:
        rp = REC / f"records_{tag}.jsonl"
        pp = REC / f"pairs_{tag}.npz"
        if not rp.exists():
            print(f"skip {name} (no records)")
            continue
        recs = [Rec(json.loads(l)) for l in rp.open()]
        if pp.exists():
            pairs = np.load(pp)
            for r in recs:
                r.pc = pairs[str(r.r["ref_id"])].astype(np.float32)
        else:
            for r in recs:
                r.pc = None
        rt = np.array([reltype(r.rel) for r in recs])
        sp = rt != "none"
        ref = np.array([r.gt[r.final] for r in recs])
        idx_o = np.array([int(own(r)) for r in recs])
        ours = np.array([recs[i].gt[idx_o[i]] for i in range(len(recs))])
        snap = np.array([recs[i].gt[recs[i].ps[recs[i].final]] for i in range(len(recs))])
        full = np.array([recs[i].gt[recs[i].ps[idx_o[i]]] for i in range(len(recs))])
        fired = idx_o != np.array([r.final for r in recs])
        entry = dict(
            n=len(recs), spatial_frac=float(sp.mean()),
            fired_frac=float(fired.mean()), fired_n=int(fired.sum()),
            released_mIoU=float(ref.mean()), ours_mIoU=float(ours.mean()),
            full_mIoU=float(full.mean()),
            delta_overall=float((ours - ref).mean()),
            delta_full=float((full - ref).mean()),
            delta_spatial=float((ours - ref)[sp].mean()) if sp.sum() else None,
            delta_nonspatial=float((ours - ref)[~sp].mean()),
            delta_on_fired=float((ours - ref)[fired].mean()) if fired.sum() else None,
            snap_delta=float((snap - ref).mean()),
            snap_full_delta=float((full - ref).mean()))
        # residual strata among spatial expressions
        for key, mask in (("horizontal", rt == "horizontal"), ("vertical", rt == "vertical"),
                          ("middle", rt == "middle")):
            entry[f"delta_{key}"] = float((ours - ref)[mask].mean()) if mask.sum() else None
            entry[f"n_{key}"] = int(mask.sum())
        # spatial stratum restricted to expressions the operator's parse sees
        entry["n_spatial"] = int(sp.sum())
        if recs and recs[0].pc is not None:
            tax = taxonomy(recs, lambda r: r.final)
            entry["wrong_instance_gap_share"] = tax["by_class"]["wrong-instance"]["gap_share"]
            entry["under_seg_gap_share"] = tax["by_class"]["under-segmentation"]["gap_share"]
            entry["oracle_mIoU"] = tax["oracle_mIoU"]
        out[name] = entry
        print(f"{name}: n={entry['n']} spatial={entry['spatial_frac']*100:.1f}% "
              f"fired={entry['fired_frac']*100:.1f}% ours={entry['delta_overall']:+.4f} "
              f"full={entry['delta_full']:+.4f} snap={entry['snap_delta']:+.4f} "
              f"spatial-stratum={entry['delta_spatial']:+.4f}", flush=True)

    (OUT / "cross_dataset.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'cross_dataset.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
