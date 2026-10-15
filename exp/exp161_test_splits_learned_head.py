#!/usr/bin/env python
"""EXP-161 — evaluate the promoted learned head on the five test columns.

Re-runs the exp159 arm battery (reference vs learned_head + attributions) on
the cached test-split records under the aligned protocol, so the compared
table's testA/testB/test(u) columns stay consistent with the promoted shipped
head.  Deterministic re-analysis over cached records; heads are fitted on the
RefCOCO-family held-out TRAIN tune refs only (per family, reused across its
test splits).

    python exp/exp161_test_splits_learned_head.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp159_head_rebuild as e159  # noqa: E402

TC = Path("exp/out_test_columns")
OUTF = Path("exp/out_final")

SPLITS = [
    ("refcoco_testA", "refcoco", "testA"),
    ("refcoco_testB", "refcoco", "testB"),
    ("refcocoplus_testA", "refcocoplus", "testA"),
    ("refcocoplus_testB", "refcocoplus", "testB"),
    ("refcocog_testu", "refcocog", "test"),
]


def main() -> int:
    te = e159.TextEncoder()
    models_by_family = {}
    results = {}
    for tag, family, split in SPLITS:
        records = TC / f"records_tc_{tag}.jsonl"
        if not records.is_file():
            print(f"missing {records}; skipping {tag}")
            continue
        if family not in models_by_family:
            print(f"[{family}] fitting heads on tune refs", flush=True)
            models_by_family[family] = e159.fit_learned_head(family, te)
        out = TC / f"learned_{tag}.json"
        rc = e159.evaluate(family, split=split, records_path=records, out_path=out,
                           te=te, models=models_by_family[family])
        d = json.loads(out.read_text())
        results[tag] = dict(dataset=family, split=split, n=d["n"],
                            arms=d["arms"],
                            box_queried_pct=d["box_queried_pct"],
                            box_none_records=d["box_none_records"])
    agg = dict(
        what=("promoted learned head on the five compared-table test columns; "
              "same cached records, aligned protocol (mIoU = mean per-expression "
              "IoU; oIoU = pooled intersection/union)"),
        splits=results,
        generated="exp161_test_splits_learned_head.py",
    )
    OUTF.mkdir(parents=True, exist_ok=True)
    (OUTF / "test_columns_learned.json").write_text(json.dumps(agg, indent=2))
    ref = json.loads((OUTF / "head_rebuild_parts/refcoco.json").read_text())
    print("wrote exp/out_final/test_columns_learned.json")
    return 0 if len(results) == len(SPLITS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
