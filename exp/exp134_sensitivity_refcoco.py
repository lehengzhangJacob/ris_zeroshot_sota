#!/usr/bin/env python
"""EXP-134 — Parameter Sensitivity Analysis for the RefCOCO-family rule.

Sweeps the shipped dense-box-prior rule's key hyperparameters (box-fill weight
beta, gate eps, candidate count k_dec, semantic mix gem_mix) over the held-out
RefCOCO train tuning rows by re-running exp40_final_eval.py per setting (the
same deterministic re-analysis the shipped numbers come from) and merging the
curves into exp/out_final/sensitivity.json.

    python exp/exp134_sensitivity_refcoco.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUTF = ROOT / "exp/out_final"
RECORDS = ROOT / "exp/tuning/records_tune_refcoco.jsonl"
STUDY = OUTF / "optuna_study.json"

PLANS = [
    ("mode (box-fill measure: the operative box-fill knob under loc_only)",
     "mode", ["iou", "inside", "cover"]),
    ("k_dec (candidates kept)", "k_dec", [5, 10, 15, 20, 30]),
    ("gem_mix (semantic mix)", "gem_mix", [0.0, 0.25, 0.490569059596547, 0.75, 1.0]),
    ("beta (box-fill weight; inert under the shipped combine=loc_only)",
     "beta", [1.0, 2.20722374350054, 3.4]),
    ("eps (box-agreement gate; inert under the shipped combine=loc_only)",
     "eps", [0.005, 0.020794812033278407, 0.04]),
]


def run(params: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"best_params": params}, fh)
        pf = Path(fh.name)
    out = Path(tempfile.mkstemp(suffix=".json")[1])
    cmd = [PY, str(ROOT / "exp/exp40_final_eval.py"), "--dataset", "refcoco",
           "--records", str(RECORDS), "--split", "train", "--phrase-source", "sent",
           "--params", str(pf), "--output", str(out)]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    d = json.loads(out.read_text())
    pf.unlink(); out.unlink()
    return d


def main() -> int:
    base = json.loads(STUDY.read_text())["best_params"]
    anchor = run(base)
    print(f"[anchor] shipped: mIoU {anchor['rows']['ours']:.3f} "
          f"oIoU {anchor['oIoU']['ours']:.3f}", flush=True)

    sweeps = []
    for label, key, values in PLANS:
        settings = []
        for v in values:
            p = dict(base); p[key] = v
            d = run(p)
            settings.append(dict(
                value=v,
                mIoU=round(float(d["rows"]["ours"]), 4),
                oIoU=round(float(d["oIoU"]["ours"]), 4),
                is_shipped=bool(v == base[key] if isinstance(v, str)
                               else abs(float(v) - float(base[key])) < 1e-9),
            ))
            print(f"  {key}={v}: mIoU {settings[-1]['mIoU']:.2f} oIoU {settings[-1]['oIoU']:.2f}", flush=True)
        vals = {s_["mIoU"] for s_ in settings}
        entry = dict(parameter=label, key=key, shipped_value=base[key],
                     family="refcoco", settings=settings)
        if len(vals) == 1:
            entry["note"] = ("flat across the swept range on this population: "
                             "the parameter is not read by the shipped rule "
                             "configuration (combine=loc_only)")
        sweeps.append(entry)

    p = OUTF / "sensitivity.json"
    d = json.loads(p.read_text())
    d["sweeps"] = [s_ for s_ in d["sweeps"] if s_.get("family") != "refcoco"]
    d["sweeps"].extend(sweeps)
    d["refcoco_family"] = dict(
        population=str(RECORDS.relative_to(ROOT)),
        population_n=len([1 for _ in open(RECORDS)]),
        split="train (held-out refs)",
        frame="RefCOCO held-out train refs; mIoU and pooled oIoU per setting",
        tool="exp/exp134_sensitivity_refcoco.py (exp40 re-analysis per setting)",
        shipped_anchor=dict(mIoU=round(float(anchor["rows"]["ours"]), 4),
                            oIoU=round(float(anchor["oIoU"]["ours"]), 4)),
    )
    d["note"] += (" The RefCOCO-family curves come from the held-out train refs "
                  "of the same tuning split; both frames are descriptive and "
                  "select nothing.")
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"merged into {p}: {len(d['sweeps'])} sweeps total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
