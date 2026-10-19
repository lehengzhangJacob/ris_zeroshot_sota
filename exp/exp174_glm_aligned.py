#!/usr/bin/env python
"""EXP-174 — full-val GLM-aligned box-prior evaluation (gate directive 2026-10-18).

Runs exp40's frozen training-free rule with GLM-4.1V-Thinking-Flash sentence
boxes (exp171's full-val cache) on the three official val splits and assembles
glm_aligned.json against the same-rows LLaVA numbers from main_<ds>.json.

Prerequisites: exp171 has cached every unique (image, sentence) pair.
    python exp/exp174_glm_aligned.py
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
REC = {"refcoco": "records_refcoco.jsonl",
       "refcoco+": "records_refcocoplus.jsonl",
       "refcocog": "records_refcocog.jsonl"}
TAG = {"refcoco": "refcoco", "refcoco+": "refcocoplus", "refcocog": "refcocog"}


def main() -> int:
    res = {}
    procs = []
    for ds, rec in REC.items():
        out = OUTF / f"main_glm_{TAG[ds]}.json"
        cmd = [sys.executable, str(ROOT / "exp/exp40_final_eval.py"),
               "--dataset", ds, "--records", str(ROOT / "exp/out_extreme" / rec),
               "--split", "val", "--phrase-source", "sent", "--box-source", "glm",
               "--output", str(out), "--tag", f"glm_full_{TAG[ds]}",
               "--min-coverage", "95.0"]
        print("[run]", " ".join(cmd), flush=True)
        procs.append((ds, out, subprocess.Popen(cmd, cwd=ROOT)))
    for ds, out, proc in procs:
        if proc.wait() != 0:
            raise SystemExit(f"exp40 failed for {ds}")
        g = json.loads(out.read_text())
        llava = json.loads((OUTF / f"main_{TAG[ds]}.json").read_text())
        res[TAG[ds]] = dict(
            n=g["n"], box_coverage=g["box_coverage"], box_rate=g.get("box_rate"),
            mIoU=g["mIoU"], oIoU=g["oIoU"]["ours"], llava_mIoU=llava["rows"]["ours"],
            delta_mIoU=round(g["mIoU"] - llava["rows"]["ours"], 4),
            source=str(out.relative_to(ROOT)))
        print(ds, res[TAG[ds]], flush=True)
    doc = dict(
        model="glm-4.1v-thinking-flash",
        endpoint="Zhipu open.bigmodel.cn paas/v4 (component-matched GLOD-RIS MLLM)",
        prompt="byte-identical to the LLaVA box template; full sentence",
        rule="frozen training-free sentence-box rule (exp40 params), val evaluation only",
        sampling=dict(scope="full official val splits; every unique (image, sentence) pair cached",
                      cache="exp/out_glm_boxes (exp171 namespace, exp140-compatible)",
                      records={k: f"exp/out_extreme/{v}" for k, v in REC.items()}),
        datasets=res,
        comparison="same rows, same frozen rule; LLaVA-OneVision-7B boxes vs GLM-4.1V boxes",
        verdict=("GLM-4.1V-Thinking-Flash boxes evaluated on the full validation splits under the "
                 "same frozen rule. The aligned comparison isolates the box prior; differences are "
                 "reported as GLM minus LLaVA on identical records."),
        generated="exp/exp174_glm_aligned.py")
    (OUTF / "glm_aligned.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc["datasets"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
