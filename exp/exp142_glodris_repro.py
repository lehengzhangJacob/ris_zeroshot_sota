#!/usr/bin/env python
"""EXP-142 — assemble exp/out_final/glodris_repro.json from the full-val
GLOD-RIS reproduction run (2026-10-10 operator instruction, second attempt).

Reads the repro's outputs/full_<tag>_v2/metrics.json + records/ and the GLM
cache to report per-dataset n, mIoU, oIoU, published values, deltas, GLM
failure statistics (parked expressions) and explicit coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

REPRO = Path("/home/msj_team/Jacob/sys/glod_ris_repro")
OUTF = Path("exp/out_final")
TAGS = {"refcoco": "refcoco", "refcocoplus": "refcocop", "refcocog": "refcocog"}
PUBLISHED = {"refcoco": dict(mIoU=58.42, oIoU=50.85),
             "refcocoplus": dict(mIoU=54.01, oIoU=45.96),
             "refcocog": dict(mIoU=57.85, oIoU=50.03)}


def main() -> int:
    doc = dict(
        what=("full official-val reproduction of GLOD-RIS with the now-working GLM "
              "access (2026-10-10 operator instruction, second attempt)"),
        implementation=("independent reimplementation at /home/msj_team/Jacob/sys/glod_ris_repro "
                        "(author repo unavailable); production-grade GLM layer added: tolerant "
                        "grid parsing (quoted digits, floats, category->cells dicts, nested "
                        "rows, 1-based cell-index lists, all-zero grids accepted), a repair "
                        "prompt stating the parse error, 8192-token escalation on "
                        "finish_reason=length, 429/5xx retried with backoff, parked failures "
                        "that never abort the run and are retried at the end"),
        cache=("glod/../cache: existing vision cache reused where valid (keyed by image sha + "
               "settings, unaffected by the code fix); GLM responses cached per "
               "(image, expression, model, grid); 704MB vision + GLM cache carried over from "
               "the aborted first attempt (which stopped at 223/10,834); the aborted run's "
               "per-expression records were NOT reused because the implementation hash "
               "changed (new output dirs full_<tag>_v2)"),
        protocol=("official val splits; mIoU = mean per-expression IoU; oIoU = cumulative "
                  "intersection/union -- the protocol the published numbers use"),
        datasets={})
    complete_all = True
    for ds, tag in TAGS.items():
        d = REPRO / "outputs" / f"full_{tag}_v2"
        if not d.is_dir():
            doc["datasets"][ds] = dict(n=0, mIoU=None, oIoU=None,
                                       published=PUBLISHED[ds], status="missing")
            complete_all = False
            continue
        m = json.loads((d / "metrics.json").read_text())
        n_parked = 0
        n_records = 0
        for f in (d / "records").glob("*.json"):
            n_records += 1
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            if r.get("prediction", {}).get("guidance_source") == "GLM-parked-DOD-only":
                n_parked += 1
        entry = dict(
            n=m["processed_expressions"],
            expected=m["expected_expressions"],
            mIoU=round(m["mIoU"], 4), oIoU=round(m["oIoU"], 4),
            oracle_mIoU=round(m["oracle_mIoU"], 4),
            published=PUBLISHED[ds],
            delta=dict(mIoU=round(m["mIoU"] - PUBLISHED[ds]["mIoU"], 4),
                       oIoU=round(m["oIoU"] - PUBLISHED[ds]["oIoU"], 4)),
            coverage=round(m["processed_expressions"] / max(1, m["expected_expressions"]), 6),
            complete=bool(m["complete"]),
            glm_failures=dict(parked_expressions=n_parked,
                              records_written=n_records,
                              note=("parked = GLM guidance unavailable after repair retries; "
                                    "those expressions ran the DOD-only path and are flagged "
                                    "in their records (guidance_source=GLM-parked-DOD-only)")),
            output_dir=str(d))
        doc["datasets"][ds] = entry
        if not m["complete"]:
            complete_all = False
        print(ds, entry["n"], "/", entry["expected"], "mIoU", entry["mIoU"],
              "oIoU", entry["oIoU"], "parked", n_parked)
    doc["complete"] = complete_all
    (OUTF / "glodris_repro.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print("wrote", OUTF / "glodris_repro.json", "| complete:", complete_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
