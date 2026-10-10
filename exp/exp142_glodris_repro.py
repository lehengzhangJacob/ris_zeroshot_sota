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
VERSION = "v3"  # all-zero-grid fix + park-on-400 (2026-10-10)
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
                        "that never abort the run and are retried at the end. Pipeline "
                        "hardening (found by the full-val run itself): all-zero grids use a "
                        "uniform localization map; parked guidance routes the selector to a "
                        "DOD-only configuration (no category/localization terms); guidance "
                        "without a usable category drops only the category term"),
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
        d = REPRO / "outputs" / f"full_{tag}_{VERSION}"
        if not d.is_dir():
            doc["datasets"][ds] = dict(n=0, mIoU=None, oIoU=None,
                                       published=PUBLISHED[ds], status="missing")
            complete_all = False
            continue
        m = json.loads((d / "metrics.json").read_text())
        n_parked = 0
        n_records = 0
        n_zero_grid = 0
        n_missing_category = 0
        for f in (d / "records").glob("*.json"):
            n_records += 1
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            pred = r.get("prediction", {})
            if pred.get("guidance_source") == "GLM-parked-DOD-only":
                n_parked += 1
            glm = pred.get("glm") or {}
            if glm.get("localization_skipped") == "all-zero grid":
                n_zero_grid += 1
            if glm.get("category_missing"):
                n_missing_category += 1
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
            run_hash=m.get("run_hash"),
            glm_failures=dict(parked_expressions=n_parked,
                              records_written=n_records,
                              note=("parked = GLM guidance unavailable after repair retries; "
                                    "those expressions ran the DOD-only path and are flagged "
                                    "in their records (guidance_source=GLM-parked-DOD-only)")),
            fallbacks=dict(all_zero_grids=n_zero_grid,
                           category_missing=n_missing_category,
                           note=("all-zero grid = GLM answered no occupied cell; the pipeline "
                                 "uses a uniform (uninformative) localization map and flags "
                                 "localization_skipped in the record. category_missing = a "
                                 "guidance answer without a usable category; only the category "
                                 "term is dropped for that expression.")),
            output_dir=str(d))
        doc["datasets"][ds] = entry
        if not m["complete"]:
            complete_all = False
        print(ds, entry["n"], "/", entry["expected"], "mIoU", entry["mIoU"],
              "oIoU", entry["oIoU"], "parked", n_parked)
    doc["complete"] = complete_all
    # Diagnostic controls on RefCOCO val (no API cost, GPU-only): they localize
    # WHERE the reimplementation diverges -- the candidate pool is not the limit
    # (oracle mIoU ~82), so each component's effect on the selected mIoU is the
    # measurable object.  Diagnostic evidence only; the reported reproduction
    # number is the unflagged full pipeline.
    controls = {}
    for tag, label in (("dodonly", "no GLM guidance at all (DOD-only)"),
                       ("nocat", "GLM localization only (category term dropped)"),
                       ("noloc", "GLM category only (localization term dropped)"),
                       ("nodod", "no orthogonal decoupling (raw foreground features)")):
        mp = REPRO / "outputs" / f"ctrl_{tag}_v3" / "metrics.json"
        if mp.is_file():
            m = json.loads(mp.read_text())
            controls[tag] = dict(label=label, n=m["processed_expressions"],
                                 complete=bool(m["complete"]),
                                 mIoU=round(m["mIoU"], 4), oIoU=round(m["oIoU"], 4))
    if controls:
        doc["refcoco_component_controls"] = controls
    (OUTF / "glodris_repro.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print("wrote", OUTF / "glodris_repro.json", "| complete:", complete_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
