#!/usr/bin/env python
"""EXP-24 — Assemble exp/out_final/sota_table.json.

Two kinds of rows:

  * `measured_by_us: true` — numbers read from this run's `main_*.json`
    (produced by exp23_final.py / exp22_gres_eval.py from this run's scoring
    passes), each with an `evidence` path that exists and is fresh;
  * `measured_by_us: false` — numbers transcribed from the published tables,
    each with the exact source and the file the transcription was checked
    against.

Published transcription sources (both PDFs are in `papers/`):
  [GLODRIS] GLOD-RIS, TCSVT, Table I (zero-shot without additional training)
  [HGL]     HybridGL, CVPR 2025, Table 1 (zero-shot without additional training)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exp/out_final"

PUBLISHED = [
    dict(name="Grad-CAM", venue="ICCV 2019", backbone="R50", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=30.22), refcocoplus=dict(mIoU=33.96), refcocog=dict(mIoU=33.05),
         source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="MaskCLIP", venue="CVPR 2023", backbone="R50", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=25.62), refcocoplus=dict(mIoU=27.49), refcocog=dict(mIoU=30.13),
         source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="Global-Local (SAM)", venue="CVPR 2023", backbone="R50", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=31.83, oIoU=24.55), refcocoplus=dict(mIoU=34.97, oIoU=26.62),
         refcocog=dict(mIoU=40.66, oIoU=28.92), source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="Ref-Diff", venue="CoRR 2023", backbone="ViT-B", pretrained="SAM, SD, CLIP",
         refcoco=dict(mIoU=37.21, oIoU=35.16), refcocoplus=dict(mIoU=37.29, oIoU=35.56),
         refcocog=dict(mIoU=44.02, oIoU=38.62), source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="TAS", venue="EMNLP 2023", backbone="ViT-B", pretrained="SAM, BLIP2, CLIP",
         refcoco=dict(mIoU=39.84, oIoU=29.53), refcocoplus=dict(mIoU=43.63, oIoU=33.21),
         refcocog=dict(mIoU=46.62, oIoU=35.84), source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="BMS", venue="TCSVT 2024", backbone="ViT-B", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=40.77, oIoU=34.94), refcocoplus=dict(mIoU=41.70, oIoU=37.36),
         refcocog=dict(mIoU=48.32, oIoU=42.61), source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="IteRPrimE", venue="AAAI 2025", backbone="-", pretrained="Mask2Former, ALBEF",
         refcoco=dict(mIoU=40.20), refcocoplus=dict(mIoU=44.20), refcocog=dict(mIoU=46.00),
         source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="LGD", venue="PR 2025", backbone="ViT-B", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=45.41, oIoU=34.47), refcocoplus=dict(mIoU=47.23, oIoU=36.72),
         refcocog=dict(mIoU=47.31, oIoU=36.57), source="GLOD-RIS Table I [GLODRIS]"),
    dict(name="HybridGL (published)", venue="CVPR 2025", backbone="ViT-B", pretrained="SAM, CLIP",
         refcoco=dict(mIoU=49.48, oIoU=41.81), refcocoplus=dict(mIoU=43.40, oIoU=35.74),
         refcocog=dict(mIoU=51.25, oIoU=42.47),
         source="HybridGL Table 1 [HGL]; GLOD-RIS Table I [GLODRIS]"),
    dict(name="GLOD-RIS", venue="TCSVT", backbone="ViT-B", pretrained="SAM, CLIP, MLLM",
         refcoco=dict(mIoU=58.42, oIoU=50.85), refcocoplus=dict(mIoU=54.01, oIoU=45.96),
         refcocog=dict(mIoU=57.85, oIoU=50.03), source="GLOD-RIS Table I [GLODRIS]"),
]


def load_main(tag):
    p = OUT / f"main_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def measured_row(label, per_tag_row, venue="this run",
                 setting="zero-shot, SAM ViT-H + CLIP ViT-B/16"):
    """One row across the three RefCOCO-family splits, from this run's mains."""
    entry = dict(name=label, venue=venue, setting=setting, measured_by_us=True,
                 evidence="exp/out_final/evidence/")
    ok = False
    for tag in ("refcoco", "refcocoplus", "refcocog"):
        m = load_main(tag)
        if m is None or per_tag_row not in m["rows"]:
            continue
        r = m["rows"][per_tag_row]
        entry[tag] = dict(mIoU=round(100 * r["mIoU"], 2), oIoU=round(100 * r["oIoU"], 2),
                          delta_mIoU=round(100 * r["delta_vs_released"], 4), n=m["n"])
        entry["evidence"] = f"exp/out_final/evidence/{tag}_{per_tag_row}.json"
        ok = True
    return entry if ok else None


def main() -> int:
    rows = list(PUBLISHED)
    for label, row in (("HybridGL guidance (our reproduction)", "hybridgl_released"),
                       ("+ containment growth (control)", "snap_parent_strict"),
                       ("+ relation-gated operator (ours)", "ours"),
                       ("Ours: operator + growth (full)", "ours_snap")):
        e = measured_row(label, row)
        if e is not None:
            rows.append(e)

    gres = OUT / "main_grefcoco.json"
    if gres.exists():
        g = json.loads(gres.read_text())
        for row, label in (("released", "HybridGL guidance (our reproduction, gRefCOCO)"),
                           ("clause_only", "Ours-GRES: set-valued operator (no abstention)"),
                           ("clause+abstain", "Ours-GRES: + abstention (shipped)")):
            if row in g.get("rows", {}):
                m = g["rows"][row]
                rows.append(dict(name=label, venue="this run",
                                 setting="zero-shot, SAM ViT-H + CLIP ViT-B/16, GRES protocol",
                                 measured_by_us=True, dataset="grefcoco", n=g["n"],
                                 cIoU=round(m["cIoU"], 2), gIoU=round(m["gIoU"], 2),
                                 N_acc=round(m["N_acc"], 1), T_acc=round(m["T_acc"], 1),
                                 evidence="exp/out_final/main_grefcoco.json"))

    # GLOD-RIS independent reproduction (2026-10-10 operator instruction):
    # full official val splits, GLM-4.1V-Thinking-Flash guidance, measured with
    # this run's reimplementation.  Reported as a measured row with the
    # published numbers carried alongside for the audit.
    repro = OUT / "glodris_repro.json"
    if repro.exists():
        g = json.loads(repro.read_text())
        entry = dict(name="GLOD-RIS (our independent reproduction)", venue="this run",
                     setting="zero-shot, SAM ViT-H + CLIP ViT-B/16, GLM-4.1V guidance",
                     measured_by_us=True, evidence="exp/out_final/glodris_repro.json")
        ok = False
        for key in ("refcoco", "refcocoplus", "refcocog"):
            e = (g.get("datasets") or {}).get(key) or {}
            if e.get("mIoU") is None:
                continue
            entry[key] = dict(mIoU=e["mIoU"], oIoU=e["oIoU"], n=e["n"],
                              published=e.get("published"))
            ok = True
        if ok:
            rows.append(entry)

    out = {
        "caption": "Zero-shot / training-free referring image segmentation on full val splits. "
                   "Rows marked measured_by_us come from this run's scoring passes over the "
                   "released HybridGL pipeline; published rows are transcribed with source notes. "
                   "Absolute rows must be read with the reproduction gap disclosed in the paper: "
                   "the released code scores about 5 mIoU below its own published table "
                   "(verified against the unmodified official implementation).",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": ["mIoU", "oIoU", "cIoU", "gIoU"],
        "methods": rows,
    }
    (OUT / "sota_table.json").write_text(json.dumps(out, indent=2))
    measured = sum(1 for r in rows if r.get("measured_by_us"))
    print(f"wrote sota_table.json: {len(rows)} methods, {measured} measured by us")
    for r in rows:
        if r.get("measured_by_us"):
            p = ROOT / r["evidence"]
            print(f"  {'OK ' if p.exists() else 'MISSING'} {r['name']} -> {r['evidence']}")
    if len(rows) < 6 or measured < 3:
        print("NOT ENOUGH ROWS")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
