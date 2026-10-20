#!/usr/bin/env python
"""EXP-185 — assemble the multi-pipeline diagnosis (operator directive 2026-10-20).

Combines the full-split oracle-gap decomposition of the released HybridGL
pipeline and the GLOD-RIS reimplementation with the new full runs
(Global-Local released selector; MaskCLIP-based dense value map) and the
documented blockers.  Writes exp/out_final/multipipeline_diagnosis.json in the
schema required by extreme_bar.py and paper/tables/multipipeline.tex.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"
DATASETS = ("refcoco", "refcocoplus", "refcocog")
LABEL = {"refcoco": "RefCOCO", "refcocoplus": "RefCOCO+", "refcocog": "RefCOCOg"}


def load_ev():
    return json.loads((OUTF / "multipipeline_evidence.json").read_text())


def full_entry(name, repo, dss, evidence, **extra):
    out = dict(name=name, repo=repo, datasets=dss, evidence=evidence)
    out.update(extra)
    return out


def main() -> int:
    ev = load_ev()
    hyb = ev["pipelines"]["hybridgl_released"]
    glod = ev["pipelines"]["glodris_reimplementation"]

    def from_decomp(m):
        return {ds: dict(n=int(m[ds]["n"]), oracle=round(100 * m[ds]["oracle"], 4),
                         achieved=round(100 * m[ds]["achieved"], 4),
                         wrong_instance_share=round(m[ds]["wrong_instance_share"], 4))
                for ds in DATASETS}

    pipes = [
        full_entry("HybridGL released pipeline (this run)",
                   "https://github.com/fhgyuanshen/HybridGL",
                   from_decomp(hyb), "exp/out_final/multipipeline_evidence.json",
                   pool="SAM ViT-H pps 8"),
        full_entry("GLOD-RIS reimplementation (our reading; no released code)",
                   "none: code unreleased; local reimplementation "
                   "(this run) at /home/msj_team/Jacob/sys/glod_ris_repro",
                   from_decomp(glod), "exp/out_final/multipipeline_evidence.json",
                   pool="SAM ViT-H pps 8"),
    ]

    gl_files = [OUTF / f"globallocal_sam_{ds}.json" for ds in DATASETS]
    if all(p.is_file() for p in gl_files):
        dss = {}
        raw = {}
        for ds, p in zip(DATASETS, gl_files):
            d = json.loads(p.read_text())
            dss[ds] = dict(n=int(d["n"]), oracle=round(d["oracle_mIoU"], 4),
                           achieved=round(d["mIoU"], 4),
                           wrong_instance_share=round(d["wrong_instance_share"], 4))
            raw[ds] = d
        (OUTF / "globallocal_sam_all.json").write_text(json.dumps(raw, indent=2))
        pipes.append(full_entry(
            "Global-Local (Zero-shot-RIS) released selector, SAM ViT-H pool",
            "https://github.com/Seonghoon-Yu/Zero-shot-RIS", dss,
            "exp/out_final/globallocal_sam_all.json",
            pool="SAM ViT-H pps 8 (the released FreeSOLO R101 pool is blocked; "
                 "the selector is the release's CLIPViTFM token-masking head)"))

    mc_files = [OUTF / f"maskclip_sam_{ds}.json" for ds in DATASETS]
    if all(p.is_file() for p in mc_files):
        dss = {}
        raw = {}
        for ds, p in zip(DATASETS, mc_files):
            d = json.loads(p.read_text())
            dss[ds] = dict(n=int(d["n"]), oracle=round(d["oracle_mIoU"], 4),
                           achieved=round(d["mIoU"], 4),
                           wrong_instance_share=round(d["wrong_instance_share"], 4))
            raw[ds] = d
        (OUTF / "maskclip_sam_all.json").write_text(json.dumps(raw, indent=2))
        pipes.append(full_entry(
            "MaskCLIP-based dense value map, own component pool",
            "https://github.com/chongzhou96/MaskCLIP", dss,
            "exp/out_final/maskclip_sam_all.json",
            pool="components of its own value map at fixed 50/70/90th-percentile "
                 "thresholds, top-12 by mean value",
            adaptation="MaskCLIP is a dense open-vocabulary segmenter without an instance "
                       "pool; the component pool and the mean-value selection are this "
                       "harness's fixed adaptation, not a released RIS procedure"))

    blocked = []
    for f in ("blocked_tas.json", "blocked_refdiff.json",
              "blocked_globallocal_freesolo.json"):
        p = OUTF / f
        if p.is_file():
            b = json.loads(p.read_text())
            blocked.append(dict(name=b["name"], repo=b["repo"], reason=b["reason"],
                                evidence=str(p.relative_to(ROOT))))

    diag = dict(
        what="oracle-gap decomposition across released zero-shot pipelines, full official val splits",
        definition=("gap = oracle (best pool mask vs GT) - achieved (the pipeline's selected mask "
                    "vs GT); wrong-instance when achieved < 0.5 <= oracle; wrong_instance_share is "
                    "the fraction of the recoverable IoU mass in wrong-instance gaps. Every "
                    "reported dataset uses the full official split n."),
        pipelines=pipes,
        blocked=blocked,
        evidence_index="exp/out_final/multipipeline_evidence.json (HybridGL, GLOD-RIS); "
                       "globallocal_sam_*.json; maskclip_sam_*.json; blocked_*.json",
        verdict=("Across the released HybridGL pipeline, the GLOD-RIS reimplementation and the "
                 "Global-Local selector, wrong-instance selection carries the large majority of "
                 "the recoverable oracle gap on every dataset (HybridGL 93.9/94.1/91.8%, "
                 "GLOD-RIS 93.3/93.7/91.8%, Global-Local values in the table). The MaskCLIP "
                 "value-map adaptation is the contrast: its own component pool already caps the "
                 "oracle at ~25 mIoU, so its gap is pool-limited, while the same dense map on the "
                 "shared SAM pool still loses most of its recoverable mass to wrong-instance "
                 "selection. The diagnosis therefore replicates where a good pool exists and "
                 "identifies the pool as the limit where it does not. The exact Global-Local "
                 "FreeSOLO pool, TAS and Ref-Diff are documented as blocked with their reasons."),
        generated="exp/exp185_multipipeline.py")
    (OUTF / "multipipeline_diagnosis.json").write_text(json.dumps(diag, indent=2))

    # ---- paper table ------------------------------------------------------
    rows = []
    short = {"HybridGL": "HybridGL (released)", "GLOD": "GLOD-RIS (reimpl.)",
             "Global-Local": "Global-Local (SAM)", "MaskCLIP": "MaskCLIP (own pool)"}
    for p in pipes:
        cells = [next((v for k, v in short.items() if p["name"].startswith(k)), p["name"])]
        for ds in DATASETS:
            m = p["datasets"][ds]
            cells.append(f"{m['achieved']:.2f} / {m['oracle']:.2f}")
        cells.append("/".join(f"{100*p['datasets'][ds]['wrong_instance_share']:.1f}"
                              for ds in DATASETS))
        rows.append(" & ".join(cells) + r" \\")
    lines = [
        r"\begin{tabular}{lccc c}", r"\toprule",
        r"Pipeline & RefCOCO & RefCOCO+ & RefCOCOg & wrong-instance share \\",
        r" & achieved / oracle & achieved / oracle & achieved / oracle & (R / R+ / Rg) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
    ]
    (ROOT / "paper/tables/multipipeline.tex").write_text("\n".join(lines) + "\n")

    print(json.dumps({p["name"]: p["datasets"] for p in pipes}, indent=1))
    print("blocked:", [b["name"] for b in blocked])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
