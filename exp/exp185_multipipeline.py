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

    def from_decomp(m, oiou):
        return {ds: dict(n=int(m[ds]["n"]), oracle=round(100 * m[ds]["oracle"], 4),
                         achieved=round(100 * m[ds]["achieved"], 4),
                         oIoU=round(float(oiou[ds]), 4),
                         wrong_instance_share=round(m[ds]["wrong_instance_share"], 4))
                for ds in DATASETS}

    main_files = {"refcoco": "main_refcoco.json", "refcocoplus": "main_refcocoplus.json",
                  "refcocog": "main_refcocog.json"}
    hyb_o = {ds: json.loads((OUTF / f).read_text())["oIoU"]["released"]
             for ds, f in main_files.items()}
    glod_doc = json.loads((OUTF / "glodris_repro.json").read_text())["datasets"]
    glod_o = {ds: glod_doc[ds]["oIoU"] for ds in DATASETS}

    pipes = [
        full_entry("HybridGL released pipeline (this run)",
                   "https://github.com/fhgyuanshen/HybridGL",
                   from_decomp(hyb, hyb_o), "exp/out_final/multipipeline_evidence.json",
                   pool="SAM ViT-H pps 8"),
        full_entry("GLOD-RIS reimplementation (our reading; no released code)",
                   "none: code unreleased; local reimplementation "
                   "(this run) at /home/msj_team/Jacob/sys/glod_ris_repro",
                   from_decomp(glod, glod_o), "exp/out_final/multipipeline_evidence.json",
                   pool="SAM ViT-H pps 8"),
    ]

    gl_files = [OUTF / f"globallocal_sam_{ds}.json" for ds in DATASETS]
    if all(p.is_file() for p in gl_files):
        dss = {}
        raw = {}
        for ds, p in zip(DATASETS, gl_files):
            d = json.loads(p.read_text())
            dss[ds] = dict(n=int(d["n"]), oracle=round(d["oracle_mIoU"], 4),
                           achieved=round(d["mIoU"], 4), oIoU=round(d["oIoU"], 4),
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
                           achieved=round(d["mIoU"], 4), oIoU=round(d["oIoU"], 4),
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

    if all(p.is_file() for p in mc_files):
        samepool = {}
        for ds, p in zip(DATASETS, mc_files):
            d = json.loads(p.read_text())
            if "same_pool_control" in d:
                samepool[ds] = d["same_pool_control"]
        if samepool:
            (OUTF / "maskclip_sam_samepool_all.json").write_text(json.dumps(samepool, indent=2))

    ss_files = [OUTF / f"spluss_sam_{ds}.json" for ds in DATASETS]
    if all(p.is_file() for p in ss_files):
        dss = {}
        raw = {}
        for ds, p in zip(DATASETS, ss_files):
            d = json.loads(p.read_text())
            dss[ds] = dict(n=int(d["n"]), oracle=round(d["oracle_mIoU"], 4),
                           achieved=round(d["mIoU"], 4), oIoU=round(d["oIoU"], 4),
                           wrong_instance_share=round(d["wrong_instance_share"], 4))
            raw[ds] = d
        (OUTF / "spluss_sam_all.json").write_text(json.dumps(raw, indent=2))
        pipes.append(full_entry(
            "S+S (Segment, Select, Correct) zero-shot select stage, SAM ViT-H pool",
            "https://github.com/fgirbal/segment-select-correct", dss,
            "exp/out_final/spluss_sam_all.json",
            pool="SAM ViT-H pps 8 shared pool (the released stage-1 pool is "
                 "GroundingDINO+SAM; the released selection rule is run here)",
            protocol="official val splits; per-expression mIoU and pooled oIoU under our "
                     "unified evaluator; released reverse_blur CLIP selection; published "
                     "S+S zero-shot numbers are oIoU 34.39/36.29/39.29 and mIoU "
                     "40.96/41.77/45.36 (RefCOCO/+/g val), quoted as published",
            published={ds: {"mIoU": {"refcoco": 40.96, "refcocoplus": 41.77, "refcocog": 45.36}[ds],
                            "oIoU": {"refcoco": 34.39, "refcocoplus": 36.29, "refcocog": 39.29}[ds],
                            "source": "S+S ECCV 2024 Workshops zero-shot row, quoted as published; "
                                      "the published pool is GroundingDINO+SAM, so this is "
                                      "cross-harness to our SAM-pool run"}
                       for ds in DATASETS}))

    sr_files = [OUTF / f"spluss_rectangle_sam_{ds}.json" for ds in DATASETS]
    if all(p.is_file() for p in sr_files):
        dss = {}
        raw = {}
        for ds, p in zip(DATASETS, sr_files):
            d = json.loads(p.read_text())
            dss[ds] = dict(n=int(d["n"]), oracle=round(d["oracle_mIoU"], 4),
                           achieved=round(d["mIoU"], 4), oIoU=round(d["oIoU"], 4),
                           wrong_instance_share=round(d["wrong_instance_share"], 4))
            raw[ds] = d
        (OUTF / "spluss_rectangle_sam_all.json").write_text(json.dumps(raw, indent=2))
        pipes.append(full_entry(
            "S+S released selection variant (rectangle prompt), SAM ViT-H pool",
            "https://github.com/fgirbal/segment-select-correct", dss,
            "exp/out_final/spluss_rectangle_sam_all.json",
            pool="SAM ViT-H pps 8 shared pool (released stage-1 pool is GroundingDINO+SAM)",
            protocol="official val splits; released vp_and_max_sim_clip_choice with the rectangle visual "
                     "prompt option, CLIP ViT-B/32, on the shared SAM pool; published S+S zero-shot "
                     "numbers quoted as published (reverse_blur, GroundingDINO+SAM pool), cross-harness",
            published={ds: {"mIoU": {"refcoco": 40.96, "refcocoplus": 41.77, "refcocog": 45.36}[ds],
                            "oIoU": {"refcoco": 34.39, "refcocoplus": 36.29, "refcocog": 39.29}[ds],
                            "comparable": False,
                            "source": "S+S ECCV 2024 Workshops zero-shot row, quoted as published; "
                                      "not directly comparable: this row uses a different visual prompt "
                                      "(rectangle) and pool, so the published reverse_blur numbers are "
                                      "shown for context only"}
                       for ds in DATASETS}))

    rc_path = OUTF / "refchess_records_refcoco.jsonl"
    if rc_path.is_file():
        rows = [json.loads(l) for l in open(rc_path)]
        if len(rows) >= 10834:
            n = len(rows)
            achieved = 100.0 * sum(r["iou"] for r in rows) / n
            oracle = 100.0 * sum(r["oracle_iou"] for r in rows) / n
            I_tot = sum(r["inter"] for r in rows); U_tot = sum(r["union"] for r in rows)
            gap = sum(max(0.0, r["oracle_iou"] - r["iou"]) for r in rows)
            wrong = sum(max(0.0, r["oracle_iou"] - r["iou"]) for r in rows
                        if r["iou"] < 0.5 <= r["oracle_iou"])
            pipes.append(full_entry(
                "RefChess MCTS selection (released code; CLIP ViT-B/32 backbone, GEM heatmap, 16 simulations)",
                "https://github.com/Tongshiyan/RefChess",
                {"refcoco": dict(n=int(n), oracle=round(oracle, 4), achieved=round(achieved, 4),
                                 oIoU=round(100.0 * I_tot / U_tot if U_tot else 0.0, 4),
                                 wrong_instance_share=round(wrong / gap if gap else 0.0, 4))},
                "exp/out_final/refchess_records_refcoco.jsonl",
                pool="SAM ViT-H pps 8 (min region 1600, containment filtered) built by the released code",
                protocol=("official RefCOCO val (unc), n=10,834, per-expression IoU and pooled oIoU under our "
                          "evaluator; released RefChess MCTS code with two resource substitutions: FG-CLIP "
                          "backbone replaced by OpenAI CLIP ViT-B/32 and the hybrid heatmap by GEM-only; "
                          "16 MCTS simulations instead of the paper's 2048. Published RefChess val numbers "
                          "are mIoU 55.19 / oIoU 48.47 (FG-CLIP, hybrid heatmap, 2048 sims), quoted as "
                          "published and cross-harness to this reduced configuration."),
                published={"refcoco": {"mIoU": 55.19, "oIoU": 48.47,
                                       "source": "RefChess ICML 2026 val row, quoted as published; "
                                                 "cross-harness to this reduced CLIP/GEM/16-sim run"}}))

    blocked = []
    for f in ("blocked_tas.json", "blocked_refdiff.json",
              "blocked_globallocal_freesolo.json"):
        p = OUTF / f
        if p.is_file():
            b = json.loads(p.read_text())
            blocked.append(dict(name=b["name"], repo=b["repo"], reason=b["reason"],
                                evidence=str(p.relative_to(ROOT))))

    PUBLISHED = {
        "HybridGL": {ds: {"mIoU": {"refcoco": 49.48, "refcocoplus": 43.40, "refcocog": 51.25}[ds],
                          "oIoU": {"refcoco": 41.81, "refcocoplus": 35.74, "refcocog": 42.47}[ds],
                          "source": "HybridGL CVPR 2025 val columns, quoted as published"}
                    for ds in DATASETS},
        "GLOD": {ds: {"mIoU": {"refcoco": 58.42, "refcocoplus": 54.01, "refcocog": 57.85}[ds],
                      "oIoU": {"refcoco": 50.85, "refcocoplus": 45.96, "refcocog": 50.03}[ds],
                      "source": "GLOD-RIS TCSVT 2026 Table I val columns, quoted as published"}
                 for ds in DATASETS},
        "Global-Local": {ds: {"mIoU": {"refcoco": 31.83, "refcocoplus": 34.97, "refcocog": 40.66}[ds],
                              "oIoU": {"refcoco": 24.55, "refcocoplus": 26.62, "refcocog": 28.92}[ds],
                              "source": "Global-Local (SAM) published val row, quoted as published; "
                                        "the compared paper's original pool is FreeSOLO R101, so this "
                                        "published row is cross-harness to our SAM-pool run"}
                         for ds in DATASETS},
        "MaskCLIP": {ds: {"mIoU": {"refcoco": 25.62, "refcocoplus": 27.49, "refcocog": 30.13}[ds],
                          "oIoU": {"refcoco": 20.18, "refcocoplus": 22.06, "refcocog": 23.05}[ds],
                          "comparable": False,
                          "source": "MaskCLIP-based published RIS row, quoted as published; this run "
                                    "uses a harness adaptation (dense value map + component pool), so "
                                    "the published numbers are not directly comparable"}
                     for ds in DATASETS},
    }
    PROTOCOL = {
        "HybridGL": ("official val splits (RefCOCO/RefCOCO+ UNC; RefCOCOg UMD Val(U)); "
                     "per-expression mIoU against the GT mask and pooled oIoU; our unified "
                     "evaluator; SAM ViT-H pps 8 pool; released HybridGL code on our harness"),
        "GLOD": ("official val splits (same); our unified evaluator; SAM ViT-H pps 8 pool; "
                 "our end-to-end reimplementation of the described method; code unreleased"),
        "Global-Local": ("official val splits (same); our unified evaluator; released CLIPViTFM "
                         "selector (released CLIP ViT-B/32) on the shared SAM ViT-H pool; the "
                         "original FreeSOLO R101 pool is unobtainable (documented blocker)"),
        "MaskCLIP": ("official val splits (same); our unified evaluator; released MaskCLIP dense "
                     "value map (CLIP ViT-B/16) with a fixed own-component pool (50/70/90th "
                     "percentile thresholds, top-12 by mean value); this is a harness adaptation, "
                     "not a released RIS procedure"),
    }
    for p in pipes:
        for tag in ("HybridGL", "GLOD", "Global-Local", "MaskCLIP"):
            if p["name"].startswith(tag):
                p["protocol"] = PROTOCOL[tag]
                p["published"] = PUBLISHED[tag]
                break

    diag = dict(
        what="oracle-gap decomposition across released, reimplemented and adapted zero-shot pipelines, full official val splits",
        definition=("gap = oracle (best pool mask vs GT) - achieved (the pipeline's selected mask "
                    "vs GT); wrong-instance when achieved < 0.5 <= oracle; wrong_instance_share is "
                    "the fraction of the recoverable IoU mass in wrong-instance gaps. Every "
                    "reported dataset uses the full official split n. All achieved/oracle values "
                    "in this file are single-protocol (our unified official evaluator); published "
                    "numbers are reported separately per pipeline in `published` and are "
                    "cross-harness as labelled there."),
        pipelines=pipes,
        blocked=blocked,
        evidence_index="exp/out_final/multipipeline_evidence.json (HybridGL, GLOD-RIS); "
                       "globallocal_sam_*.json; maskclip_sam_*.json; blocked_*.json",
        verdict=("Across the released HybridGL pipeline, the GLOD-RIS reimplementation, the "
                 "Global-Local selector and the S+S selection, wrong-instance selection carries the "
                 "large majority of the recoverable oracle gap on every dataset (HybridGL 94.0/94.2/91.8%, "
                 "GLOD-RIS 93.3/93.7/91.8%, Global-Local 95.1/94.2/92.5%, S+S 95.4/94.7/93.2%). The MaskCLIP "
                 "value-map adaptation is the contrast: its own component pool caps the oracle "
                 "at 25.2/28.1/31.8 mIoU with wrong-instance shares 24.3/27.3/31.6%, so its gap "
                 "is pool-limited; the same dense map on the shared SAM pool selects "
                 "8.2/9.8/11.6 mIoU and is wrong-instance dominated (96.7/96.3/95.2%). The "
                 "diagnosis therefore replicates where a good pool exists and identifies the "
                 "pool as the limit where it does not. The S+S rectangle-prompt variant preserves "
                 "the same composition (96.7/96.5/95.3%). The exact Global-Local FreeSOLO pool, "
                 "TAS and Ref-Diff are documented as blocked with their reasons."),
        falsification=[
            dict(
                claim=("every score-based selection sits far below its own pool oracle and the gap is "
                       "dominated by wrong-instance selection"),
                test=("MaskCLIP's own dense-map component pool: oracle 25.20/28.08/31.79 mIoU vs "
                      "achieved 8.21/10.55/14.30; wrong-instance share 24.3/27.3/31.6% of the gap mass"),
                verdict=("REFUTED as a universal claim: for this pipeline the pool, not wrong-instance "
                         "selection, is the limit (most of the gap is in records where even the oracle "
                         "is below IoU 0.5). The diagnosis is conditional on the pool containing the "
                         "referent, as the same-pool control shows."),
                evidence="exp/out_final/maskclip_sam_all.json"),
            dict(
                claim=("a dense/compositional encoder that scores whole regions with global-local "
                       "features repairs selection"),
                test=("released Global-Local CLIPViTFM selector on the shared SAM ViT-H pool, full "
                      "splits: achieved 32.95/36.31/41.62 vs oracle 82.21/82.40/79.12; wrong-instance "
                      "share 95.1/94.2/92.5% of the gap mass"),
                verdict=("NOT SUPPORTED: the global-local scorer remains wrong-instance dominated on "
                         "every dataset; no repair of the selection failure is observed."),
                evidence="exp/out_final/globallocal_sam_all.json"),
            dict(
                claim=("the diagnosis holds when the pool contains the referent, even for a selector "
                       "that is pool-limited on its own masks"),
                test=("the same MaskCLIP dense value map scoring the shared SAM ViT-H proposals: "
                      "achieved 8.15/9.83/11.57 vs oracle 82.21/82.40/79.12; wrong-instance share "
                      "96.7/96.3/95.2%"),
                verdict=("SUPPORTED: on a good pool the same dense selector's recoverable mass is "
                         "wrong-instance dominated; its own-pool result bounds the claim rather than "
                         "overturning it."),
                evidence="exp/out_final/maskclip_sam_samepool_all.json"),
        ],
        generated="exp/exp185_multipipeline.py")
    (OUTF / "multipipeline_diagnosis.json").write_text(json.dumps(diag, indent=2))

    # ---- paper table ------------------------------------------------------
    rows = []
    short = {"HybridGL": "HybridGL (released)", "GLOD": "GLOD-RIS (reimpl.)",
             "Global-Local": "Global-Local (SAM)", "MaskCLIP": "MaskCLIP (own pool)",
             "S+S": "S+S (SAM)", "RefChess": "RefChess (CLIP/GEM)"}
    for p in pipes:
        if "rectangle prompt" in p["name"]:
            cells = ["S+S (rectangle)"]
        else:
            cells = [next((v for k, v in short.items() if p["name"].startswith(k)), p["name"])]
        for ds in DATASETS:
            m = p["datasets"].get(ds)
            cells.append(f"{m['achieved']:.2f} / {m['oracle']:.2f}" if m else "--")
        cells.append("/".join(f"{100*p['datasets'][ds]['wrong_instance_share']:.1f}"
                              if ds in p["datasets"] else "--" for ds in DATASETS))
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
