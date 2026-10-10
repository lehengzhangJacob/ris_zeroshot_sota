#!/usr/bin/env python
"""EXP-43 — write sota_table.json and ablation.json from fresh evidence.

Every measured row points at an evidence file written by this run; published
rows carry their citation.  Run after exp40 has produced the main_*.json files.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

OUT = Path("exp/out_final")
EVID = OUT / "evidence"
MAINS = {
    "refcoco": OUT / "main_refcoco.json",
    "refcocoplus": OUT / "main_refcocoplus.json",
    "refcocog": OUT / "main_refcocog.json",
    "grefcoco": OUT / "main_grefcoco.json",
}

# published zero-shot RIS numbers (val), as reported by their papers
PUBLISHED_ZERO_SHOT = [
    dict(name="Grad-CAM", venue="ICCV 2019", refcoco=30.22, refcocoplus=33.96, refcocog=33.05),
    dict(name="MaskCLIP", venue="CVPR 2023", refcoco=25.62, refcocoplus=27.49, refcocog=30.13),
    dict(name="Global-Local (SAM)", venue="CVPR 2023", refcoco=31.83, refcocoplus=34.97, refcocog=40.66),
    dict(name="Ref-Diff", venue="CoRR 2023", refcoco=37.21, refcocoplus=37.29, refcocog=44.02),
    dict(name="TAS", venue="EMNLP 2023", refcoco=39.84, refcocoplus=43.63, refcocog=46.62),
    dict(name="BMS", venue="TCSVT 2024", refcoco=40.77, refcocoplus=41.70, refcocog=48.32),
    dict(name="IteRPrimE", venue="AAAI 2025", refcoco=40.20, refcocoplus=44.20, refcocog=46.00),
    dict(name="LGD", venue="PR 2025", refcoco=45.41, refcocoplus=47.23, refcocog=47.31),
    dict(name="HybridGL (published)", venue="CVPR 2025",
         refcoco=49.48, refcocoplus=43.40, refcocog=51.25),
    dict(name="GLOD-RIS (published)", venue="TCSVT 2026",
         refcoco=58.42, refcocoplus=54.01, refcocog=57.85),
]
PUBLISHED_GREFCOCO = [
    dict(name="LAVT (supervised)", venue="CVPR 2022", grefcoco_ciou=57.64, grefcoco_giou=58.40),
    dict(name="ReLA (supervised)", venue="CVPR 2024", grefcoco_ciou=62.42, grefcoco_giou=63.60),
    dict(name="SAM4MLLM (supervised)", venue="ECCV 2024", grefcoco_ciou=67.83, grefcoco_giou=71.86),
    dict(name="InstAlign (supervised)", venue="arXiv 2411.15087",
         grefcoco_ciou=68.94, grefcoco_giou=74.34),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", type=Path, default=Path("exp/tuning"))
    args = ap.parse_args()
    EVID.mkdir(parents=True, exist_ok=True)

    mains = {}
    for k, p in MAINS.items():
        if p.is_file():
            mains[k] = json.loads(p.read_text())

    methods = []
    for row in PUBLISHED_ZERO_SHOT:
        r = dict(name=row["name"], venue=row["venue"], measured_by_us=False,
                 refcoco=dict(mIoU=row["refcoco"]),
                 refcocoplus=dict(mIoU=row["refcocoplus"]),
                 refcocog=dict(mIoU=row["refcocog"]), setting="zero-shot / training-free")
        methods.append(r)
    for row in PUBLISHED_GREFCOCO:
        methods.append(dict(name=row["name"], venue=row["venue"], measured_by_us=False,
                            grefcoco=dict(cIoU=row["grefcoco_ciou"], gIoU=row["grefcoco_giou"]),
                            setting="trained on gRefCOCO train"))

    # measured-by-us rows: copy the evidence so each row has a stable path
    def copy_ev(src: Path, name: str):
        dst = EVID / name
        if src.is_file():
            shutil.copyfile(src, dst)
            return str(dst)
        return None

    ev_rel = copy_ev(MAINS["refcoco"], "sota_released_and_ours.json") if "refcoco" in mains else None
    if "refcoco" in mains:
        m = mains["refcoco"]
        rows = m.get("rows", {})
        methods.append(dict(name="HybridGL released (our reproduction)", venue="this run",
                            measured_by_us=True, evidence=ev_rel,
                            refcoco=dict(mIoU=rows.get("released")),
                            refcocoplus=dict(mIoU=mains.get("refcocoplus", {}).get("rows", {}).get("released")),
                            refcocog=dict(mIoU=mains.get("refcocog", {}).get("rows", {}).get("released")),
                            setting="zero-shot / training-free, reproduced"))
        methods.append(dict(name="+ containment growth (control)", venue="this run",
                            measured_by_us=True, evidence=ev_rel,
                            refcoco=dict(mIoU=rows.get("snap")),
                            refcocoplus=dict(mIoU=mains.get("refcocoplus", {}).get("rows", {}).get("snap")),
                            refcocog=dict(mIoU=mains.get("refcocog", {}).get("rows", {}).get("snap")),
                            setting="zero-shot, control"))
        methods.append(dict(name="Ours: dense VLM box prior (shipped)", venue="this run",
                            measured_by_us=True, evidence=ev_rel,
                            refcoco=dict(mIoU=rows.get("ours")),
                            refcocoplus=dict(mIoU=mains.get("refcocoplus", {}).get("rows", {}).get("ours")),
                            refcocog=dict(mIoU=mains.get("refcocog", {}).get("rows", {}).get("ours")),
                            setting="zero-shot / training-free (frozen SAM+CLIP+VLM)"))
        methods.append(dict(name="Ours: no-box control", venue="this run",
                            measured_by_us=True, evidence=ev_rel,
                            refcoco=dict(mIoU=rows.get("ctrl_no_box")),
                            refcocoplus=dict(mIoU=mains.get("refcocoplus", {}).get("rows", {}).get("ctrl_no_box")),
                            refcocog=dict(mIoU=mains.get("refcocog", {}).get("rows", {}).get("ctrl_no_box")),
                            setting="zero-shot, control"))
    if "grefcoco" in mains:
        g = mains["grefcoco"]
        ev = copy_ev(MAINS["grefcoco"], "sota_grefcoco.json")
        ship = g.get("shipped") or {}
        methods.append(dict(name="Ours: learned selector + existence branch on gRefCOCO",
                            venue="this run", measured_by_us=True, evidence=ev,
                            grefcoco=dict(cIoU=ship.get("cIoU", g.get("cIoU")),
                                          gIoU=ship.get("gIoU", g.get("gIoU"))),
                            n=ship.get("n"),
                            setting="training-free features; rule tuned on held-out "
                                    "gRefCOCO train (never val)"))

    (OUT / "sota_table.json").write_text(json.dumps(dict(
        caption="Referring image segmentation, validation splits. mIoU unless noted; "
                "gRefCOCO uses the official cIoU/gIoU protocol.",
        methods=methods), indent=2))
    print(f"sota_table.json: {len(methods)} methods, "
          f"{sum(1 for m in methods if m.get('measured_by_us'))} measured by us")

    # ---------------- ablations ----------------
    rows = []
    if "refcoco" in mains:
        m = mains["refcoco"]
        r = m["rows"]
        ev = copy_ev(MAINS["refcoco"], "ablation_val_rows.json")
        def per_ds(key):
            return {ds: mains[ds]["rows"].get(key) for ds in ("refcoco", "refcocoplus", "refcocog")
                    if ds in mains}
        rows.append(dict(name="released HybridGL pipeline", evidence=ev, mIoU=per_ds("released")))
        rows.append(dict(name="+ containment growth", evidence=ev, mIoU=per_ds("snap")))
        rows.append(dict(name="+ dense VLM box prior (shipped)", evidence=ev, mIoU=per_ds("ours")))
        rows.append(dict(name="box prior removed (control)", evidence=ev,
                         mIoU=per_ds("ctrl_no_box")))
    for tag, label in (("noun", "box prompt: noun phrase (held-out tuning split)"),
                       ("sent", "box prompt: full sentence (held-out tuning split)")):
        p = Path(f"exp/out_final/optuna_study_{tag}.json")
        if p.is_file():
            d = json.loads(p.read_text())
            ev = copy_ev(p, f"ablation_study_{tag}.json")
            rows.append(dict(name=label, evidence=ev,
                             tuning_mIoU={k.replace("mIoU_", ""): v
                                          for k, v in d["best_per_dataset"].items()},
                             n_trials=d["n_trials"]))
    (OUT / "ablation.json").write_text(json.dumps(dict(rows=rows), indent=2))
    print(f"ablation.json: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
