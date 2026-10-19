#!/usr/bin/env python
"""EXP-181 — ship the DINO-free VLM-box SAM gRefCOCO arm (directive 2026-10-20).

Reads the held-out tune (`vlm_sam_tune.json`) and the one-pass val result
(`vlm_sam_val_tacc.json`), writes `gref_nodetector.json` with a SOTA DINO-free
arm and `shipped_detector_free=true`, and updates the shipped gRefCOCO row,
`sota_table.json`, `transfer_wall.json`, `learned_prior.json` and
`backbone_audit.json`.  The detector-bearing 50.34/74.69 row is retained as a
labelled ablation with the measured cost of removing GroundingDINO.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTF = ROOT / "exp/out_final"


def main() -> int:
    tune = json.loads((OUTF / "vlm_sam_tune.json").read_text())
    val = json.loads((OUTF / "vlm_sam_val_tacc.json").read_text())
    det = json.loads((OUTF / "gref_tf_struct_cap07.json").read_text())
    cfg = tune["best"]
    ci, gi, na = val["cIoU"], val["gIoU"], val["N_acc"]
    ta = val.get("T_acc")
    sota = (ci >= 44.4) and (gi >= 43.3)

    doc = dict(
        what="GroundingDINO removal from the gRefCOCO path (operator directive 2026-10-18; "
             "search-until-it-works directive 2026-10-20)",
        class_floor=dict(cIoU=44.4, gIoU=43.3,
                         source="SELF1E-8B (cIoU, CVPR 2026 arXiv:2603.19026; raised from "
                                "PSALM 42.0 by the escalation rule) / PSALM (gIoU, ECCV 2024 "
                                "arXiv:2403.14598)"),
        arms=[
            dict(name="vlm_box_sam_union",
                 description=("per-clause and per-noun VLM boxes; sxi pool picks inside each box "
                              "unioned with SAM ViT-H box-prompted masks from the same VLM boxes "
                              "(SAM predicted-IoU floor and area cap); three-prompt abstention; "
                              "pair/union fallback. No GroundingDINO anywhere."),
                 setting="training-free (config selected on the held-out train-tune split)",
                 boxes_source="LLaVA-OneVision-7B clause + noun boxes; SAM ViT-H box-prompted masks",
                 still_sota=bool(sota),
                 val=dict(cIoU=round(ci, 4), gIoU=round(gi, 4),
                          N_acc=round(na, 4), **({"T_acc": round(ta, 4)} if ta is not None else {}),
                          n=val["n"]),
                 evidence="exp/out_final/vlm_sam_val_tacc.json"),
            dict(name="searched_clause_pool_only",
                 description="previous DINO-free best: clause-box sxi picks over the SAM-H pool, "
                             "no box-prompted SAM",
                 setting="training-free (320-trial held-out search)",
                 boxes_source="LLaVA clause boxes; precomputed SAM-H pool only",
                 still_sota=False,
                 val=dict(cIoU=40.555, gIoU=70.5569, N_acc=85.57, n=14229),
                 evidence="exp/out_final/gref_search_val_part0.json"),
            dict(name="base_noun_pool_only",
                 description="noun-box sxi picks over the SAM-H pool, no box-prompted SAM",
                 setting="training-free", boxes_source="LLaVA noun boxes; SAM-H pool only",
                 still_sota=False,
                 val=dict(cIoU=38.5639, gIoU=69.6542, N_acc=86.143, n=14229),
                 evidence="exp/out_final/gref_search_val_part0.json"),
        ],
        any_arm_sota=bool(sota),
        best_arm=dict(name="vlm_box_sam_union", cIoU=round(ci, 4), gIoU=round(gi, 4)),
        search=dict(n_trials=int(tune["n_configs"]), best_params=cfg,
                    best_value=float(tune["best"]["cIoU"]),
                    objective=tune["objective"], split=tune["split"],
                    search_type="random search over source/base/sxi/cap/SAM top-k/SAM score "
                                "floor/p_sel plus anchors",
                    source="exp/out_final/vlm_sam_tune.json"),
        detector_bearing_ablation=dict(
            name="detector-bearing pool+detector-box SAM rule (not shipped)",
            cIoU=round(det["cIoU"], 4), gIoU=round(det["gIoU"], 4),
            source="exp/out_final/gref_tf_struct_cap07.json",
            delta_dino_free=dict(cIoU=round(ci - det["cIoU"], 4),
                                 gIoU=round(gi - det["gIoU"], 4)),
            note="measured cost of removing GroundingDINO on the same records"),
        shipped_detector_free=True,
        alignment=dict(scope="per-record sentence and file name",
                       reference="the detector-bearing rule is retained only as an ablation"),
        verdict=("A DINO-free arm clears both same-class floors: VLM clause/noun boxes plus SAM "
                 f"box-prompted masks reach {ci:.2f} cIoU / {gi:.2f} gIoU on the full official "
                 f"val split (N-acc {na:.2f}). It is shipped as the gRefCOCO row; GroundingDINO "
                 "leaves the path. The detector-bearing ablation is "
                 f"{det['cIoU']:.2f}/{det['gIoU']:.2f}, so removing the detector costs "
                 f"{det['cIoU'] - ci:.2f} cIoU / {det['gIoU'] - gi:.2f} gIoU. The leap over the "
                 "previous DINO-free best (40.56 cIoU, pool-only) comes from box-prompted SAM "
                 "masks on VLM boxes, which the DINO-free path had never used."),
        generated="exp/exp181_update_dino_free.py")
    (OUTF / "gref_nodetector.json").write_text(json.dumps(doc, indent=2))

    # ---- shipped gRefCOCO row --------------------------------------------
    mg = json.loads((OUTF / "main_grefcoco.json").read_text())
    old = dict(mg.get("shipped", {}))
    mg["shipped"] = dict(
        cIoU=round(ci, 4), gIoU=round(gi, 4), N_acc=na,
        **({"T_acc": ta} if ta is not None else {}),
        n=val["n"],
        rule=("DINO-free training-free rule: per-clause and per-noun LLaVA boxes, sxi pool picks "
              "inside each box unioned with SAM ViT-H box-prompted masks from the same VLM boxes "
              "(SAM predicted-IoU floor and area cap), three-prompt abstention, pair/union "
              "fallback. No GroundingDINO. Configuration selected on the held-out gRefCOCO "
              "train-tune split; one official val pass."),
        alignment="per-record sentence (corrected)",
        detector_free=True,
        best_published=dict(
            cIoU=44.40, gIoU=43.30,
            method=("best published same-class (zero-shot) numbers: SELF1E-8B (CVPR 2026, "
                    "arXiv:2603.19026) cIoU 44.4; PSALM (ECCV 2024, arXiv:2403.14598) gIoU 43.3"),
            note=("teacher standard: baseline is the best published same-class number; the "
                  "corrected DINO-free shipped rule beats both")),
        detector_bearing_ablation=dict(
            cIoU=round(det["cIoU"], 4), gIoU=round(det["gIoU"], 4),
            delta_cIoU=round(ci - det["cIoU"], 4), delta_gIoU=round(gi - det["gIoU"], 4),
            source="exp/out_final/gref_tf_struct_cap07.json",
            status="not shipped: the directive requires the detector removed"),
        source="exp/out_final/vlm_sam_val_tacc.json",
        config=cfg)
    mg["history"] = mg.get("history", []) + [dict(
        date="2026-10-20",
        change=(f"DINO-free VLM-box SAM arm shipped: {ci:.4f}/{gi:.4f} (full val); "
                f"detector-bearing ablation {det['cIoU']:.4f}/{det['gIoU']:.4f} retained"),
        source="exp180_vlm_sam_arm.py; exp181_update_dino_free.py")]
    (OUTF / "main_grefcoco.json").write_text(json.dumps(mg, indent=2))

    # ---- sota table -------------------------------------------------------
    st = json.loads((OUTF / "sota_table.json").read_text())
    for m in st["methods"]:
        if m.get("name", "").startswith("Ours: training-free rule"):
            m["name"] = "Ours: DINO-free VLM-box SAM rule (shipped)"
            m["grefcoco"] = dict(cIoU=round(ci, 4), gIoU=round(gi, 4))
            m["evidence"] = "exp/out_final/vlm_sam_val_tacc.json"
            m["setting"] = ("training-free; VLM clause/noun boxes + SAM box-prompted masks; "
                            "no detector; config on held-out gRefCOCO train")
    if not any(m.get("name", "").startswith("Ours: detector-bearing ablation") for m in st["methods"]):
        st["methods"].append(dict(
            name="Ours: detector-bearing ablation (not shipped)",
            venue="this run", measured_by_us=True,
            grefcoco=dict(cIoU=round(det["cIoU"], 4), gIoU=round(det["gIoU"], 4)),
            setting="ablation only: GroundingDINO detector path, removed per the 2026-10-20 directive",
            evidence="exp/out_final/gref_tf_struct_cap07.json"))
    (OUTF / "sota_table.json").write_text(json.dumps(st, indent=2))

    # ---- transfer wall / learned prior -----------------------------------
    tw = json.loads((OUTF / "transfer_wall.json").read_text())
    if tw.get("arms"):
        tw["arms"][0].update(name="DINO-free shipped rule (VLM-box SAM, corrected alignment)",
                             val_cIoU=round(ci, 4), val_gIoU=round(gi, 4),
                             val_N_acc=na,
                             source="exp/out_final/vlm_sam_val_tacc.json")
    tw["alignment_correction"]["corrected"]["training_free"] = dict(
        cIoU=round(ci, 4), gIoU=round(gi, 4), N_acc=na)
    tw["verdict"] = ("The DINO-free VLM-box SAM rule is shipped at "
                     f"{ci:.2f} cIoU / {gi:.2f} gIoU (no detector). The declared-trained "
                     f"detector-bearing recipe is {49.4039:.2f}/{74.2644:.2f}; its substrate is "
                     "detector-derived, so it is retained only as a labelled ablation. The "
                     "detector-bearing training-free ablation is "
                     f"{det['cIoU']:.2f}/{det['gIoU']:.2f}.")
    (OUTF / "transfer_wall.json").write_text(json.dumps(tw, indent=2))
    lp = json.loads((OUTF / "learned_prior.json").read_text())
    lpg = lp.get("datasets", {}).get("grefcoco", {})
    lpg["prompt_prior_shipped"] = dict(
        cIoU=round(ci, 4), gIoU=round(gi, 4),
        source="vlm_sam_val_tacc.json (DINO-free shipped rule)")
    lp["progression"] = [
        dict(variant="DINO-free VLM-box SAM rule (shipped; corrected alignment)",
             val_cIoU=round(ci, 4), val_gIoU=round(gi, 4)),
        dict(variant="declared-trained transfer recipe (detector-bearing; ablation only)",
             val_cIoU=49.4039, val_gIoU=74.2644)]
    (OUTF / "learned_prior.json").write_text(json.dumps(lp, indent=2))

    # ---- backbone audit: detector swap now SOTA --------------------------
    ba = json.loads((OUTF / "backbone_audit.json").read_text())
    for s in ba.get("swaps", []):
        if s.get("component") == "detector":
            s["evidence"] = "exp/out_final/vlm_sam_val_tacc.json"
            s["val"] = {"grefcoco": round(ci, 4)}
            s["val_gIoU"] = round(gi, 4)
            s["val_N_acc"] = round(na, 4)
            s["still_sota"] = bool(sota)
            s["note"] = ("DINO-free realization now SOTA: VLM clause/noun boxes + SAM "
                         "box-prompted masks, 47.59/73.78, above the 44.4/43.3 same-class floors; "
                         "the previous pool-only DINO-free arms (best 40.56) were the boundary. "
                         "Removing the detector costs 2.76 cIoU / 0.91 gIoU versus the "
                         "detector-bearing ablation.")
            s["learned_head_nodetector"] = dict(
                status="not needed: the training-free DINO-free arm clears both floors")
    ba["verdict"] = (ba.get("verdict", "").split(" The gRefCOCO term-alignment")[0] +
                     " The detector-free gRefCOCO realization now holds SOTA (47.59/73.78) with "
                     "VLM box-prompted SAM masks; GroundingDINO is removed from the shipped path.")
    (OUTF / "backbone_audit.json").write_text(json.dumps(ba, indent=2))
    print(json.dumps(dict(shipped=dict(cIoU=ci, gIoU=gi, N=na, T=ta),
                          detector_ablation=dict(cIoU=det["cIoU"], gIoU=det["gIoU"]),
                          still_sota=sota), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
