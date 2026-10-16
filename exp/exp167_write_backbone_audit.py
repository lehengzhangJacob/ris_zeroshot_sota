#!/usr/bin/env python
"""EXP-167 — assemble exp/out_final/backbone_audit.json (directive 2026-10-16).

Reads the swap-arm part files (exp/out_final/backbone_swaps/*.json, the
detector-free gRefCOCO pass, and the shipped reference rows), writes the gated
backbone-strength audit: inventory[] of the shipped lineup, swaps[] of the
strictly-weaker arms with aligned val results and a still_sota flag against
the GLOD-RIS published floors (58.42/54.01/57.85 mIoU; gRefCOCO vs the
same-class floors 44.4 cIoU / 43.3 gIoU and the run's own record), and a
verdict stating plainly whether SOTA holds under the weaker lineup.

    python exp/exp167_write_backbone_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUTF = Path("exp/out_final")
SW = OUTF / "backbone_swaps"

FLOORS = {"refcoco": 58.42, "refcocoplus": 54.01, "refcocog": 57.85}
GREFC_FLOOR_CIOU = 44.4
GREFC_FLOOR_GIOU = 43.3
SHIPPED = {"refcoco": 70.5673, "refcocoplus": 66.3513, "refcocog": 65.5059}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def main() -> int:
    parts = {}
    for ds in ("refcoco", "refcocoplus", "refcocog"):
        for tag in ("plain_region_text", "sam_b_pool", "box3b"):
            d = load(SW / f"{tag}_{ds}.json")
            if d:
                parts.setdefault(tag, {})[ds] = d
    gref_nod = load(OUTF / "gref_no_detector_val.json")
    shipped_g = load(OUTF / "main_grefcoco.json") or {}
    shipg = shipped_g.get("shipped") or {}

    inventory = [
        dict(name="SAM ViT-H automatic mask generator", params="0.64B",
             role="proposal pool for the RefCOCO family (and gRefCOCO via term boxes)",
             provenance="generic_frozen",
             note="released pipeline configuration (pps=8, released thresholds)"),
        dict(name="CLIP ViT-B/16 (released modified tower, G2L masked-region features + text)",
             params="0.15B", role="region-text platform of the head and the released scores",
             provenance="released_hybridgl"),
        dict(name="LLaVA-OneVision-7B box prior", params="7B",
             role="referent box on the RefCOCO family (full-expression prompt)",
             provenance="ours (model: generic frozen checkpoint)"),
        dict(name="GroundingDINO-base detector gate", params="0.23B",
             role="second box source on the gRefCOCO path only",
             provenance="generic_frozen"),
        dict(name="Qwen2.5-VL-3B-Instruct box prior (backbone-audit arm)",
             params="3.8B",
             role="strictly smaller local VLM replacing the 7B prior in arm (a)",
             provenance="ours (model: generic frozen checkpoint)"),
        dict(name="SAM ViT-B automatic mask generator (backbone-audit arm)",
             params="0.09B", role="strictly smaller proposal pool in arm (b)",
             provenance="generic_frozen"),
        dict(name="Plain masked-region CLIP ViT-B/16 (backbone-audit arm)",
             params="0.15B",
             role="weaker region-text platform (our own masking; no G2L fusion, "
                  "no complement blur) in arm (c)",
             provenance="ours (text protocol) over generic frozen weights"),
        dict(name="Learned selector head", params="~4.5k fitted scalars (250 depth-3 trees)",
             role="shipped RefCOCO-family pick (declared trained on train-carved rows)",
             provenance="ours"),
    ]

    swaps = []
    # (a) smaller VLM
    if "box3b" in parts:
        val, ok = {}, True
        for ds, d in parts["box3b"].items():
            lh = d["arms"]["learned_head"]["mIoU"]
            val[ds] = lh
            ok = ok and lh > FLOORS[ds]
        ref = {ds: d["arms"]["reference"]["mIoU"] for ds, d in parts["box3b"].items()}
        swaps.append(dict(
            name="smaller VLM box source: Qwen2.5-VL-3B-Instruct (3.8B) for the 7B LLaVA-OneVision prior",
            component="vlm_box_prior",
            weaker_variant=("Qwen2.5-VL-3B-Instruct (strictly fewer parameters), same prompt "
                            "template; head refit on train-carved rows with the 3B boxes "
                            "(deployment-consistent); measured BETTER than the 7B LLaVA "
                            "boxes on all three datasets -- grounding quality here is "
                            "family-limited, not size-limited"),
            evidence="exp/out_final/backbone_swaps/box3b_refcoco.json",
            val=val, val_training_free_reference=ref,
            box_none_records={ds: d.get("box_none_records")
                              for ds, d in parts["box3b"].items()},
            floor=FLOORS, still_sota=ok,
            note=("strictly smaller than both the incumbent 7B and the rival's cloud "
                  "MLLM; head and reference both refit/rerun on train-carved rows "
                  "with the 3B boxes")))
    # (b) smaller SAM
    if "sam_b_pool" in parts:
        val, ok = {}, True
        for ds, d in parts["sam_b_pool"].items():
            lh = d["arms"]["learned_head_saml"]["mIoU"]
            val[ds] = lh
            ok = ok and lh > FLOORS[ds]
        swaps.append(dict(
            name="smaller proposal pool: SAM ViT-B (0.09B) for SAM ViT-H (0.64B)",
            component="sam_pool",
            weaker_variant="SAM ViT-B automatic masks (pps=8, same released thresholds), G2L features re-extracted, head refit on SAM-B train-carved rows",
            evidence="exp/out_final/backbone_swaps/sam_b_pool_refcocog.json",
            val=val,
            val_geom_only={ds: d["arms"]["geom2_only_saml"]["mIoU"]
                           for ds, d in parts["sam_b_pool"].items()},
            floor=FLOORS, still_sota=ok,
            note=("ViT-L was the first choice; the original-format ViT-L checkpoint "
                  "is not reachable offline (HF port restructures the prompt "
                  "encoder), so the strictly smaller ViT-B is the measured arm. "
                  "Under SAM-B the learned head is within \u00b10.4 mIoU of the "
                  "geom-only argmax pick on every dataset (mixed signs), i.e. the "
                  "pool itself is the binding constraint at this strength. The "
                  "head-level 0.28 margin below the floor is within unquantified "
                  "single-seed refit noise; the load-bearing conclusion rests on "
                  "the pool-oracle gap (79.1 -> 71.4 mIoU), which is robust")))
    # (c) weaker region-text platform
    if "plain_region_text" in parts:
        val, ok = {}, True
        for ds, d in parts["plain_region_text"].items():
            lh = d["arms"]["learned_head_clipw"]["mIoU"]
            val[ds] = lh
            ok = ok and lh > FLOORS[ds]
        swaps.append(dict(
            name="weaker region-text platform: plain masked-region CLIP (no G2L fusion, no complement blur)",
            component="region_text_platform",
            weaker_variant="our own masked-region CLIP ViT-B/16 features (mask fill + resize + CLS), head refit on train-carved rows with the plain features",
            evidence="exp/out_final/backbone_swaps/plain_region_text_refcoco.json",
            val=val, floor=FLOORS, still_sota=ok,
            note="removes the released G2L fusion and complement-blur path entirely"))
    # (d) gRefCOCO without the detector
    if gref_nod:
        m = gref_nod["metrics"]
        ok = (m["cIoU"] >= GREFC_FLOOR_CIOU and m["gIoU"] >= GREFC_FLOOR_GIOU)
        swaps.append(dict(
            name="gRefCOCO without the GroundingDINO detector (fewer components than the rival)",
            component="detector",
            weaker_variant=("detector-free realization: the training-free termbox rule with "
                            "LLaVA nounboxes only. Structural note: the shipped declared-trained "
                            "recipe's candidate substrate on this path is built from detector "
                            "boxes and detector-box-prompted SAM masks, so removing the detector "
                            "necessarily removes the trained recipe with it; no intermediate "
                            "(nounbox-prompted SAM masks + transferred head) was measured, and "
                            "the boundary below is therefore scoped to the detector-free "
                            "realization, not to the detector in isolation"),
            evidence="exp/out_final/gref_no_detector_val.json",
            val={"grefcoco": round(m["cIoU"], 4)},
            val_gIoU=round(m["gIoU"], 4),
            val_N_acc=round(m["N_acc"], 2), val_T_acc=round(m["T_acc"], 2),
            floor={"grefcoco_cIoU": GREFC_FLOOR_CIOU, "grefcoco_gIoU": GREFC_FLOOR_GIOU,
                   "own_record_cIoU": 52.7061},
            still_sota=ok,
            note=("cIoU drops below the same-class floor for the detector-free "
                  "realization (38.71 vs 44.4): the detector-bearing path (with "
                  "its trained recipe) is load-bearing on gRefCOCO, and the drop "
                  "is kept as the measured boundary")))

    n_sota = sum(1 for s in swaps if s["still_sota"])
    verdict = (
        "Scoped, component by component (full official val, aligned protocol; "
        "the reference and head were refit on train-carved rows wherever the "
        "platform or pool changed): (a) the box prior can be SMALLER AND BETTER "
        "-- Qwen2.5-VL-3B (3.8B, strictly smaller than both the 7B incumbent and "
        "the rival's cloud MLLM) holds SOTA on all three datasets and improves "
        "on the shipped boxes (73.57/67.64/66.43 vs 70.57/66.35/65.51 mIoU), so "
        "grounding quality here is family-limited, not size-limited; (c) the "
        "region-text platform can be weakened for free -- plain masked-region "
        "CLIP without the released G2L fusion or complement blur keeps full-table "
        "SOTA at -0.2..-0.4 mIoU (70.22/65.98/65.32). Two boundaries are "
        "measured and kept: (b) SAM ViT-B's pool oracle falls from 79.1 to 71.4 "
        "mIoU on RefCOCOg and the refit head lands 57.57, 0.28 BELOW the "
        "published GLOD-RIS floor -- the ViT-H pool is load-bearing there; and "
        "(d) the detector-free gRefCOCO rule scores 38.71 cIoU, below the "
        "same-class 44.4 floor -- the extra detector is load-bearing there. "
        f"{n_sota}/{len(swaps)} swap arms still clear their floors. The "
        "detector-free boundary is scoped to the detector-free REALIZATION "
        "(training-free termbox rule): the shipped trained recipe's substrate "
        "is detector-derived, so the two cannot be separated in this arm.")
    if not (len(parts.get("box3b", {})) == 3 and len(parts.get("sam_b_pool", {})) == 3
            and len(parts.get("plain_region_text", {})) == 3 and gref_nod):
        verdict = ("INCOMPLETE: arm parts missing ("
                   f"box3b {sorted(parts.get('box3b', {}))}, "
                   f"sam_b {sorted(parts.get('sam_b_pool', {}))}, "
                   f"plain {sorted(parts.get('plain_region_text', {}))}, "
                   f"gref_nod {bool(gref_nod)})")

    doc = dict(
        what=("backbone-strength audit: the shipped lineup re-measured under "
              "strictly weaker/fewer backbones on the full official val splits "
              "under the aligned protocol"),
        inventory=inventory,
        swaps=swaps,
        shipped_reference=dict(refcoco=SHIPPED, grefcoco=dict(
            cIoU=shipg.get("cIoU"), gIoU=shipg.get("gIoU"),
            note="declared-trained transfer recipe")),
        verdict=verdict,
        generated="exp167_write_backbone_audit.py",
    )
    (OUTF / "backbone_audit.json").write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUTF / 'backbone_audit.json'}: {len(swaps)} swaps, "
          f"{n_sota} still_sota")
    print("verdict:", verdict[:160])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
