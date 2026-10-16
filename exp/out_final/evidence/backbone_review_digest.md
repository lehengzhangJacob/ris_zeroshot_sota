# Review digest — backbone-strength audit (2026-10-16 directive)

Machine artifact: `exp/out_final/backbone_audit.json` (gated). Arm parts:
`exp/out_final/backbone_swaps/{box3b,sam_b_pool,plain_region_text}_<dataset>.json`,
`exp/out_final/gref_no_detector_val.json`. Scripts: `exp/exp164_boxer3b.py`
(3B box extraction), `exp/exp165_backbone_swaps.py` (SAM-B pipeline, plain
region-text platform, 3B eval), `exp/exp163_gref_no_detector.py` (detector-free
gRefCOCO pass), `exp/exp167_write_backbone_audit.py` (assembler; also the
`download_chunked.py` mirrors helper). Paper: complexity + limitations updates,
appendix note.

## What the directive asked and what was measured

Strictly weaker/fewer backbones than the rival's (published GLOD-RIS floors
58.42 / 54.01 / 57.85 mIoU; gRefCOCO vs same-class floors 44.4 cIoU / 43.3
gIoU and the run's own 52.71 record), each arm under a deployment-consistent
head (refit on the train-carved tune rows whenever a changed component feeds
the head's inputs -- box source, feature platform or pool; the head's terms
consume boxes and region features directly, so each arm refits), on the full
official val splits under the aligned protocol:

- (a) **smaller VLM** — Qwen2.5-VL-3B-Instruct (3.8B) replacing
  LLaVA-OneVision-7B for the boxes; head refit on the tune rows with the 3B
  boxes. Full val, learned head: **refcoco 73.57 / refcocoplus 67.64 /
  refcocog 66.43 mIoU** (shipped 7B-boxes: 70.57/66.35/65.51) — the SMALLER
  model is BETTER on all three datasets (grounding quality here is
  family-limited, not size-limited; consistent with the earlier measured
  Qwen-family box-IoU advantage). Still SOTA with margin. Reference arm under
  3B boxes: 71.09/65.79/65.72. Box-none answers: 19/38/23 records.
- (b) **smaller pool** — SAM ViT-B (0.09B) replacing ViT-H (0.64B): pools and
  G2L features regenerated, per-proposal arrays re-derived from the new masks
  (the cached record arrays belong to the SAM-H pool), head refit on SAM-B
  tune rows. Full val: **refcoco 62.06 / refcocoplus 58.75 / refcocog 57.57**
  vs floors 58.42/54.01/57.85 — RefCOCOg lands 0.28 BELOW its floor; the SAM-B
  pool oracle is 71.4 vs 79.1 (SAM-H) on RefCOCOg, so the pool is load-bearing
  there. ViT-L was the first choice; the original-format ViT-L checkpoint is
  not reachable offline (the HF port restructures the prompt encoder), so the
  strictly smaller ViT-B is the measured arm.
- (c) **weaker region-text platform** — our own masked-region CLIP features
  (mask fill + resize + CLS pool; no released G2L fusion, no complement blur),
  head refit on the tune rows. Full val: **refcoco 70.22 / refcocoplus 65.98 /
  refcocog 65.32 mIoU** (shipped learned head: 70.57/66.35/65.51) — the
  released fusion path is worth ~0.3 mIoU. Still SOTA.
- (d) **gRefCOCO without GroundingDINO** — the training-free termbox rule with
  LLaVA nounboxes only (no detector boxes, no detector-box-prompted SAM
  masks): **cIoU 38.71 / gIoU 69.69** vs same-class floors 44.4 / 43.3. The
  arm DROPS BELOW the cIoU floor — kept as the measured boundary, scoped to
  the detector-free REALIZATION: the shipped declared-trained recipe's
  candidate substrate is detector-derived, so detector removal necessarily
  removes it too; the intermediate arm (nounbox-prompted SAM masks + the
  trained recipe) was not measured and is noted as such.

## Review questions

1. Claim-vs-artifact consistency in `backbone_audit.json` (inventory >= 4,
   swaps >= 4, >= 3 distinct components, fresh evidence, val{} numeric,
   still_sota bools, verdict scoping).
2. Is the deployment-consistency discipline sound (refits only where the
   platform/pool changes; the clipw and box3b arms' refits declared; the
   no-detector arm allowed to serve the training-free realization directly)?
3. Is the boundary reported honestly (no-detector below the floor; ViT-L
   unavailability noted; single-seed note carried from the previous cycle)?
4. Must-fix list, verdict, score.
