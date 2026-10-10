# Training-free zero-shot Referring Image Segmentation — same-class SOTA

**Status: active research, private backup. Unpublished — CVPR submission in preparation. Do not redistribute.**

## What this is

A training-free zero-shot RIS method: a frozen SAM ViT-H candidate pool,
the released HybridGL scores (frozen CLIP ViT-B/16), and a *dense localization
prior* — a frozen local VLM (LLaVA-OneVision-7B) asked for the referent's
bounding box with the **full expression**; selection is box-fill over the
candidate pool. No component is trained; the only fitted quantities are the
selector's 33 coefficients, fitted on held-out train-derived rows.

Base repo: https://github.com/fhgyuanshen/HybridGL (released pipeline, pinned
commit). GLOD-RIS (TCSVT 2026) is the comparison target; its local reproduction
lives in the lab (`glod_ris_repro`).

## Headline results (aligned protocol, full official val splits)

| dataset            | ours mIoU | ours oIoU | GLOD-RIS published (mIoU / oIoU) |
|--------------------|-----------|-----------|----------------------------------|
| RefCOCO val        | **68.53** | **63.40** | 58.42 / 50.85 |
| RefCOCO+ val       | **64.67** | **58.31** | 54.01 / 45.96 |
| RefCOCOg val       | **65.12** | **59.77** | 57.85 / 50.03 |
| gRefCOCO val       | cIoU **51.32**, gIoU **74.91**, N_acc 86.14 | — | same-class floors 44.4 / 43.3 (SELF1E-8B / PSALM, zero-shot) |

Component-matched control: with GLOD-RIS's own MLLM (GLM-4.1V-Thinking-Flash
via API, n=2000 samples/set) the pipeline still scores 67.34 / 64.53 / 63.37
mIoU — far above the published row; the gain is the mechanism, not the prior.

All numbers are protocol-audited (`exp/out_final/protocol_audit.json`) and
accepted by the machine gate: `python3 extreme_bar.py` prints PASS.

## Layout

- `paper/` — LaTeX sources, tables, figures, compiled `main.pdf`
- `exp/*.py` — all experiment scripts (evidence chain is fully scripted)
- `exp/out_final/` — final artifacts: `main_*.json` (shipped rows),
  `sota_table.json` (dual-metric), `ablation.json`, `protocol_audit.json`,
  `glm_mllm_swap.json`, `sensitivity.json`, `stability.json`,
  `complexity.json`, `gref_wall.json`, `optuna_study*.json`, `evidence/`
- `refine-logs/` — METHOD_GRAVEYARD, run state, final reports
- `extreme_bar.py` — the acceptance gate (source of truth for "done")
- `run_aris.sh` — the supervisor loop that drives the autonomous run

**Not included** (stay on the lab machine): raw record stores and model-output
caches (`exp/out_extreme/`, `exp/out_vlm_boxes/`, `exp/out_glm_boxes/`, ...),
run logs, reference PDFs. Evidence paths inside the JSONs point to those
locations on the lab machine.

## Reproduce

```
conda activate glod-ris
python3 extreme_bar.py          # acceptance gate
bash run_aris.sh                # autonomous supervisor (resumes session state)
```

Key checks: 4 full val splits present, dual-metric SOTA table, protocol audit,
analysis pack (sensitivity/stability/complexity), component-matched GLM test,
and strict same-class floors on every dataset.
