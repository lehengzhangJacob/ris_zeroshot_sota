# Local Assets (verified 2026-10-01/02)

## Code

| Path | What it actually is | Runnable? |
|---|---|---|
| `/home/msj_team/Jacob/sys/RIS/*.tar.gz` | **LaTeX paper sources only** — CaR, HybridGL, IteRPrimE, LGD, RefDiff, TAS all contain **0 `.py` files** | ❌ not code |
| `/home/msj_team/Jacob/sys/RIS/repos/BMS` | github.com/pcc-99/BMS — README + 1 PDF, **placeholder** | ❌ |
| `/home/msj_team/Jacob/sys/RIS/repos/GLOD-RIS` | github.com/Wujj440/GLOD-RIS — **README only, 0 `.py`** | ❌ |
| `/home/msj_team/Jacob/sys/RIS/repos/HybridGL` | github.com/fhgyuanshen/HybridGL — **50 `.py`**, full pipeline, vendored `modified_CLIP` + `segment-anything` | ✅ |
| `/home/msj_team/Jacob/sys/RIS/repos/IteRPrimE` | github.com/VoyageWang/IteRPrimE — 69 `.py`, ALBEF + Mask2Former + stanza | ✅ (needs ckpts) |
| `/home/msj_team/Jacob/sys/RIS/repos/Ref-Diff` | github.com/kodenii/Ref-Diff — 82 `.py`, needs Stable Diffusion v1.5 + `pretrain/vitb16_stego_head.ckpt` | ✅ (heavier) |
| `/home/msj_team/Jacob/sys/glod_ris_repro` | **Independent GLOD-RIS reimplementation** (not official). `glod/` package + `run.py` + pytest suite | ✅ |

## The independent GLOD-RIS reproduction

Not official, and does **not** match the paper's numbers. But it is a well-built research
harness and is the most useful local asset for this project:

- `glod/core.py` — paper Eq. (1)–(7) verbatim (`decouple`, `gaussian_map`, `select_mask`), with
  every unspecified choice exposed as a `Config` field.
- `glod/vision.py` — frozen SAM ViT-H proposals (points_per_side=8, pred_iou/stability 0.7,
  min_region_area=800) + the **pinned official HybridGL G2L encoder** (`f7eb19bb…`), with a
  content-addressed **cache of masks and per-mask foreground/background features**.
- `glod/glm.py` — ZhipuAI `glm-4.1v-thinking-flash` client, key at `/home/msj_team/Jacob/sys/glm_key`,
  responses cached under `cache/glm`.
- `run.py` — `demo` / `evaluate` / `prefetch`; `evaluate` is resumable, writes `run.json`,
  `metrics.json`, `records/*.json` (all candidate scores + cache paths), `examples/*`.
- Ablation flags: `--dod-only`, `--no-dod`, `--no-category`, `--no-localization`,
  `--no-visual-grid`, `--hard-grid`, `--grid-size`, `--mu`, `--tau`, `--fusion-basis`,
  `--spatial-reduction`, `--score-transform`.

### What it has already produced

| Run | Expressions | oIoU | mIoU | oracle mIoU | Note |
|---|---|---|---|---|---|
| `outputs/smoke_dod` | 3 | 89.63 | 89.63 | 89.63 | trivial; 3 hand-picked refs |
| `outputs/pilot_glod` | 33 | 15.59 | 24.30 | 79.87 | complete seeded subset, with GLM |
| `outputs/full_refcoco_val` | 223 / 10834 | 28.05 | 36.00 | 83.54 | **aborted** — GLM API parse error |

Two facts matter for the proposal:

1. **The oracle gap is enormous.** On `full_refcoco_val`, the SAM pool contains a mask with
   mIoU 83.54 while the pipeline achieves 36.00 — a 47-point selection loss. On `pilot_glod`,
   24.30 vs 79.87 — a 56-point loss. The paper's own Fig. 6 reports ~62–65% oIoU / ~70–73%
   mIoU oracle utilisation, i.e. it acknowledges the ceiling is the limit.
2. **The GLM dependency is operational, not just architectural.** The full run died at
   223/10834 expressions with `GLM invalid response: Grid must contain exactly n*n binary values`
   — the 3×3 grid prompt/parser is brittle, and it is the single point of failure.

## Checkpoints present

- `glod_ris_repro/checkpoints/sam_vit_h_4b8939.pth` (2.56 GB) — SAM ViT-H
- `glod_ris_repro/checkpoints/ViT-B-16.pt` (351 MB) — OpenAI CLIP ViT-B/16
- also `/home/msj_team/fmo/MatAnyone/pretrained_models/sam_vit_h_4b8939.pth`

## Data present

- `glod_ris_repro/data/refer/{refcoco,refcoco+,refcocog}/{instances.json,refs(*).p}` — RefCOCO /
  RefCOCO+ (unc) / RefCOCOg (umd) annotations
- `glod_ris_repro/data/refer/images/train2014/COCO_train2014_*.jpg` — downloaded on demand per ref
  (only the evaluation images needed, not the full 13 GB train2014)
- **No gRefCOCO / GRES annotations locally.** `python prepare_data.py` only fetches the three
  RefCOCO-family sets.

## GPUs

4× RTX 4090 (49 GB each). At time of writing GPU 1 is occupied by another process (~23 GB, 74–98%);
GPU 0, 2, 3 are effectively free. Credit the other process's card as unavailable.

## Environment

`glod-ris` — Python 3.10.18, PyTorch 2.2.1+cu121, torchvision 0.17.1+cu121, NumPy 1.26.4.
This differs from HybridGL's README (PyTorch 2.0.1/cu117), which must be recorded in any
experiment log. `torch.cuda.is_available()` is true.
