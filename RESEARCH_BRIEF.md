# Research Brief: Referring Image Segmentation

## Problem Statement

老师的任务是：围绕指代图像分割做调研，综合 GLOD-RIS 和进一步文献，选定一个 baseline，并在此基础上做改进。调研和方案已经完成。现在做改进实验，实验打脸就改方法再跑，然后写成一篇 IEEE Transactions 论文。不要重做文献调研。

GLOD-RIS 是零样本 RIS：不训练分割网络，用 SAM 出候选，用正交投影去掉 CLIP 的共享全局分量，再用九宫格高斯做定位。它是一个候选选择器，上限受 proposal 质量、固定 3×3 网格和外部 MLLM 限制。

本地 `papers/` 里还有同一方向的近期零样本 RIS。需要先读这些论文，再决定 baseline 是 GLOD-RIS 本身，还是另一篇更适合改的工作。

## Background

- **Field**: 计算机视觉
- **Sub-area**: 零样本指代图像分割（RIS）。输入是图像加指代表达，输出是掩码。
- **Key papers**: `papers/` 下的 PDF。锚点是 GLOD-RIS。Bidirectional Mask Selection、HybridGL、IteRPrimE、LGD、RefDiff、TAS 属于本任务。
- **What I already tried**: 读过 GLOD-RIS。还没有复现实验。
- **What didn't work**: 尚无。

## Constraints

- **Compute**: 4× RTX 4090 全部可用。RVOS 流程和 `/home/wyh` 的进程也在用这些卡。开跑前看 `nvidia-smi`，把作业放到还有空闲显存的卡上。不要杀已有进程。
- **Conda**: 用 `/home/anaconda3`。实验和环境安装只用已有环境 `glod-ris`（`/home/msj_team/.conda/envs/glod-ris`，Python 3.10，PyTorch 2.2.1+cu121，CUDA 可用）。先 `source /home/anaconda3/etc/profile.d/conda.sh && conda activate glod-ris`。不要改 base，不要动 `humanOmni`、`aris-rvos` 和其他环境。不要从 download.pytorch.org 下载 PyTorch。
- **Timeline**: 不设天数上限。审稿 6 分、技能里的 4 轮上限、以及“接近可写”，都不是结束。`python3 extreme_bar.py` 不通过，就继续做实验，禁止写投稿稿。
- **Target venue**: CVPR。现有 `paper/` 是被退回的草稿，不能当作成果。数字必须来自这次新实验的日志。
- **Code**: `/home/msj_team/Jacob/sys/RIS/*.tar.gz` 是论文 LaTeX，不是可运行仓库。改进的起点必须是该数据集上已发表的 SOTA，不能再从 HybridGL 往上叠选择器。RefCOCO、RefCOCO+、RefCOCOg 的起点是 GLOD-RIS。作者仓库 https://github.com/Wujj440/GLOD-RIS 没有可运行代码；本地 `/home/msj_team/Jacob/sys/glod_ris_repro` 是按论文写的独立实现，先让它在验证集上达到公布的 58.42 / 54.01 / 57.85，再在这份权重上改。对不上公布数字，就不要把 HybridGL 的结果当成已经站在 SOTA 上。gRefCOCO 的起点是 InstAlign，先找到并跑通它的发布代码，再改它。HybridGL 只作为更弱的对照。
- **Papers**: 允许下载论文 PDF。主实验四个数据集都要跑全验证集：RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO。前三个的标注和按需 COCO 图像已经在 `glod_ris_repro/data/refer`。gRefCOCO 允许下载。不要为了 RefCOCO 再下一整份重复图像。

## What I'm Looking For

- [x] Improvement on existing method
- 选定一个 baseline，说明为什么不是其他篇
- 一个超过一行包含关系截断的改进。seed-anchored gated growth 已被 `parent_strict` 解释掉（+0.0670 对 +0.0654），不要把它写成贡献。
- 方法要在 RefCOCO、RefCOCO+、RefCOCOg、gRefCOCO 四个全验证集上超过已发表的强基线，而不是只超过“直接取最高分”。一行截断若能解释涨幅，该方法作废，换方法重跑。
- 投稿前必须有：这四个数据集各自的主表、至少六个已发表方法的 SOTA 表且其中至少三个由本次实跑、结构图、不少于五组的组件消融、附录。
- 全线 SOTA：RefCOCO / RefCOCO+ / RefCOCOg 的 mIoU 必须超过 GLOD-RIS 公布的 58.42 / 54.01 / 57.85。gRefCOCO 验证集的 cIoU 和 gIoU 必须超过 InstAlign 公布的 68.94 / 74.34。若查到更新的已发表数字更高，以更高的为准。只超过自己复现的 HybridGL 不算。标准是 `extreme_bar.py`。
- 当前结构达不到时，先用 Optuna 在留出集上搜至少 300 个完成的 trial，结果写入 `exp/out_final/optuna_study.json`。300 个仍有数据集不达标，才改模型，并重新搜 300 个。不要在官方验证集或测试集上搜。

## Domain Knowledge

RIS 已经比较挤。改进要打在还没被解决的点上：proposal 里没有正确掩码、按面积做解耦、固定九宫格处理不了序数和密集场景、一次只切一个目标、依赖外部 MLLM。不要再做一轮泛泛的 CLIP 全局特征融合。

## Non-Goals

- 不要把 MCCE-REC 当 baseline。它是指代表达理解，输出框，不是分割。
- 不要把 CaR 当 baseline。它的主任务是开放词汇分割。
- 不要改去做视频任务。视频在另一条 RVOS 调研里。
- 不要调用 Codex。审稿用 `mcp__llm-chat__review`，`executor_model` 填 `deepseek-flash`，审稿模型是 `glm-5.3`。
