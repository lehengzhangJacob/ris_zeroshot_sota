# 数据泄露 / 作弊通道审计（2026-10-16，operator 质询"为什么这么好"）

审计对象：v3 shipped 结果（RefCOCO 族 70.57/66.35/65.51；gRefCOCO 52.71/75.08/86.14）。
方法：直接对源头 annotation 与拟合/评测记录做独立重算（不依赖 worker 自述）。

## 通道逐一结论

| 通道 | 检验 | 结果 |
|---|---|---|
| GT 泄漏进特征 | 自研 head 特征清单（`exp/exp159_head_rebuild.py::arm_c_features`）：our region-text / box overlap / center-in / area ratio / area / ranks / containment-parent。GT (`gt_iou`) 仅用于**拟合标签**（tune 行）与评测指标 | **干净**（无 GT 特征） |
| 图像级重叠（拟合 vs 评测） | RefCOCO 族：tune refs 全部来自 **train** split（1137/1120/765 refs），val 记录 10834/10758/4896；**图像交集 = 0**（三数据集）。gRefCOCO：拟合 800 refs / 626 图（全 train），val 14229 行 / 1501 图（全 val）；**图像交集 = 0**。全局 train vs val/test 亦为 0 | **干净**（零重叠，实测） |
| 超参/阈值选择 | 规则超参（k=15, γ=0.4906）来自 Optuna 的 train-carved held-out；gRefCOCO 阈值在 held-out 30% 两目标层上选；三 prompt 弃权/框提示研究均在 held-out（train refs） | **干净**（未在 val 上调参） |
| 评测口径 | RefCOCO 族 pooled oIoU + 官方 gRefCOCO 评估（protocol_audit.json 已审计）；test 列 published 数字为 Table I 转录+锚定校验（已声明） | 已声明 |
| 对手 published 口径 | GLOD-RIS published 数独立复现失败（−20~26，两套实现）；rule-swap 因果隔离；其 API 依赖规模失败 | 已声明（且为叙事主证据） |
| **唯一保留意见（模型选择）** | promoted head 的"选型"以 val parity 为准（component_provenance.json 全部 11 臂的 val 数字均在案）——属**选型**不是**拟合**（head 只在 train 拟合）；对冲证据：三条 val split 同时 ≥ 参考（+2.03/+1.68/+0.39）且五条 test 列（此后一次性评测）独立同向（+0.77~+2.03） | **披露于论文（"parity or better everywhere"即 promotion 标准）+ 工件全量在案** |
| VLM 预训练接触测试图 | LLaVA-OneVision 预训练含 COCO 图像（≈ 全部 benchmark 图的来源）；**无指代标注**；同类方法（GLOD-RIS 的 GLM-4.1V）同理 | **论文 limitations 已显式声明**（"weaker notion of zero-shot"，全类共性） |

## 为什么高是合理的（量级解释）
- 池 oracle 82.21 mIoU（RefCOCO val）：池中 82% 表达式已有好 mask；shipped 70.57 = oracle 的 **86% 兑现率**，training-free 版 68.53 = 83%。缺口（wrong-instance）正是 box 先验修的东西（box 贡献 +25.2/+26.8/+17.6）。
- 对 published 旗舰的 +12 mIoU 中，大部分是**对手数字不可复现**（实测 32.68 vs published 58.42）；对"实测对手"的差距是 +37.9。
- 未越界：全监督 gRefCOCO cIoU 68.94（InstAlign）远高于我们 52.71；RefCOCO 族现代全监督 ~73-76 mIoU 高于我们 70.57。

## 复现命令
- 图像重叠：见本审计的 python 片段（refs(unc).p / refs(umd).p / grefs(unc).json ↔ records_tune_*.jsonl / records_*.jsonl / records_grefcoco_14229.jsonl）。
- 特征清单：exp/exp159_head_rebuild.py:155-191。
