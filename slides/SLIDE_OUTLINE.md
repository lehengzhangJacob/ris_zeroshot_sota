# 导师汇报 PPT 大纲（简单版，9 页，2026-10-17 复现框架降级后）

**工作**: Where Is It? A Dense Localization Prior for Training-Free Referring Image Segmentation
**类型**: 组会/导师汇报（非会议 talk）；无 speaker notes；中文
**输出**: slides/main.pdf + slides/presentation.pptx

| # | 页面 | 内容 | 图 |
|:-:|------|------|:--:|
| 1 | 标题 | 论文标题（中译）+ 一句话结果：全线 SOTA | - |
| 2 | 任务与现状 | propose-then-select 范式；published 旗舰 GLOD-RIS 58.42/54.01/57.85 | - |
| 3 | 诊断（核心） | 瓶颈是 selection 不是 mask 质量：oracle 82.21 vs released 44.82；wrong-instance 主导；SigLIP 更强反而更差 | - |
| 4 | **方法框图** | 简化流水线：SAM 候选池 → 每候选特征 → 自研 learned selector → 输出/弃权；冻结 VLM 全句出 box；gRefCOCO 集值扩展 | ★ |
| 5 | 创新点 1 | 全句 box 先验协议 +13.4/+10.6/+11.6；box 用作选择器；三 prompt 弃权；被杀掉的替代方案 | - |
| 6 | 创新点 2 | 组件来源审计：released 分数全部移出 head；自研 head 上位 70.57/66.35/65.51；review ACCEPT 9/10 | - |
| 7 | 结果 | 全线 SOTA 表（val 3×2 + test 5 列 vs published）+ gRefCOCO + 组件匹配验证 | 表 |
| 8 | 诚实边界 + 组件必要性 | 全监督 cIoU 差距量化；对比口径说明；backbone 实测（3.8B>7B、弱平台仍 SOTA、两个实测边界） | - |
| 9 | 总结 | 一句话 takeaway + 下一步 | - |

## 数据来源
全部来自 paper/sections/*.tex 与 exp/out_final/*.json（已通过 18 项验收门与跨家族 review）。

## 检查点记录
- 大纲由用户请求直接指定（10 页、必含框图/创新点/结果/边界四块），未另行停机确认。

## 2026-10-17 框架修正（operator）
- GLOD-RIS 复现短差不再作为「诊断/发现」呈现（无代码不可归因，不作评断）；降级为边界页的一句对比口径说明。
- 叙事驱动 = 我们自己的 oracle 分解诊断。
