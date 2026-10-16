# 跨家族 Review 记录（glm-5.3，thread 396cc42b062b）

审查对象：SLIDE_OUTLINE.md + main.tex（10 页导师汇报 deck）。executor deepseek-flash / reviewer glm-5.3（跨家族）。

## 采纳并已修正
1. **44.82 归属歧义**（PPT 第 3/4 页矛盾点）：已在第 4 页明确为"所基于的 HybridGL released pipeline 44.82"，与第 3 页 GLOD-RIS 复现 32.68 区分。
2. **自研 VLM 未具名**：第 6 页改为"冻结 7B VLM（LLaVA-OneVision-7B）"。
3. **"免训练 vs declared trained"口径**：第 7 页明确"免训练版 68.53/64.67/65.12 保留为 zero-shot 参考行；两者均全线超 published"，先发制人回答"这还算 zero-shot 吗"。
4. **rule-swap 缺数字**：第 3 页补上"32.9 vs 44.8（同池同特征）"。
5. **过宽表述**："全领域都在修后者" → "现有工作多聚焦 mask 质量与合并"。
6. **跨赛道比较防杠**：gIoU > InstAlign 标注"参考行"；sl9 已有 cIoU 差距量化页兜底。
7. **API 崩溃表述**：改为"其 MLLM API 输出在高负载下解析失败（运行至第 223 条即中止）"。

## 记录但不采纳
- reviewer 建议把"组件来源审计"降为佐证（采纳：已改为标题副注"自研选择头上位（组件来源审计）"）。
- 图 5 加 annotation callout：框图底部caption已承担该作用，保持简洁。
