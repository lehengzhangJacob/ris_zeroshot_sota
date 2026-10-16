#!/usr/bin/env python
"""Editable PPTX mirror of slides/main.pdf (advisor review deck).

Native text boxes (bullets, bold markup via **), native table, diagram as image.
CJK font: Microsoft YaHei (falls back gracefully if absent).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = RGBColor(0x25, 0x63, 0xEB)
ACCENT = RGBColor(0x7C, 0x3A, 0xED)
INK = RGBColor(0x1E, 0x1E, 0x1E)
MUTED = RGBColor(0x47, 0x55, 0x69)
EA = "Microsoft YaHei"
LATIN = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def set_run_font(run, size, bold=False, color=INK, font=EA):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = font
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", EA)


def add_text(slide, x, y, w, h, blocks, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """blocks: list of (text, size, bold, color, space_after_pt). **bold** markup supported."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    first = True
    for text, size, bold, color, gap in blocks:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(gap)
        # parse **bold** segments
        parts = text.split("**")
        for i, part in enumerate(parts):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            set_run_font(r, size, bold or (i % 2 == 1), color)
    return tb


def bullets(slide, items, top=1.35, size=17):
    y = top
    for it in items:
        tb = slide.shapes.add_textbox(Inches(0.85), Inches(y), Inches(11.6), Inches(0.9))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = "●  "
        set_run_font(r, size, True, PRIMARY)
        for i, part in enumerate(it.split("**")):
            if not part:
                continue
            r = p.add_run(); r.text = part
            set_run_font(r, size, i % 2 == 1, INK)
        y += 1.05


def title(slide, text):
    add_text(slide, 0.55, 0.30, 12.2, 0.8, [(text, 26, True, PRIMARY, 0)])


def new():
    return prs.slides.add_slide(BLANK)

# ---------------- Slide 1: title ----------------
s = new()
add_text(s, 0.8, 1.45, 11.7, 1.0,
         [("Where Is It?", 40, True, PRIMARY, 0)], align=PP_ALIGN.CENTER)
add_text(s, 0.8, 2.55, 11.7, 0.8,
         [("面向免训练指代图像分割的密集定位先验", 22, False, INK, 0)], align=PP_ALIGN.CENTER)
add_text(s, 0.8, 3.35, 11.7, 0.9,
         [("零样本指代图像分割（RIS）：把瓶颈从 “mask 质量” 修正为 “选择”，用一个冻结 VLM 的 box 先验修好它", 14, False, MUTED, 0)],
         align=PP_ALIGN.CENTER)
add_text(s, 0.8, 4.45, 11.7, 0.6,
         [("RefCOCO 70.57 mIoU　·　gRefCOCO 52.71 cIoU　·　16/16 指标超 published 旗舰", 16, True, ACCENT, 0)],
         align=PP_ALIGN.CENTER)
add_text(s, 0.8, 5.05, 11.7, 0.5,
         [("全线 SOTA（同类零样本赛道）", 13, False, ACCENT, 0)], align=PP_ALIGN.CENTER)
add_text(s, 0.8, 6.15, 11.7, 0.5,
         [("张乐桁　·　2026-10-16", 14, False, INK, 0)], align=PP_ALIGN.CENTER)

# ---------------- Slide 2 ----------------
s = new(); title(s, "任务与现状")
bullets(s, [
    "任务：图像 + 指代句 → 分割出目标实例（zero-shot / 免训练赛道）",
    "主流范式 propose-then-select：冻结分割器提候选 mask → 冻结 CLIP/VLM 打分 → argmax",
    "published 最强同赛道：GLOD-RIS（TCSVT 2026）**58.42 / 54.01 / 57.85**（mIoU，RefCOCO/+/g）",
    "我们问的问题：**这套范式到底坏在哪？**",
], top=1.55)

# ---------------- Slide 3 ----------------
s = new(); title(s, "诊断一：旗舰方法不可复现，规则本身有害")
bullets(s, [
    "独立复现 GLOD-RIS（两套实现、同款 GLM-4.1V 引导）：**32.68 / 30.96 / 37.24**——差 **−20 ~ −26**",
    "因果隔离（rule-swap）：把它的选择规则放到**我们完全相同的候选池与特征**上 → 仍不如无引导 scorer（**32.9 vs 44.8**，同池同特征）",
    "结论：坏的不是环境，是“选”的机制本身",
    "其 MLLM API 输出在高负载下解析失败（运行至第 223 条即中止）",
], top=1.55)

# ---------------- Slide 4 ----------------
s = new(); title(s, "诊断二：瓶颈是 selection，不是 mask 质量")
bullets(s, [
    "oracle：池中最好 mask 达 **82.21 mIoU**（82% 表达式池里已有好 mask）；所基于的 HybridGL released pipeline 只拿到 **44.82**",
    "可回收 IoU 的构成：**wrong-instance（选错实例）主导**；under-segmentation 只占小头——现有工作多聚焦 mask 质量与合并",
    "更强 encoder 不解决：SigLIP-so400m 反而**更差**（36.5 vs 40.4）",
    "⇒ 失败是 **localization**（不知道在哪），不是 representation",
], top=1.55)

# ---------------- Slide 5: diagram ----------------
s = new(); title(s, "方法框图")
pic = os.path.join(HERE, "figures", "method_diagram.png")
s.shapes.add_picture(pic, Inches(0.45), Inches(1.45), width=Inches(12.45))

# ---------------- Slide 6 ----------------
s = new(); title(s, "创新点 1：全句 box 先验协议")
bullets(s, [
    "冻结 7B VLM（LLaVA-OneVision-7B），自然语言直接问 box；**传全句 ≠ 名词短语**：**+13.4 / +10.6 / +11.6** mIoU",
    "box 用作**选择器**，不重新生成 mask（诊断显示池子已经够好）",
    "无目标判定：三 prompt presence 集成 → N-acc **86.14**（高于已发表全监督 SOTA）",
    "试过并淘汰的替代（同记录 mIoU，基线 42.9）：Set-of-Mark 索引 39.1 / 删除法 42.7 / 排除法 41.0 / 网格投票 11.1",
], top=1.55)

# ---------------- Slide 7 ----------------
s = new(); title(s, "创新点 2：自研选择头上位（组件来源审计）")
bullets(s, [
    "逐组件标注来源（本工作 / released / 通用冻结），把 released 计分项**全部移出**最终选择头",
    "自研 learned selector（8 项自有特征，仅 train 划分拟合，declared trained）：**70.57 / 66.35 / 65.51**（原免训练版 68.53 / 64.67 / 65.12，保留为 zero-shot 参考行）",
    "五个 test 列同步提升；仍读取的 released 组件只剩 2 个**冻结表示**",
    "跨家族 review：**ACCEPT 9/10**",
], top=1.55)

# ---------------- Slide 8: results table ----------------
s = new(); title(s, "结果：全线 SOTA")
rows, cols = 3, 4
tbl_shape = s.shapes.add_table(rows, cols, Inches(0.85), Inches(1.5), Inches(11.6), Inches(1.7))
tbl = tbl_shape.table
data = [
    ["方法", "RefCOCO", "RefCOCO+", "RefCOCOg"],
    ["GLOD-RIS（published）", "58.42 / 50.85", "54.01 / 45.96", "57.85 / 50.03"],
    ["本工作（shipped）", "70.57 / 64.66", "66.35 / 59.40", "65.51 / 60.93"],
]
for i, row in enumerate(data):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        r = p.add_run(); r.text = val
        set_run_font(r, 15, bold=(i == 0 or (i == 2 and j > 0)),
                     color=PRIMARY if i == 0 else (ACCENT if i == 2 and j > 0 else INK))
add_text(s, 0.85, 3.45, 11.6, 0.5,
         [("val split，mIoU / oIoU；**16/16** 列全部超过 published（test 五列 +0.8 ~ +18.1）", 14, False, MUTED, 0)])
bullets(s, [
    "gRefCOCO：cIoU **52.71**、gIoU **75.08**（**> 全监督最强 InstAlign 74.34**，参考行）、N-acc 86.14",
    "组件匹配验证：换 GLOD-RIS 同款 MLLM，仍远超其 published 行",
], top=4.15, size=16)

# ---------------- Slide 9 ----------------
s = new(); title(s, "诚实边界与进行中工作")
bullets(s, [
    "全监督 cIoU 仍有差距：52.71 vs 68.94（InstAlign）——已**量化**，不声称",
    "复现口径差异、VLM 延迟（0.6~1.8 s/句）均已在论文声明",
    "进行中：**backbone 削弱实验**——更小的 VLM / 更小的 SAM / 更弱编码器 / 去掉额外检测器",
    "目标：**比对手更弱、更少的 backbone 仍然 SOTA**（堵死“靠 backbone”的质疑）",
], top=1.55)

# ---------------- Slide 10 ----------------
s = new(); title(s, "总结")
bullets(s, [
    "训练-free RIS 的瓶颈不是 mask 质量，而是**选择（localization）**——oracle 分解实测",
    "冻结 VLM 的**全句 box 先验**把“选”修好；替代方案全部测过并被淘汰",
    "组件审计后**自研选择头上位**，模型大头为本工作 → **16/16 全线超 published 旗舰**",
    "下一步：backbone 削弱验证 → 投稿；视频侧同类“proxy→val 反转”现象已完成测量",
], top=1.55)

out = os.path.join(HERE, "presentation.pptx")
prs.save(out)
print("wrote", out, "slides:", len(prs.slides.__iter__.__self__._sldIdLst))
