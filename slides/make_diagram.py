#!/usr/bin/env python
"""Simplified method pipeline diagram for the advisor deck (matplotlib).

灰 = 冻结现成组件（无 RIS 训练）  紫 = 本工作提出  蓝 = 输入/输出
Outputs: figures/method_diagram.png (300dpi) + figures/method_diagram.pdf
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties, fontManager

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "figures")
os.makedirs(FIGD, exist_ok=True)

# ---- CJK font: extract the SC face from the Noto ttc collections ----
def resolve_fonts():
    reg = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    out = ["/tmp/NotoSansCJKsc-Regular.otf", "/tmp/NotoSansCJKsc-Bold.otf"]
    try:
        from fontTools.ttLib import TTCollection
        for src, dst in zip((reg, bold), out):
            coll = TTCollection(src)
            for f in coll.fonts:
                if f["name"].getDebugName(1) == "Noto Sans CJK SC":
                    f.save(dst)
                    break
        if all(os.path.exists(p) for p in out):
            return out
    except Exception as exc:  # noqa: BLE001
        print("fonttools path failed, falling back to ttc:", exc)
    return [reg, bold]

REG, BOLD = resolve_fonts()
fontManager.addfont(REG)
fontManager.addfont(BOLD)
FAM = FontProperties(fname=REG).get_name()
F = lambda size, bold=False: FontProperties(fname=BOLD if bold else REG, size=size)  # noqa: E731

# ---- palette ----
BLUE_F, BLUE_E = "#DBEAFE", "#2563EB"
PURP_F, PURP_E = "#EDE9FE", "#7C3AED"
GRAY_F, GRAY_E = "#F1F5F9", "#64748B"
INK = "#1E1E1E"
ARROW = "#334155"

fig, ax = plt.subplots(figsize=(12.0, 5.1), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.35,rounding_size=1.1",
                 linewidth=1.6, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontproperties=F(fs, bold), color=INK, zorder=3, linespacing=1.35)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=16, linewidth=1.7,
                 color=ARROW, zorder=1,
                 linestyle="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                ha="center", va="center", fontproperties=F(9.5, True),
                color=PURP_E, zorder=3)

YT, YB = 42, 12          # top-row / bottom-row y of box bottoms
H = 12                   # box height

# top row
box(0.5,  YT, 12, H, "图像", BLUE_F, BLUE_E, 12, True)
box(16.5, YT, 20, H, "SAM ViT-H 候选池\n（冻结，~50 masks）", GRAY_F, GRAY_E, 10)
box(40.5, YT, 22, H, "每候选特征\n几何 + box 对齐\n+ 冻结 CLIP 区域特征", PURP_F, PURP_E, 9.8)
box(66.5, YT, 18, H, "自研 learned selector\n（DECLARED TRAINED）", PURP_F, PURP_E, 10, True)
box(88.5, YT, 11, H, "输出\nmask / 空", BLUE_F, BLUE_E, 10.5)

# bottom row
box(0.5,  YB, 12, H, "表达式", BLUE_F, BLUE_E, 12, True)
box(16.5, YB, 26, H, "冻结 VLM：LLaVA-OneVision-7B\n全句 → bounding box", GRAY_F, GRAY_E, 10)
box(66.5, YB, 18, H, "集值扩展 + 三 prompt 弃权\n(gRefCOCO)", PURP_F, PURP_E, 10)

# arrows
arrow(12.5, YT + H/2, 16.5, YT + H/2)                 # image -> pool
arrow(36.5, YT + H/2, 40.5, YT + H/2)                 # pool -> features
arrow(62.5, YT + H/2, 66.5, YT + H/2)                 # features -> selector
arrow(84.5, YT + H/2, 88.5, YT + H/2)                 # selector -> output
arrow(12.5, YB + H/2, 16.5, YB + H/2)                 # expression -> VLM
arrow(42.5, YB + H, 51, YT, label="box", lx=2.5, ly=-1.5)   # VLM -> features
arrow(42.8, YB + H, 70, YT, dashed=True)                # VLM -> selector (dashed feed)
arrow(75.5, YT, 75.5, YB + H, label=None)             # selector -> set-valued (down)
arrow(84.5, YB + H/2, 92.5, YT, label=None)           # set-valued -> output (up-right)

# legend
ax.text(0.5, 58.6, "灰 = 冻结现成组件（无 RIS 训练）    紫 = 本工作提出的机制    蓝 = 输入 / 输出",
        ha="left", va="center", fontproperties=F(10.5, True), color=INK)
ax.text(0.5, 2.0, "box 是“选择”的锚点：不重新生成 mask，只在候选池里选出与 box 最贴合的那个实例",
        ha="left", va="center", fontproperties=F(9.5), color="#475569")

fig.tight_layout(pad=0.2)
fig.savefig(os.path.join(FIGD, "method_diagram.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(FIGD, "method_diagram.pdf"), facecolor="white")
print("wrote", os.path.join(FIGD, "method_diagram.png"))
print("family:", FAM)
