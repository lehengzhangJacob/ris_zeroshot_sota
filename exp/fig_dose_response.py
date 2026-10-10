#!/usr/bin/env python
"""Figure: the operator's benefit tracks how often expressions are spatial.

Left  — per-dataset effect vs spatial-expression frequency (the RefCOCO+ point
        was predicted before the run).
Right — per-relation-type deltas on RefCOCO, the selection split's target.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REC = {"order": ["wrong-instance", "under-segmentation", "boundary", "correct", "over-segmentation"]}


def main() -> int:
    cd = json.loads(Path("exp/out_final/cross_dataset.json").read_text())
    fin = json.loads(Path("exp/out_final/final_refcoco.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))

    names = ["refcoco", "refcocog", "refcoco+"]
    lbl = {"refcoco": "RefCOCO", "refcocog": "RefCOCOg", "refcoco+": "RefCOCO+"}
    col = {"refcoco": "#c0392b", "refcocog": "#7f8c8d", "refcoco+": "#2980b9"}
    for nm in names:
        v = cd[nm]
        ax1.scatter(v["spatial_frac"] * 100, v["delta_overall"] * 100, s=90,
                    color=col[nm], zorder=3, label=lbl[nm])
        ax1.annotate(f"{lbl[nm]}\n({v['n']:,} expr.)",
                     (v["spatial_frac"] * 100, v["delta_overall"] * 100),
                     textcoords="offset points", xytext=(9, 6), fontsize=8, color=col[nm])
        ax1.annotate("", xy=(v["spatial_frac"] * 100, v["delta_spatial"] * 100),
                     xytext=(v["spatial_frac"] * 100, v["delta_overall"] * 100),
                     arrowprops=dict(arrowstyle="->", color=col[nm], lw=1.1, alpha=0.75))
        ax1.scatter(v["spatial_frac"] * 100, v["delta_spatial"] * 100, s=34, marker="D",
                    facecolor="white", edgecolor=col[nm], zorder=3, linewidths=1.3)
    ax1.axhline(0, color="#444", lw=0.8)
    ax1.set_xlabel("expressions carrying a spatial relation  (%)", fontsize=9)
    ax1.set_ylabel("Δ mIoU over the released pipeline", fontsize=9)
    ax1.set_title("(a) benefit tracks spatial-reference frequency", fontsize=9)
    ax1.text(0.5, -0.36, "filled: all expressions · hollow diamond: spatial stratum only",
             transform=ax1.transAxes, fontsize=7.5, ha="center", color="#555")
    ax1.tick_params(labelsize=8)
    ax1.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    st = fin["all"]["rows"]["ours"]["strata"]
    sn = fin["all"]["rows"]["ours"]["strata_n"]
    keys = [k for k in ("horizontal", "vertical", "middle", "ordinal", "none") if k in st]
    vals = [st[k] * 100 for k in keys]
    cnts = [sn[k] for k in keys]
    colors = ["#c0392b" if k != "none" else "#95a5a6" for k in keys]
    axes_ = np.arange(len(keys))
    ax2.bar(axes_, vals, color=colors, width=0.62)
    for i, (v, c) in enumerate(zip(vals, cnts)):
        ax2.text(i, v + (0.6 if v >= 0 else -1.6), f"{v:+.1f}\nn={c}", ha="center",
                 fontsize=7.6, color="#333")
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.set_xticks(axes_)
    ax2.set_xticklabels([{"horizontal": "left/right", "vertical": "top/bottom",
                          "middle": "middle", "ordinal": "ordinal\n(abstains)",
                          "none": "no relation\n(abstains)"}[k] for k in keys], fontsize=8)
    ax2.set_ylabel("Δ mIoU over the released pipeline", fontsize=9)
    ax2.set_title("(b) RefCOCO, by relation type", fontsize=9)
    ax2.set_ylim(-4, max(vals) * 1.42)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = Path("exp/out_fig")
    fig.savefig(out / "dose_response.pdf", bbox_inches="tight")
    fig.savefig(out / "dose_response.png", dpi=200, bbox_inches="tight")
    print("saved", out / "dose_response.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
