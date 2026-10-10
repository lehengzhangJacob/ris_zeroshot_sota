#!/usr/bin/env python
"""Figure: error taxonomy of the oracle gap at two operating points."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["wrong-instance", "under-segmentation", "boundary", "correct", "over-segmentation"]
LABEL = {"wrong-instance": "wrong instance", "under-segmentation": "under-segmentation",
         "boundary": "boundary", "correct": "correct", "over-segmentation": "over-segmentation"}
COLOR = {"wrong-instance": "#c0392b", "under-segmentation": "#e67e22", "boundary": "#7f8c8d",
         "correct": "#27ae60", "over-segmentation": "#2980b9"}


def main() -> int:
    tax = json.loads(Path("exp/out_exp5/taxonomy_refcoco_guided.json").read_text())
    guided = tax["released_guided"]["by_class"]
    g2l = tax["argmax_g2l"]["by_class"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, data, title, n in ((axes[0], g2l, "(a) unguided argmax  (mIoU 0.352, oracle 0.822)",
                                tax["argmax_g2l"]["n"]),
                               (axes[1], guided, "(b) released HybridGL guidance  (mIoU 0.448, oracle 0.822)",
                                tax["released_guided"]["n"])):
        shares = [data[c]["gap_share"] for c in ORDER]
        prev = [data[c]["prevalence"] for c in ORDER]
        y = np.arange(len(ORDER))
        ax.barh(y - 0.18, shares, height=0.36, color=[COLOR[c] for c in ORDER], label="share of gap mass")
        ax.barh(y + 0.18, prev, height=0.36, color=[COLOR[c] for c in ORDER], alpha=0.42,
                hatch="//", edgecolor="white", label="prevalence")
        for i, (s, p) in enumerate(zip(shares, prev)):
            ax.text(s + 0.012, i - 0.18, f"{s:.3f}", va="center", fontsize=7.5)
            ax.text(p + 0.012, i + 0.18, f"{p:.3f}", va="center", fontsize=7.5, color="#555")
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("fraction", fontsize=8.5)
        ax.set_title(f"{title}\nn = {n:,} expressions", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    axes[0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = Path("exp/out_fig")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "taxonomy.pdf", bbox_inches="tight")
    fig.savefig(out / "taxonomy.png", dpi=200, bbox_inches="tight")
    print("saved", out / "taxonomy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
