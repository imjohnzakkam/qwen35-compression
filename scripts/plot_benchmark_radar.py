#!/usr/bin/env python3
"""Radar chart of the 4-bit variants' benchmark scores, each axis relative to BF16.

Scores are the full-suite results recorded in docs/experiments (02, 04, 06). Each axis is scaled
to its BF16 score: the dashed ring is BF16, the center is 85% of it, and gains reach past the ring.
WikiText-2 perplexity (lower is better) is listed under the chart instead.

    uv run --no-project --with matplotlib python scripts/plot_benchmark_radar.py \
        --output docs/figures/benchmark-radar.png --glaze-label "Glaze v2"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Clockwise from the top.
TASKS = (
    "MMLU-Pro",
    "GSM8K",
    "MATH-500",
    "IFEval",
    "HellaSwag",
    "ARC-Challenge",
    "MMBench",
    "MMMU",
    "MathVista",
    "OCRBench",
    "DocVQA",
    "TextVQA",
)
BF16 = (74.6, 83.2, 83.4, 82.3, 65.4, 51.1, 85.4, 69.6, 81.0, 86.3, 95.3, 82.8)
# name: (scores in TASKS order, WikiText-2 perplexity, checkpoint size, colour)
MODELS = {
    "GPTQ": (
        (71.4, 81.9, 75.8, 80.8, 64.3, 50.8, 84.1, 64.9, 78.0, 86.0, 94.8, 81.7),
        11.43,
        "3.78 GB",
        "#1f6fde",
    ),
    "AWQ": (
        (71.9, 82.6, 73.4, 79.3, 64.8, 50.0, 82.9, 65.7, 78.0, 87.1, 95.1, 81.8),
        11.55,
        "3.79 GB",
        "#ee5a4f",
    ),
    "AutoRound": (
        (72.8, 82.3, 73.2, 80.6, 64.8, 50.0, 84.7, 66.9, 80.3, 87.2, 95.3, 82.5),
        11.45,
        "3.80 GB",
        "#22a093",
    ),
    "Glaze": (
        (73.7, 82.8, 83.2, 82.8, 64.8, 49.6, 85.7, 66.7, 80.6, 86.3, 95.4, 82.2),
        11.30,
        "3.80 GB",
        "#e3a032",
    ),
}
CENTER = 0.85


def radius(score: float, reference: float) -> float:
    return (score / reference - CENTER) / (1 - CENTER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glaze-label", default="Glaze", help="Legend name for Glaze")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
    fig = plt.figure(figsize=(10, 10.4), dpi=200, facecolor="white")
    fig.text(
        0.05, 0.962, "Benchmark radar vs BF16", fontsize=22, fontweight="bold", color="#1b1b1b"
    )
    fig.text(
        0.05,
        0.937,
        "Qwen3.5-4B, instruct mode · each axis scaled to its BF16 score",
        fontsize=11.5,
        color="#6a6a6a",
    )
    ax = fig.add_axes((0.2, 0.3, 0.6, 0.5), projection="polar")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    angles = [2 * math.pi * i / len(TASKS) for i in range(len(TASKS))]
    closed = angles + angles[:1]

    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(color="#e2e4e7", linewidth=0.8)
    for share, label in ((0.90, "90%"), (0.95, "95%"), (1.0, "BF16")):
        r = radius(share, 1.0)
        style = (
            {"color": "#a9afb8", "linewidth": 1.3, "linestyle": "--"}
            if share == 1.0
            else {
                "color": "#dfe1e5",
                "linewidth": 0.9,
            }
        )
        ax.plot(closed, [r] * len(closed), zorder=1, **style)
        ax.text(0.04, r, label, fontsize=9, color="#8a8f98", va="center")

    for name, (scores, _, _, colour) in MODELS.items():
        values = [radius(s, b) for s, b in zip(scores, BF16, strict=True)]
        values += values[:1]
        is_glaze = name == "Glaze"
        ax.fill(closed, values, color=colour, alpha=0.06 if is_glaze else 0.04, zorder=2)
        ax.plot(
            closed,
            values,
            color=colour,
            linewidth=2.6 if is_glaze else 1.5,
            marker="o",
            markersize=4.5 if is_glaze else 3.5,
            zorder=4 if is_glaze else 3,
        )

    for angle, task, reference in zip(angles, TASKS, BF16, strict=True):
        ax.text(
            angle,
            1.33,
            f"{task}\nBF16 {reference:.1f}",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#363636",
            linespacing=1.5,
        )

    names = list(MODELS)
    for index, name in enumerate(names):
        _, ppl, size, colour = MODELS[name]
        x = 0.07 + index * 0.23
        column = 0.33 + index * 0.18
        label = args.glaze_label if name == "Glaze" else name
        fig.patches.append(
            matplotlib.patches.FancyBboxPatch(
                (x, 0.165),
                0.022,
                0.026,
                boxstyle="round,pad=0.002",
                transform=fig.transFigure,
                facecolor=colour,
                edgecolor="none",
            )
        )
        fig.text(
            x + 0.033, 0.178, label, fontsize=15, fontweight="bold", color="#1b1b1b", va="center"
        )
        fig.text(x + 0.033, 0.152, size, fontsize=11, color="#6a6a6a", va="center")
        fig.text(column, 0.078, label, fontsize=9.5, color="#686868", ha="center")
        fig.text(
            column, 0.055, f"{ppl:.2f}", fontsize=11, fontweight="bold", color=colour, ha="center"
        )
    fig.add_artist(
        matplotlib.lines.Line2D([0.05, 0.95], [0.115], color="#e2e4e7", transform=fig.transFigure)
    )
    fig.text(0.05, 0.078, "WikiText-2", fontsize=11, fontweight="bold", color="#363636")
    fig.text(0.05, 0.055, "perplexity, lower is better", fontsize=9, color="#6a6a6a")
    fig.text(
        0.5,
        0.018,
        "Center = 85% of BF16 · dashed ring = BF16 · points beyond the ring beat BF16",
        fontsize=9.5,
        color="#6a6a6a",
        ha="center",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
