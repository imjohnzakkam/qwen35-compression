#!/usr/bin/env python3
"""Radar charts of BF16, the 4-bit baselines, Glaze v2 and Blaze-Q, each against its own BF16.

Two panels, one per model, because each method has benchmark scores on one size only: Glaze v2 on
Qwen3.5-4B (docs/experiments/06) and Blaze-Q on Qwen3.5-2B (docs/experiments/08, G5). Five axes,
each scaled to that model's BF16 value: MMLU-Pro, MATH-500 and IFEval, plus two drift measures on
BF16's own 1,501 answers, token agreement (100 minus the top-1 flip rate) and per-token likelihood
of BF16's tokens (exp of minus the mean NLL). BF16 is the dashed ring; the center is 75% of it.

    uv run --no-project --with matplotlib python scripts/plot_blaze_radar.py \
        --output docs/figures/blaze-radar.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Clockwise from the top.
AXES = ("MMLU-Pro", "MATH-500", "IFEval", "Token agreement", "Token likelihood")
CENTER = 0.75
COLOURS = {
    "BF16": "#7d8590",
    "GPTQ W4A16": "#1f6fde",
    "AutoRound W4A16": "#22a093",
    "Glaze v2": "#e3a032",
    "Blaze-Q": "#d6336c",
}


def point(scores: tuple[float, float, float], flip_pct: float, nll: float) -> tuple[float, ...]:
    return (*scores, 100 - flip_pct, 100 * math.exp(-nll))


# Each panel: title, MMLU-Pro set, {model: (axis values, size)}; BF16 first.
# 4B: full MMLU-Pro, drift from records 03 and 06 (BF16 NLL 0.12797; loss = BF16 + excess).
# 2B: MMLU-Pro's 1001-question subset, drift from G5.
PANELS = (
    (
        "Qwen3.5-4B",
        "MMLU-Pro: full set",
        {
            "BF16": (point((74.6, 83.4, 82.3), 0.351, 0.12797), "9.32 GB"),
            "GPTQ W4A16": (point((71.4, 75.8, 80.8), 5.70, 0.12797 + 0.0541), "3.78 GB"),
            "AutoRound W4A16": (point((72.8, 73.2, 80.6), 4.96, 0.12797 + 0.0417), "3.80 GB"),
            "Glaze v2": (point((73.7, 83.2, 82.8), 3.29, 0.12797 + 0.0143), "3.80 GB"),
        },
    ),
    (
        "Qwen3.5-2B",
        "MMLU-Pro: 1001-question subset",
        {
            "BF16": (point((56.44, 64.6, 65.25), 0.377, 0.154706), "4.55 GB"),
            "GPTQ W4A16": (point((45.85, 52.4, 64.51), 6.775, 0.227677), "2.389 GB"),
            "Blaze-Q": (point((56.74, 67.8, 65.80), 0.712, 0.155168), "2.388 GB"),
        },
    ),
)


def radius(value: float, reference: float) -> float:
    return (value / reference - CENTER) / (1 - CENTER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Arial", "Helvetica", "DejaVu Sans"]
    fig = plt.figure(figsize=(16, 9.6), dpi=170, facecolor="white")
    fig.text(0.04, 0.955, "Glaze v2 and Blaze-Q against BF16", fontsize=22, fontweight="bold")
    fig.text(
        0.04,
        0.922,
        "Instruct mode · each axis scaled to that model's BF16 value · "
        "4-bit methods at one size per model",
        fontsize=12,
        color="#6a6a6a",
    )
    angles = [2 * math.pi * i / len(AXES) for i in range(len(AXES))]
    closed = angles + angles[:1]

    for index, (title, mmlu_note, models) in enumerate(PANELS):
        ax = fig.add_axes((0.07 + index * 0.48, 0.27, 0.38, 0.52), projection="polar")
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        ax.spines["polar"].set_visible(False)
        ax.grid(color="#e2e4e7", linewidth=0.8)
        for share, label in ((0.80, "80%"), (0.90, "90%")):
            r = radius(share, 1.0)
            ax.plot(closed, [r] * len(closed), color="#dfe1e5", linewidth=0.9, zorder=1)
            ax.text(math.pi / 5, r, label, fontsize=8.5, color="#8a8f98", va="center")
        reference = models["BF16"][0]
        for name, (values, _) in models.items():
            colour = COLOURS[name]
            points = [radius(v, b) for v, b in zip(values, reference, strict=True)]
            points += points[:1]
            if name == "BF16":
                ax.plot(closed, points, color=colour, linewidth=1.6, linestyle="--", zorder=5)
                continue
            is_new = name in ("Glaze v2", "Blaze-Q")
            ax.fill(closed, points, color=colour, alpha=0.08 if is_new else 0.03, zorder=2)
            ax.plot(
                closed,
                points,
                color=colour,
                linewidth=2.6 if is_new else 1.5,
                marker="o",
                markersize=4.5 if is_new else 3.5,
                zorder=4 if is_new else 3,
            )
        for angle, axis, value in zip(angles, AXES, reference, strict=True):
            ax.text(
                angle,
                1.55,
                f"{axis}\nBF16 {value:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#363636",
                linespacing=1.4,
            )
        x = 0.06 + index * 0.48
        fig.text(x, 0.855, title, fontsize=17, fontweight="bold", color="#1b1b1b")
        fig.text(x, 0.832, mmlu_note, fontsize=10, color="#6a6a6a")
        for row, (name, (_, size)) in enumerate(models.items()):
            y = 0.185 - row * 0.032
            colour = COLOURS[name]
            if name == "BF16":
                fig.add_artist(
                    matplotlib.lines.Line2D(
                        [x + 0.005, x + 0.027],
                        [y, y],
                        color=colour,
                        linewidth=1.8,
                        linestyle="--",
                        transform=fig.transFigure,
                    )
                )
            else:
                fig.patches.append(
                    matplotlib.patches.FancyBboxPatch(
                        (x + 0.008, y - 0.009),
                        0.016,
                        0.018,
                        boxstyle="round,pad=0.002",
                        transform=fig.transFigure,
                        facecolor=colour,
                        edgecolor="none",
                    )
                )
            fig.text(x + 0.037, y, name, fontsize=12.5, fontweight="bold", va="center")
            fig.text(x + 0.2, y, size, fontsize=11, color="#6a6a6a", va="center")
    fig.add_artist(
        matplotlib.lines.Line2D([0.04, 0.96], [0.052], color="#e2e4e7", transform=fig.transFigure)
    )
    fig.text(
        0.5,
        0.022,
        "Center = 75% of BF16 · dashed = BF16 · points beyond it beat BF16 · Blaze-Q is "
        "fake-quantized · Blaze on the 4B has drift scores only (see the table)",
        fontsize=9.5,
        color="#6a6a6a",
        ha="center",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor="white")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
