from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weightbank.configs import CONFIGS, FULL, PINNED, PLAIN_DEPTHS, PLAIN_WIDTHS, ROUTED, SEEDS
from weightbank.train import Run, run_path

TEMPLATE = Path("report.in.md")
REPORT = Path("report.md")
CHART = Path("chart.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
WIDTH_LINE = "#52514e"
DEPTH_LINE = "#a3a29c"
ROUTED_COLORS = {"ff-routed": "#2a78d6", "attn-routed": "#eb6834", "both-routed": "#1baf7a"}
LABELS = {
    "ff-routed": "feed-forward routed",
    "attn-routed": "attention routed",
    "both-routed": "both routed",
}


@dataclass
class Summary:
    name: str
    stored: int
    losses: list[float]

    @property
    def mean(self) -> float:
        return mean(self.losses)

    @property
    def spread(self) -> float:
        return stdev(self.losses)


def load(name: str) -> list[Run]:
    return [Run(**json.loads(run_path(name, seed).read_text())) for seed in SEEDS]


def summarise(name: str) -> Summary:
    runs = load(name)
    (stored,) = {run.stored_weights for run in runs}
    return Summary(name, stored, [run.validation_loss for run in runs])


def interpolate(curve: Sequence[Summary], stored: int) -> float:
    for low, high in zip(curve, curve[1:]):
        if low.stored <= stored <= high.stored:
            share = (math.log(stored) - math.log(low.stored)) / (math.log(high.stored) - math.log(low.stored))
            return low.mean + share * (high.mean - low.mean)
    raise ValueError(f"{stored} stored weights lie outside the plain curve")


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def runs_table() -> str:
    rows = [
        [
            run.config,
            str(run.seed),
            f"{run.stored_weights:,}",
            f"{run.validation_loss:.4f}",
            f"{run.wall_seconds:.0f}",
        ]
        for name in CONFIGS
        for run in load(name)
    ]
    return table(["configuration", "seed", "stored weights", "validation loss", "wall seconds"], rows)


def summary_table() -> str:
    rows = [
        [
            summary.name,
            f"{summary.stored:,}",
            f"{summary.mean:.4f}",
            f"{summary.spread:.4f}",
            f"{min(summary.losses):.4f}",
            f"{max(summary.losses):.4f}",
        ]
        for summary in map(summarise, CONFIGS)
    ]
    return table(["configuration", "stored weights", "mean loss", "seed sd", "min", "max"], rows)


def gap_table(width_curve: Sequence[Summary], depth_curve: Sequence[Summary]) -> str:
    rows = []
    for name in ROUTED:
        summary = summarise(name)
        by_width = interpolate(width_curve, summary.stored)
        by_depth = interpolate(depth_curve, summary.stored)
        rows.append(
            [
                name,
                f"{summary.stored:,}",
                f"{summary.mean:.4f}",
                f"{summary.spread:.4f}",
                f"{by_width:.4f}",
                f"{summary.mean - by_width:+.4f}",
                f"{by_depth:.4f}",
                f"{summary.mean - by_depth:+.4f}",
            ]
        )
    return table(
        [
            "configuration",
            "stored weights",
            "mean loss",
            "seed sd",
            "narrower plain model at equal stored weights",
            "routed minus narrower",
            "shallower plain model at equal stored weights",
            "routed minus shallower",
        ],
        rows,
    )


def sanity_table() -> str:
    full = summarise(FULL)
    pinned = summarise(PINNED)
    rows = [
        [summary.name, f"{summary.stored:,}", *(f"{loss:.4f}" for loss in summary.losses), f"{summary.mean:.4f}", f"{summary.spread:.4f}"]
        for summary in (full, pinned)
    ]
    header = ["configuration", "stored weights", *(f"seed {seed}" for seed in SEEDS), "mean", "seed sd"]
    return table(header, rows) + f"\n\nDifference of means, pinned minus full: {pinned.mean - full.mean:+.4f} nats/byte."


def entropy(shares: Sequence[float]) -> float:
    return -sum(share * math.log(share) for share in shares if share > 0) / math.log(len(shares))


def picks_table(name: str) -> str:
    rows = []
    for kind, attribute in (("feed-forward", "feed_forward_picks"), ("attention", "attention_picks")):
        for run in load(name):
            counts: list[list[int]] = getattr(run, attribute)
            if not counts:
                continue
            varied = [1 - max(row) / sum(row) for row in counts]
            total = sum(map(sum, counts))
            usage = [sum(row[chunk] for row in counts) / total for chunk in range(len(counts[0]))]
            rows.append(
                [
                    kind,
                    str(run.seed),
                    " ".join(f"{share:.3f}" for share in varied),
                    " ".join(f"{share:.3f}" for share in usage),
                    f"{entropy(usage):.3f}",
                ]
            )
    return table(
        [
            "bank",
            "seed",
            "per module, share of tokens off the module's most common pick",
            "share of all picks per chunk",
            "usage entropy (1 = even)",
        ],
        rows,
    )


def chart(width_curve: Sequence[Summary], depth_curve: Sequence[Summary]) -> None:
    figure, axes = plt.subplots(figsize=(9, 5.6), dpi=160, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    for curve, color, label in (
        (width_curve, WIDTH_LINE, "plain, narrower (8 modules)"),
        (depth_curve, DEPTH_LINE, "plain, shallower (width 256)"),
    ):
        stored = [summary.stored / 1e6 for summary in curve]
        axes.plot(stored, [summary.mean for summary in curve], color=color, linewidth=2, marker="o", markersize=5, label=label, zorder=2)
        axes.fill_between(
            stored,
            [min(summary.losses) for summary in curve],
            [max(summary.losses) for summary in curve],
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
    for name in ROUTED:
        summary = summarise(name)
        color = ROUTED_COLORS[name]
        x = summary.stored / 1e6
        axes.errorbar(
            [x],
            [summary.mean],
            yerr=[[summary.mean - min(summary.losses)], [max(summary.losses) - summary.mean]],
            color=color,
            marker="D",
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            capsize=5,
            linewidth=2,
            linestyle="none",
            label=LABELS[name],
            zorder=4,
        )
        axes.annotate(LABELS[name], (x, summary.mean), xytext=(10, 8), textcoords="offset points", color=INK, fontsize=10)
    full = width_curve[-1]
    axes.annotate("full control", (full.stored / 1e6, full.mean), xytext=(-8, -16), textcoords="offset points", color=MUTED, fontsize=10, ha="right")
    axes.set_xlabel("stored weights (millions)", color=MUTED)
    axes.set_ylabel("validation loss (nats per byte)", color=MUTED)
    axes.set_title("Validation loss against stored weights, enwik8, 3 seeds (bars and bands: min to max)", color=INK, fontsize=11, loc="left")
    axes.grid(color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(CHART, facecolor=SURFACE)


def main() -> None:
    full = summarise(FULL)
    width_curve = [*(summarise(f"plain-w{width}") for width in PLAIN_WIDTHS), full]
    depth_curve = [*(summarise(f"plain-l{depth}") for depth in PLAIN_DEPTHS), full]
    chart(width_curve, depth_curve)
    sections = {
        "runs": runs_table(),
        "summary": summary_table(),
        "gaps": gap_table(width_curve, depth_curve),
        "sanity": sanity_table(),
        **{f"picks:{name}": picks_table(name) for name in ROUTED},
    }
    text = TEMPLATE.read_text()
    for key, section in sections.items():
        text = text.replace("{{" + key + "}}", section)
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
