from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from weightbank.compress.lm import LAYERS
from weightbank.compress.run import (
    BANKS,
    BITS,
    EXTRA_SEEDS,
    FIXED,
    LADDER,
    ROUTED_BANK,
    RUNS,
    STEPS,
    SWEEP_BANK,
    Arm,
    Full,
    Kind,
    chosen_rate,
    plan,
    read,
    trials,
)
from weightbank.report import GRID, INK, MUTED, SURFACE, table

TEMPLATE = Path("compress.in.md")
REPORT = Path("compress.md")
CHART = Path("compress-chart.png")
SIMILARITY = Path("compress-similarity.png")

COLOUR: dict[Kind, str] = {"bank": "#2a78d6", "low-rank": "#eb6834", "low-rank-kept": "#1baf7a", "routed": "#eda100"}
LABEL: dict[Kind, str] = {
    "bank": "shared bank",
    "low-rank": "low rank, every layer",
    "low-rank-kept": "low rank, same layers kept whole",
    "routed": "routed bank",
}
MARKER = {16: "o", 8: "s", 4: "^"}
BLUES = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")


def megabytes(count: int) -> str:
    return f"{count / 1e6:.1f}"


def figure(value: float) -> str:
    if value >= 1e6:
        return f"{value:.2e}"
    return f"{value:,.0f}" if value >= 1000 else f"{value:.2f}"


def off_diagonal(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [matrix[first][second] for first in range(len(matrix)) for second in range(first)]


def original_table(original: dict[str, Any]) -> str:
    return table(
        ["weights stored as", "stored MB", "perplexity"],
        [
            [
                "16-bit floats, as published" if bits == 16 else f"{bits}-bit",
                megabytes(original["bytes"][str(bits)]),
                figure(original["perplexity"][str(bits)]),
            ]
            for bits in BITS
        ],
    )


def arms() -> list[Arm]:
    return [arm for arm in plan() if arm.seed == 0]


def shared_layers(record: dict[str, Any]) -> int:
    assignment: list[int] = record["assignment"]
    return sum(assignment.count(chunk) > 1 for chunk in assignment)


def layout(arm: Arm, record: dict[str, Any]) -> str:
    if arm.kind in ("bank", "routed"):
        return f"{arm.chunks} matrices, rank {arm.rank}"
    ranks = sorted({rank for rank in record["ranks"] if rank is not None})
    return "rank " + (str(ranks[0]) if len(ranks) == 1 else f"{ranks[0]} to {ranks[-1]}")


def arms_table() -> str:
    rows = []
    for arm in arms():
        record = arm.read()
        rows.append(
            [
                f"{LABEL[arm.kind]}, {arm.chunks}/{arm.rank}",
                layout(arm, record),
                f"{record['weight_error']:.3f}" if "weight_error" in record else "",
                *(megabytes(record["bytes"][str(bits)]) for bits in BITS),
                figure(record["fit_perplexity"]),
                *(figure(record["healed_perplexity"][str(bits)]) for bits in BITS),
            ]
        )
    return table(
        [
            "arm, bank size/rank",
            "feed-forward weights",
            "weight error",
            "MB at 16 bits",
            "MB at 8",
            "MB at 4",
            "perplexity before healing",
            "healed",
            "healed, 8-bit",
            "healed, 4-bit",
        ],
        rows,
    )


def versus_table() -> str:
    rows = []
    for chunks, rank in BANKS:
        records = {kind: Arm(kind, chunks, rank).read() for kind in FIXED}
        healed = {kind: records[kind]["healed_perplexity"]["16"] for kind in FIXED}
        rows.append(
            [
                f"{chunks}/{rank}",
                f"{LAYERS - shared_layers(records['bank'])} of {LAYERS}",
                megabytes(records["bank"]["bytes"]["16"]),
                *(figure(healed[kind]) for kind in FIXED),
                f"{healed['bank'] / healed['low-rank-kept']:.2f}",
            ]
        )
    return table(
        [
            "bank size/rank",
            "layers stored whole",
            "MB",
            LABEL["bank"],
            LABEL["low-rank"],
            LABEL["low-rank-kept"],
            "bank / same-layers control",
        ],
        rows,
    )


def assignment_table() -> str:
    rows = []
    for chunks in sorted({chunks for chunks, _ in BANKS}):
        rank = next(rank for size, rank in BANKS if size == chunks)
        assignment: list[int] = Arm("bank", chunks, rank).read()["assignment"]
        groups = [[layer for layer in range(LAYERS) if assignment[layer] == chunk] for chunk in sorted(set(assignment))]
        shared = sorted((group for group in groups if len(group) > 1), key=lambda group: group[0])
        rows.append([str(chunks), "; ".join(", ".join(map(str, group)) for group in shared)])
    return table(["bank size", "layers that share a matrix (layers are numbered from 0)"], rows)


def unmatched_table() -> str:
    rows = []
    for chunks, rank in BANKS:
        record = Arm("bank", chunks, rank).read()
        rows.append([f"{chunks}/{rank}", f"{record['unmatched_weight_error']:.3f}", f"{record['weight_error']:.3f}"])
    return table(["bank size/rank", "weight error, units as published", "weight error, units matched"], rows)


def seeds_table() -> str:
    rows = []
    for kind in FIXED:
        values = [
            Arm(kind, *SWEEP_BANK, Full(seed)).read()["healed_perplexity"]["16"] for seed in (0, *EXTRA_SEEDS)
        ]
        rows.append([LABEL[kind], *(figure(value) for value in values), figure(fmean(values))])
    return table(["arm", *(f"seed {seed}" for seed in (0, *EXTRA_SEEDS)), "mean"], rows)


def sweep_table() -> str:
    rows = []
    for kind in FIXED:
        tried = trials(kind)
        cells = [figure(tried[rate].read()["validation_perplexity"]) if rate in tried else "" for rate in LADDER]
        best = LADDER.index(chosen_rate(kind))
        cells[best] = f"**{cells[best]}**"
        rows.append([LABEL[kind], *cells])
    return table(["arm", *(f"{rate:g}" for rate in LADDER)], rows)


def routed_table() -> str:
    rows = []
    kinds: tuple[Kind, ...] = ("bank", "routed", "low-rank-kept", "low-rank")
    for kind in kinds:
        record = Arm(kind, *ROUTED_BANK).read()
        moved = f"{100 * record['off_assignment']:.1f}%" if "off_assignment" in record else ""
        rows.append(
            [
                LABEL[kind],
                megabytes(record["bytes"]["16"]),
                figure(record["fit_perplexity"]),
                figure(record["healed_perplexity"]["16"]),
                moved,
            ]
        )
    return table(["arm", "MB", "perplexity before healing", "healed", "picks off the fixed assignment"], rows)


def similarity_table(similarity: dict[str, Any]) -> str:
    rows = []
    named = {
        "raw": "feed-forward step, hidden units as published",
        "matched": "feed-forward step, hidden units matched",
        "scrambled_matched": "the same matching on scrambled layers",
    }
    for key, name in named.items():
        values = off_diagonal(similarity[key])
        rows.append([name, f"{fmean(values):.4f}", f"{min(values):.4f}", f"{max(values):.4f}"])
    for key, matrix in similarity["attention_raw"].items():
        values = off_diagonal(matrix)
        rows.append([f"attention {key} matrix, as published", f"{fmean(values):.4f}", f"{min(values):.4f}", f"{max(values):.4f}"])
    return table(["weights compared", "mean cosine", "lowest", "highest"], rows)


def style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, which="major", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9, length=0)


def megabyte_axis(axes: Axes) -> None:
    axes.set_xscale("log")
    ticks = [40, 60, 80, 120, 160, 200, 270]
    axes.set_xticks(ticks, [str(tick) for tick in ticks])
    axes.set_xticks([], minor=True)
    axes.set_xlabel("stored megabytes, whole model", color=MUTED, fontsize=10)


def mark(axes: Axes, x: float, y: float, kind: Kind, bits: int, healed: bool) -> None:
    axes.plot(
        [x / 1e6],
        [y],
        marker=MARKER[bits],
        markersize=7.5,
        markerfacecolor=COLOUR[kind] if healed else SURFACE,
        markeredgecolor=SURFACE if healed else COLOUR[kind],
        markeredgewidth=1.0 if healed else 1.6,
        linestyle="none",
        zorder=3,
    )


def frontier(axes: Axes, original: dict[str, Any]) -> None:
    points = [(original["bytes"][str(bits)] / 1e6, original["perplexity"][str(bits)], bits) for bits in BITS]
    axes.plot([x for x, _, _ in points], [y for _, y, _ in points], color=INK, linewidth=2, zorder=2)
    for x, y, bits in points:
        axes.plot([x], [y], marker=MARKER[bits], markersize=9, color=INK, markeredgecolor=SURFACE, zorder=4)


def chart(original: dict[str, Any]) -> None:
    canvas, (whole, detail) = plt.subplots(1, 2, figsize=(14, 6.6), dpi=160, facecolor=SURFACE, width_ratios=(1, 1.3))
    for axes in (whole, detail):
        style(axes)
        megabyte_axis(axes)
        axes.set_yscale("log")
        frontier(axes, original)
    highest = 0.0
    for arm in arms():
        record = arm.read()
        size, before, after = record["bytes"]["16"], record["fit_perplexity"], record["healed_perplexity"]
        whole.plot([size / 1e6] * 2, [before, after["16"]], color=COLOUR[arm.kind], linewidth=1, alpha=0.6, zorder=1)
        mark(whole, size, before, arm.kind, 16, healed=False)
        mark(whole, size, after["16"], arm.kind, 16, healed=True)
        for bits in BITS:
            mark(detail, record["bytes"][str(bits)], after[str(bits)], arm.kind, bits, healed=True)
            highest = max(highest, after[str(bits)])
    whole.set_title("Before and after healing, 16-bit weights", color=INK, fontsize=11, loc="left")
    whole.set_ylabel("perplexity on WikiText-2 test, lower is better", color=MUTED, fontsize=10)
    detail.set_title("After healing, stored at 16, 8 and 4 bits", color=INK, fontsize=11, loc="left")
    detail.set_ylim(original["perplexity"]["16"] * 0.9, highest * 1.15)
    ticks = [tick for tick in (20, 30, 50, 100, 200, 500, 1000, 2000) if tick < highest * 1.15]
    detail.set_yticks(ticks, [str(tick) for tick in ticks])
    detail.set_yticks([], minor=True)
    for axes in (whole, detail):
        axes.annotate(
            "original model",
            (original["bytes"]["16"] / 1e6, original["perplexity"]["16"]),
            textcoords="offset points",
            xytext=(-8, 9),
            ha="right",
            color=INK,
            fontsize=9,
        )
    for bits in (8, 4):
        detail.annotate(
            f"original, {bits}-bit",
            (original["bytes"][str(bits)] / 1e6, original["perplexity"][str(bits)]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            color=INK,
            fontsize=9,
        )
    kinds = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=COLOUR[kind], label=LABEL[kind])
        for kind in COLOUR
    ]
    kinds.append(Line2D([], [], marker="o", markersize=8, color=INK, linewidth=2, label="original model, rounded to 8 and 4 bits, not healed"))
    states = [
        Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.6, label="before healing"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=MUTED, label="after healing"),
    ]
    widths = [
        Line2D([], [], marker=MARKER[bits], linestyle="none", markersize=8, color=MUTED, label=f"{bits}-bit")
        for bits in BITS
    ]
    canvas.legend(handles=kinds, loc="lower left", bbox_to_anchor=(0.04, 0.0), ncols=5, frameon=False, fontsize=9, labelcolor=INK)
    whole.legend(handles=states, loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    detail.legend(handles=widths, loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    canvas.suptitle(
        "Stored bytes against quality: a shared weight bank fitted to a trained model",
        color=INK,
        fontsize=13,
        x=0.04,
        ha="left",
    )
    canvas.tight_layout(rect=(0, 0.05, 1, 0.97))
    canvas.savefig(CHART, facecolor=SURFACE)
    plt.close(canvas)


def similarity_chart(similarity: dict[str, Any]) -> None:
    canvas, panels = plt.subplots(1, 3, figsize=(14, 4.8), dpi=160, facecolor=SURFACE, width_ratios=(1, 1, 1.1))
    colours = LinearSegmentedColormap.from_list("blues", BLUES)
    matched = np.asarray(similarity["matched"])
    top = float(np.max(matched - np.eye(LAYERS)))
    titles = {"raw": "Hidden units as published", "matched": "Hidden units matched between layers"}
    image = None
    for axes, (key, title) in zip(panels, titles.items()):
        values = np.asarray(similarity[key]) - np.eye(LAYERS)
        np.fill_diagonal(values, np.nan)
        image = axes.imshow(values, cmap=colours, vmin=0, vmax=top)
        axes.set_title(title, color=INK, fontsize=11, loc="left")
        axes.set_xlabel("layer", color=MUTED, fontsize=10)
        axes.set_ylabel("layer", color=MUTED, fontsize=10)
        axes.tick_params(colors=MUTED, labelsize=9, length=0)
        for spine in axes.spines.values():
            spine.set_visible(False)
    assert image is not None
    bar = canvas.colorbar(image, ax=list(panels[:2]), shrink=0.8, pad=0.02)
    bar.set_label("cosine between two layers' feed-forward weights", color=MUTED, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=9, length=0)
    bar.outline.set_visible(False)  # type: ignore[operator]
    energy = panels[2]
    style(energy)
    counts = list(range(1, LAYERS + 1))
    energy.plot(counts, [count / LAYERS for count in counts], color=MUTED, linewidth=2, linestyle=(0, (4, 3)))
    energy.plot(counts, similarity["shared_energy"], color=COLOUR["bank"], linewidth=2)
    energy.annotate("this model", (20, similarity["shared_energy"][19]), textcoords="offset points", xytext=(-10, 12), ha="right", color=INK, fontsize=9)
    energy.annotate("equal, unrelated layers", (20, 20 / LAYERS), textcoords="offset points", xytext=(10, -16), color=INK, fontsize=9)
    energy.set_title("Weight energy the best k shared matrices hold", color=INK, fontsize=11, loc="left")
    energy.set_xlabel("shared matrices, k", color=MUTED, fontsize=10)
    energy.set_ylabel("share of squared feed-forward weights", color=MUTED, fontsize=10)
    energy.set_xlim(1, LAYERS)
    energy.set_ylim(0, 1)
    canvas.savefig(SIMILARITY, facecolor=SURFACE, bbox_inches="tight")
    plt.close(canvas)


def main() -> None:
    original = read(RUNS / "original.json")
    similarity = read(RUNS / "similarity.json")
    tables = {
        "original": original_table(original),
        "similarity": similarity_table(similarity),
        "arms": arms_table(),
        "versus": versus_table(),
        "assignment": assignment_table(),
        "unmatched": unmatched_table(),
        "seeds": seeds_table(),
        "sweep": sweep_table(),
        "routed": routed_table(),
        "steps": f"{STEPS:,}",
        "sweep_bank": "{}/{}".format(*SWEEP_BANK),
    }
    text = TEMPLATE.read_text()
    for name, body in tables.items():
        text = text.replace("{" + name + "}", body)
    REPORT.write_text(text)
    chart(original)
    similarity_chart(similarity)


if __name__ == "__main__":
    main()
