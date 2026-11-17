from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weightbank.configs import CONFIGS, FULL, MODULES, NARROWER, NO_ROUTER, OWN_CHUNK, ROUTED, SEEDS, SHALLOWER, WIDTH
from weightbank.sweep import ARMS, FIRST_RATE, FIRST_STEPS, LONG_STEPS, SWEEP_SEED, SWEEP_STEPS, rate_sweep
from weightbank.train import LOG_WINDOWS, RUNS, PickStats, Plan, Run

TEMPLATE = Path("report.in.md")
REPORT = Path("report.md")
CHART = Path("chart.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NARROWER_LINE = "#52514e"
SHALLOWER_LINE = "#a3a29c"
LABELS = {"ff-routed": "feed-forward routed", "attn-routed": "attention routed", "both-routed": "both routed"}
COLORS = {"ff-routed": "#2a78d6", "attn-routed": "#eb6834", "both-routed": "#1baf7a"}
assert LABELS.keys() == COLORS.keys() == ROUTED.keys()


@dataclass
class Summary:
    name: str
    runs: list[Run]

    @property
    def stored(self) -> int:
        (stored,) = {run.stored_weights for run in self.runs}
        return stored

    @property
    def rate(self) -> float:
        (rate,) = {run.peak_rate for run in self.runs}
        return rate

    @property
    def losses(self) -> list[float]:
        return [run.validation_loss for run in self.runs]

    @property
    def loss(self) -> float:
        return fmean(self.losses)

    @property
    def spread(self) -> float:
        return stdev(self.losses)


def interpolate(curve: Sequence[Summary], stored: int) -> float:
    for low, high in zip(curve, curve[1:]):
        if low.stored <= stored <= high.stored:
            share = (math.log(stored) - math.log(low.stored)) / (math.log(high.stored) - math.log(low.stored))
            return low.loss + share * (high.loss - low.loss)
    raise ValueError(f"{stored} stored weights lie outside the plain curve")


@dataclass
class Budget:
    steps: int
    summaries: dict[str, Summary]

    @property
    def narrower(self) -> list[Summary]:
        return [*(self.summaries[name] for name in NARROWER), self.summaries[FULL]]

    @property
    def shallower(self) -> list[Summary]:
        return [*(self.summaries[name] for name in SHALLOWER), self.summaries[FULL]]

    def gap(self, name: str) -> float:
        summary = self.summaries[name]
        plain = min(interpolate(curve, summary.stored) for curve in (self.narrower, self.shallower))
        return summary.loss - plain


def budget(steps: int, rates: dict[str, float], seeds: Sequence[int]) -> Budget:
    summaries = {}
    for name, rate in rates.items():
        runs = [Plan(name, seed, steps, rate).read() for seed in seeds]
        if not all(math.isfinite(run.validation_loss) for run in runs):
            raise ValueError(f"{name}: a run ended with a loss that is not finite")
        summaries[name] = Summary(name, runs)
    return Budget(steps, summaries)


def swept(name: str) -> dict[float, float]:
    _, losses = rate_sweep(name)
    if losses is None:
        raise ValueError(f"{name}: the learning-rate sweep is not finished")
    return losses


def chosen(name: str) -> float:
    losses = swept(name)
    return min(losses, key=losses.__getitem__)


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def runs_table() -> str:
    runs = sorted(
        (Run.read(path) for path in RUNS.glob("*/*.json")),
        key=lambda run: (run.steps, list(CONFIGS).index(run.config), run.peak_rate, run.seed),
    )
    rows = [
        [
            run.config,
            str(run.seed),
            f"{run.steps:,}",
            f"{run.peak_rate:g}",
            f"{run.stored_weights:,}",
            f"{run.validation_loss:.4f}",
            f"{run.training_seconds:.0f}",
        ]
        for run in runs
    ]
    return table(
        ["configuration", "seed", "steps", "learning rate", "stored weights", "final validation loss", "wall seconds (training)"],
        rows,
    )


def summary_table(summaries: dict[str, Summary]) -> str:
    rows = [
        [
            summary.name,
            f"{summary.rate:g}",
            f"{summary.stored:,}",
            f"{summary.loss:.4f}",
            f"{summary.spread:.4f}",
            f"{min(summary.losses):.4f}",
            f"{max(summary.losses):.4f}",
        ]
        for summary in summaries.values()
    ]
    return table(["configuration", "learning rate", "stored weights", "mean loss", "seed sd", "lowest", "highest"], rows)


def sweep_table() -> str:
    losses = {name: swept(name) for name in ARMS}
    rates = sorted({rate for arm in losses.values() for rate in arm})
    rows = []
    for name, arm in losses.items():
        best = chosen(name)
        cells = ["" if rate not in arm else f"**{arm[rate]:.4f}**" if rate == best else f"{arm[rate]:.4f}" for rate in rates]
        rows.append([name, *cells, f"{best:g}"])
    return table(["configuration", *(f"{rate:g}" for rate in rates), "chosen"], rows)


def both_budgets_table(first: Budget, tuned: Budget, long: Budget) -> str:
    rows = []
    for name in (*ROUTED, NO_ROUTER):
        rows.append(
            [
                name,
                f"{long.summaries[name].stored:,}",
                f"{first.summaries[name].rate:g}",
                f"{first.gap(name):+.4f}",
                f"{long.summaries[name].rate:g}",
                f"{tuned.gap(name):+.4f}",
                f"{long.gap(name):+.4f}",
                f"{long.gap(name) - first.gap(name):+.4f}",
            ]
        )
    return table(
        [
            "configuration",
            "stored weights",
            f"learning rate, {first.steps:,} steps",
            f"gap, {first.steps:,} steps (3 seeds)",
            "chosen learning rate",
            f"gap, {tuned.steps:,} steps at the chosen rates (seed {SWEEP_SEED} only)",
            f"gap, {long.steps:,} steps at the chosen rates (3 seeds)",
            "long minus first",
        ],
        rows,
    )


def plain_table(first: Budget, long: Budget) -> str:
    rows = [
        [
            name,
            f"{long.summaries[name].stored:,}",
            f"{first.summaries[name].loss:.4f}",
            f"{first.summaries[name].spread:.4f}",
            f"{long.summaries[name].rate:g}",
            f"{long.summaries[name].loss:.4f}",
            f"{long.summaries[name].spread:.4f}",
            f"{long.summaries[name].loss - first.summaries[name].loss:+.4f}",
        ]
        for name in (*NARROWER, *SHALLOWER, FULL)
    ]
    return table(
        [
            "configuration",
            "stored weights",
            f"loss, {first.steps:,} steps at {FIRST_RATE:g}",
            "seed sd",
            "chosen learning rate",
            f"loss, {long.steps:,} steps",
            "seed sd",
            "long minus first",
        ],
        rows,
    )


def gap_table(budget: Budget) -> str:
    rows = []
    for name in (*ROUTED, NO_ROUTER):
        summary = budget.summaries[name]
        by_width = interpolate(budget.narrower, summary.stored)
        by_depth = interpolate(budget.shallower, summary.stored)
        rows.append(
            [
                name,
                f"{summary.rate:g}",
                f"{summary.stored:,}",
                f"{summary.loss:.4f}",
                f"{summary.spread:.4f}",
                f"{by_width:.4f}",
                f"{summary.loss - by_width:+.4f}",
                f"{by_depth:.4f}",
                f"{summary.loss - by_depth:+.4f}",
                f"{budget.gap(name):+.4f}",
            ]
        )
    return table(
        [
            "configuration",
            "learning rate",
            "stored weights",
            "mean loss",
            "seed sd",
            "narrower plain at equal stored weights",
            "minus narrower",
            "shallower plain at equal stored weights",
            "minus shallower",
            "minus the better plain",
        ],
        rows,
    )


def paired_table(summaries: dict[str, Summary], name: str, against: str) -> str:
    ours, theirs = summaries[name], summaries[against]
    differences = [a - b for a, b in zip(ours.losses, theirs.losses)]
    rows = [
        [summary.name, f"{summary.stored:,}", *(f"{loss:.4f}" for loss in summary.losses), f"{summary.loss:.4f}", f"{summary.spread:.4f}"]
        for summary in (theirs, ours)
    ]
    rows.append([f"{name} minus {against}", "", *(f"{d:+.4f}" for d in differences), f"{fmean(differences):+.4f}", ""])
    return table(["configuration", "stored weights", *(f"seed {seed}" for seed in SEEDS), "mean", "seed sd"], rows)


def logged(run: Run, step: int) -> float:
    (loss,) = [loss for at, loss in run.curve if at == step]
    return loss


def during_training_table(budget: Budget) -> str:
    summaries = budget.summaries
    steps = [budget.steps * part // 8 for part in range(1, 9)]
    rows = [
        [
            summary.name,
            *(f"{fmean(logged(run, step) for run in summary.runs):.4f}" for step in steps),
        ]
        for summary in summaries.values()
    ]
    return table(["configuration", *(f"step {step:,}" for step in steps)], rows)


def sampled_table(summaries: dict[str, Summary]) -> str:
    rows = [
        [
            name,
            f"{summaries[name].loss:.4f}",
            f"{fmean(run.sampled_picks_validation_loss for run in summaries[name].runs):.4f}",
        ]
        for name in ROUTED
    ]
    return table(["configuration", "highest-scoring picks (reported)", "picks sampled as in training"], rows)


def entropy(shares: Sequence[float]) -> float:
    return -sum(share * math.log(share) for share in shares if share > 0) / math.log(len(shares))


def shares(values: Sequence[float]) -> str:
    return " ".join(f"{value:.2f}" for value in values)


def off_common(stats: PickStats) -> list[float]:
    return [1 - max(row) / sum(row) for row in stats.counts]


def pick_variation_table(first: Budget, long: Budget) -> str:
    rows = []
    for name in ROUTED:
        for kind in ("feed_forward", "attention"):
            cells = []
            for budget in (first, long):
                stats: list[PickStats] = [getattr(run, kind) for run in budget.summaries[name].runs]
                if None in stats:
                    break
                cells += [
                    f"{fmean(share for run in stats for share in off_common(run)):.2f}",
                    f"{min(share for run in stats for share in off_common(run)):.2f}",
                    f"{fmean(value for run in stats for value in run.confidence):.2f}",
                ]
            if cells:
                rows.append([name, kind.replace("_", "-"), *cells])
    header = ["configuration", "bank"]
    for budget in (first, long):
        header += [
            f"off the common pick, mean, {budget.steps:,} steps",
            f"lowest module, {budget.steps:,} steps",
            f"router probability, {budget.steps:,} steps",
        ]
    return table(header, rows)


def picks_table(summary: Summary) -> str:
    rows = []
    for kind in ("feed_forward", "attention"):
        for run in summary.runs:
            stats: PickStats | None = getattr(run, kind)
            if stats is None:
                continue
            total = sum(map(sum, stats.counts))
            usage = [sum(row[chunk] for row in stats.counts) / total for chunk in range(len(stats.counts[0]))]
            rows.append(
                [
                    kind.replace("_", "-"),
                    str(run.seed),
                    shares(off_common(stats)),
                    shares(stats.confidence),
                    shares(usage),
                    f"{entropy(usage):.2f}",
                ]
            )
    return table(
        [
            "bank",
            "seed",
            "share of tokens off the module's most common pick, per module",
            "router's mean probability for its pick, per module",
            "share of all picks, per chunk",
            "usage evenness",
        ],
        rows,
    )


def chart(budget: Budget) -> None:
    summaries, narrower, shallower = budget.summaries, budget.narrower, budget.shallower
    figure, axes = plt.subplots(figsize=(9, 5.6), dpi=160, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.set_xscale("log")
    for curve, color, label in (
        (narrower, NARROWER_LINE, f"plain, narrower ({MODULES} modules)"),
        (shallower, SHALLOWER_LINE, f"plain, fewer modules (width {WIDTH})"),
    ):
        stored = [summary.stored / 1e6 for summary in curve]
        axes.plot(stored, [summary.loss for summary in curve], color=color, linewidth=2, marker="o", markersize=5, label=label, zorder=2)
        axes.fill_between(
            stored,
            [min(summary.losses) for summary in curve],
            [max(summary.losses) for summary in curve],
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
    points = [(name, COLORS[name], COLORS[name], LABELS[name], "D") for name in ROUTED]
    points.append((NO_ROUTER, INK, SURFACE, "both shared, no router", "s"))
    label_offsets = {"ff-routed": (0, -26), "attn-routed": (0, 14), NO_ROUTER: (-12, -4)}
    for name, color, fill, label, marker in points:
        summary = summaries[name]
        x = summary.stored / 1e6
        axes.errorbar(
            [x],
            [summary.loss],
            yerr=[[summary.loss - min(summary.losses)], [max(summary.losses) - summary.loss]],
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor=fill,
            markeredgecolor=SURFACE if fill == color else color,
            markeredgewidth=2,
            capsize=6,
            linewidth=2,
            linestyle="none",
            zorder=4,
        )
        offset = label_offsets.get(name, (10, 8))
        side = "right" if offset[0] < 0 else "center" if offset[0] == 0 else "left"
        axes.annotate(label, (x, summary.loss), xytext=offset, textcoords="offset points", color=INK, fontsize=10, ha=side)
    full = summaries[FULL]
    axes.annotate("full control", (full.stored / 1e6, full.loss), xytext=(-8, -16), textcoords="offset points", color=MUTED, fontsize=10, ha="right")
    ticks = [2, 3, 4, 5, 6]
    axes.set_xticks(ticks, [str(tick) for tick in ticks])
    axes.set_xticks([], minor=True)
    axes.set_xlim(1.5, 7.2)
    axes.set_xlabel("stored weights (millions, log scale)", color=MUTED)
    axes.set_ylabel("validation loss (nats per byte)", color=MUTED)
    axes.set_title(
        f"Validation loss against stored weights, enwik8, {budget.steps:,} steps, {len(SEEDS)} seeds (bars and bands: lowest to highest)",
        color=INK,
        fontsize=11,
        loc="left",
    )
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
    first = budget(FIRST_STEPS, {name: FIRST_RATE for name in CONFIGS}, SEEDS)
    rates = {name: chosen(name) for name in ARMS}
    tuned = budget(SWEEP_STEPS, rates, (SWEEP_SEED,))
    long = budget(LONG_STEPS, rates, SEEDS)
    chart(long)
    sections = {
        "both-budgets": both_budgets_table(first, tuned, long),
        "plain": plain_table(first, long),
        "sweep": sweep_table(),
        "gaps-long": gap_table(long),
        "gaps-first": gap_table(first),
        "sanity": paired_table(first.summaries, OWN_CHUNK, FULL),
        "no-router-long": paired_table(long.summaries, "both-routed", NO_ROUTER),
        "no-router-first": paired_table(first.summaries, "both-routed", NO_ROUTER),
        "during-training": during_training_table(long),
        "sampled": sampled_table(long.summaries),
        "log-windows": f"{LOG_WINDOWS:,}",
        "pick-variation": pick_variation_table(first, long),
        "runs": runs_table(),
        "summary-long": summary_table(long.summaries),
        "summary-first": summary_table(first.summaries),
        **{f"picks:{name}": picks_table(long.summaries[name]) for name in ROUTED},
    }
    text = TEMPLATE.read_text()
    for key, section in sections.items():
        text = text.replace("{{" + key + "}}", section)
    if "{{" in text:
        raise ValueError(f"{TEMPLATE} names a section this report does not write")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
