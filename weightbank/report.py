from __future__ import annotations

import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import ScaledTranslation

from weightbank import memory
from weightbank.configs import (
    ARCHITECTURE,
    BOTH_ROUTED,
    DRAW,
    FIXED_MAP,
    FULL,
    KEYS,
    MODULES,
    NARROWER,
    NO_ROUTER,
    NOISE_FREE,
    OWN_CHUNK,
    PICK,
    ROUNDS_ONE_AND_TWO,
    ROUTED,
    ROUTER_VARIANTS,
    SHALLOWER,
    WIDTH,
    model_config,
    named,
    other_map,
    plain_words,
    rate_source,
)
from weightbank.ledger import ABANDONED, SEEDS, STEPS, Cell, Ledger, Row
from weightbank.model import Fixed
from weightbank.sweep import (
    FIRST_RATE,
    FIRST_STEPS,
    LONG_STEPS,
    SHORT_RATES,
    SWEEP_SEED,
    Sweep,
    finished,
    long_sweep,
    lowest,
    short_sweep,
)
from weightbank.train import BATCH, EVAL_BATCH, LOG_WINDOWS, RUNS, PickStats, Plan, Run

ARMS = tuple(name for name in ROUNDS_ONE_AND_TWO if rate_source(name) == name)

TEMPLATE = Path("report.in.md")
REPORT = Path("report.md")
CHART = Path("chart.png")
CURVES = Path("curves.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NARROWER_LINE = "#52514e"
SHALLOWER_LINE = "#a3a29c"
FIXED_LINE = "#0b0b0b"
PICKING = (*ROUTED, *ROUTER_VARIANTS)
COMPARED = (*PICKING, *FIXED_MAP.values())
COLORS = {
    "ff-routed": "#2a78d6",
    "attn-routed": "#eb6834",
    "both-routed": "#1baf7a",
    NOISE_FREE: "#8a5cd6",
    KEYS: "#c9477b",
}
DODGE = {NO_ROUTER: -9, "both-routed": 0, NOISE_FREE: 9, KEYS: 18}
MARKERS = {"ff-routed": "D", "attn-routed": "D", "both-routed": "D", NOISE_FREE: "^", KEYS: "v"}
assert COLORS.keys() == MARKERS.keys() == set(PICKING)
LATER_DODGE = 9
LATER_COLORS = ("#e6ab02", "#a6761d", "#66a61e", "#b3261e", "#1f3a8a", "#0e9aa7", "#7a3b8f", "#d4578f", "#5b6b2f")
DRAW_DODGE = -8
DRAW_CLEAR = 3
LEGEND_WRAP = 76
LAST_STRETCHES = (2000, 500)
LEGEND_COLUMNS = 2


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


def covers(curve: Sequence[tuple[int, object]], stored: int) -> bool:
    return curve[0][0] <= stored <= curve[-1][0]


def interpolate(curve: Sequence[tuple[int, float]], stored: int) -> float:
    for (low, low_loss), (high, high_loss) in zip(curve, curve[1:]):
        if low <= stored <= high:
            share = (math.log(stored) - math.log(low)) / (math.log(high) - math.log(low))
            return low_loss + share * (high_loss - low_loss)
    raise ValueError(f"{stored} stored weights lie outside the plain curve")


def by_seed(curve: Sequence[tuple[int, Sequence[float]]], stored: int) -> list[float]:
    return [interpolate([(size, losses[seed]) for size, losses in curve], stored) for seed in range(len(curve[0][1]))]


def means(curve: Sequence[Summary]) -> list[tuple[int, float]]:
    return [(summary.stored, summary.loss) for summary in curve]


@dataclass
class Budget:
    steps: int
    summaries: dict[str, Summary]

    @property
    def narrower(self) -> list[Summary]:
        return [self.summaries[name] for name in (*NARROWER, FULL) if name in self.summaries]

    @property
    def shallower(self) -> list[Summary]:
        return [self.summaries[name] for name in (*SHALLOWER, FULL) if name in self.summaries]

    def curves(self) -> dict[str, list[Summary]]:
        return {"narrower": self.narrower, "shallower": self.shallower}

    def read(self, curve: str, stored: int) -> float | None:
        points = means(self.curves()[curve])
        return interpolate(points, stored) if covers(points, stored) else None

    def better(self, name: str) -> str:
        stored = self.summaries[name].stored
        reads = {curve: loss for curve in self.curves() if (loss := self.read(curve, stored)) is not None}
        if not reads:
            raise ValueError(f"{name}: no plain curve reaches {stored} stored weights")
        return min(reads, key=reads.__getitem__)

    def gaps(self, name: str, curve: str) -> list[float]:
        summary = self.summaries[name]
        reads = by_seed([(plain.stored, plain.losses) for plain in self.curves()[curve]], summary.stored)
        return [loss - read for loss, read in zip(summary.losses, reads, strict=True)]

    def gap(self, name: str) -> float:
        return fmean(self.gaps(name, self.better(name)))


def read_budget(steps: int, rates: dict[str, float], seeds: Sequence[int]) -> Budget:
    summaries = {}
    for name, rate in rates.items():
        runs = [Plan(name, seed, steps, rate).read() for seed in seeds]
        if not all(math.isfinite(run.validation_loss) for run in runs):
            raise ValueError(f"{name}: a run ended with a loss that is not finite")
        summaries[name] = Summary(name, runs)
    return Budget(steps, summaries)


def complete(name: str, sweep: Sweep | None) -> dict[float, float]:
    losses = None if sweep is None else finished(sweep)
    if losses is None:
        raise ValueError(f"{name}: a learning-rate sweep is not finished")
    return losses


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def runs_table(ledger: Ledger) -> str:
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
        for run in (Run.read(RUNS.parent / row_run.file) for row in ledger.rows for row_run in row.runs)
    ]
    return table(
        ["configuration", "seed", "steps", "learning rate", "stored weights", "final validation loss", "training seconds"],
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


def sweep_table(sweeps: dict[str, dict[float, float]]) -> str:
    rates = sorted({rate for sweep in sweeps.values() for rate in sweep})
    rows = []
    for name, sweep in sweeps.items():
        best = lowest(sweep)
        cells = ["" if rate not in sweep else f"**{sweep[rate]:.4f}**" if rate == best else f"{sweep[rate]:.4f}" for rate in rates]
        lead = min(loss for rate, loss in sweep.items() if rate != best) - sweep[best]
        rows.append([name, *cells, f"{best:g}", f"{lead:.4f}"])
    return table(["configuration", *(f"{rate:g}" for rate in rates), "lowest", "lead over the next"], rows)


def span(values: Sequence[float]) -> str:
    return f"{min(values):+.4f} to {max(values):+.4f}"


def both_budgets_table(first: Budget, long: Budget) -> str:
    rows = [
        [
            name,
            f"{long.summaries[name].stored:,}",
            f"{first.summaries[name].rate:g}",
            f"{first.gap(name):+.4f}",
            span(first.gaps(name, first.better(name))),
            f"{long.summaries[name].rate:g}",
            f"{long.gap(name):+.4f}",
            span(long.gaps(name, long.better(name))),
            f"{fmean(gap for seed, gap in zip(SEEDS, long.gaps(name, long.better(name))) if seed != SWEEP_SEED):+.4f}",
            f"{long.gap(name) - first.gap(name):+.4f}",
        ]
        for name in COMPARED
    ]
    return table(
        [
            "configuration",
            "stored weights",
            f"learning rate, {first.steps:,} steps",
            f"gap, {first.steps:,} steps",
            "lowest to highest seed",
            f"learning rate, {long.steps:,} steps",
            f"gap, {long.steps:,} steps",
            "lowest to highest seed",
            f"gap without seed {SWEEP_SEED}",
            "long minus first",
        ],
        rows,
    )


def rate_alone_table(first: Budget, at_first_rate: Budget, at_best_rates: Budget) -> str:
    rows = []
    for name in COMPARED:
        if name not in at_best_rates.summaries:
            continue
        curve = first.better(name)
        (before,) = at_first_rate.gaps(name, curve)
        (after,) = at_best_rates.gaps(name, curve)
        rows.append(
            [
                name,
                curve,
                f"{before:+.4f}",
                f"{at_best_rates.summaries[name].rate:g}",
                f"{after:+.4f}",
                f"{after - before:+.4f}",
            ]
        )
    return table(
        [
            "configuration",
            "plain curve",
            f"gap at {FIRST_RATE:g} everywhere",
            "best short rate",
            "gap at the best short rates",
            "best minus first",
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


def signed(value: float | None) -> str:
    return "" if value is None else f"{value:+.4f}"


def unsigned(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def gap_table(budget: Budget) -> str:
    rows = []
    for name in COMPARED:
        summary = budget.summaries[name]
        by_width = budget.read("narrower", summary.stored)
        by_depth = budget.read("shallower", summary.stored)
        rows.append(
            [
                name,
                f"{summary.rate:g}",
                f"{summary.stored:,}",
                f"{summary.loss:.4f}",
                f"{summary.spread:.4f}",
                unsigned(by_width),
                signed(None if by_width is None else summary.loss - by_width),
                unsigned(by_depth),
                signed(None if by_depth is None else summary.loss - by_depth),
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


def versus_table(summaries: dict[str, Summary], names: Sequence[str], against: str) -> str:
    theirs = summaries[against]
    rows = []
    for name in names:
        ours = summaries[name]
        differences = [a - b for a, b in zip(ours.losses, theirs.losses)]
        rows.append(
            [
                name,
                f"{ours.rate:g}",
                f"{ours.stored:,}",
                *(f"{loss:.4f}" for loss in ours.losses),
                f"{ours.loss:.4f}",
                f"{ours.spread:.4f}",
                "" if name == against else f"{fmean(differences):+.4f}",
                "" if name == against else span(differences),
            ]
        )
    return table(
        [
            "configuration",
            "learning rate",
            "stored weights",
            *(f"seed {seed}" for seed in SEEDS),
            "mean",
            "seed sd",
            f"minus {against}",
            "lowest to highest seed",
        ],
        rows,
    )


def partner(chunks: int) -> str:
    return FULL if chunks == MODULES else named(modules=chunks)


def fixed_map_table(first: Budget, long: Budget) -> str:
    rows = []
    for chunks, name in FIXED_MAP.items():
        cells = [str(chunks), name, f"{long.summaries[name].stored:,}", partner(chunks)]
        for budget in (first, long):
            ours, theirs = budget.summaries[name], budget.summaries[partner(chunks)]
            cells += [
                f"{ours.loss:.4f}",
                f"{theirs.loss:.4f}",
                f"{ours.loss - theirs.loss:+.4f}",
                span([a - b for a, b in zip(ours.losses, theirs.losses)]),
                f"{budget.gap(name):+.4f}",
                budget.better(name),
            ]
        rows.append(cells)
    header = ["chunks", "configuration", "stored weights", "plain partner"]
    for budget in (first, long):
        header += [
            f"loss, {budget.steps:,} steps",
            "partner's loss",
            "minus partner",
            "lowest to highest seed",
            "gap to the plain curve",
            "curve",
        ]
    return table(header, rows)


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
        for name in PICKING
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
    for name in PICKING:
        for kind in ("feed_forward", "attention"):
            cells = []
            for budget in (first, long):
                picked: list[PickStats | None] = [getattr(run, kind) for run in budget.summaries[name].runs]
                stats = [run for run in picked if run is not None]
                if not stats:
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


def cell_text(ledger: Ledger, cell: dict[str, int | str]) -> str:
    return f"`{ledger.key(ledger.cell(cell))}`"


def axes_table(ledger: Ledger) -> str:
    rows = [
        [
            axis.name,
            axis.meaning,
            ", ".join(f"**{value}**" if value == axis.default else str(value) for value in axis.tags),
            ", ".join(f"`{tag}`" for value, tag in axis.tags.items() if value != axis.default or axis.name == STEPS),
        ]
        for axis in ledger.axes
    ]
    return table(["axis", "meaning", "values (default in bold)", "tags in a run's path"], rows)


def cells_table(ledger: Ledger) -> str:
    rows = []
    for row in ledger.rows:
        losses = row.losses
        rows.append(
            [
                f"`{ledger.key(row.cell)}`",
                *(str(row.cell[axis.name]) for axis in ledger.axes),
                f"{row.stored_weights:,}",
                "" if row.rate is None else f"{row.rate:g}",
                "" if losses is None else f"{fmean(losses):.4f}",
                "" if losses is None else f"{stdev(losses):.4f}",
                str(len(row.runs)),
            ]
        )
    return table(
        ["cell", *(axis.name for axis in ledger.axes), "stored weights", "learning rate", "mean loss", "seed sd", "runs"],
        rows,
    )


def hypotheses_table(ledger: Ledger) -> str:
    rows = []
    for hypothesis in sorted(ledger.hypotheses, key=lambda hypothesis: int(hypothesis["priority"])):
        settle = [
            f"{cell_text(ledger, comparison['lower'])} below {cell_text(ledger, comparison['higher'])}: "
            + (", ".join(f"{difference:+.4f}" for difference in comparison["differences"]) if "differences" in comparison else comparison.get(ABANDONED, "not run"))
            for comparison in hypothesis["settle"]
        ]
        rows.append(
            [
                str(hypothesis["priority"]),
                hypothesis["id"],
                hypothesis["statement"],
                "<br>".join(settle) or "no cell",
                f"**{hypothesis['status']}**",
            ]
        )
    return table(["priority", "hypothesis", "statement", "settling comparisons (lower minus higher, by seed)", "status"], rows)


def rounds_text(ledger: Ledger) -> str:
    parts = []
    for number, entry in enumerate(ledger.rounds, start=1):
        lines = [f"### Round {number}: {entry['hypothesis']}", "", entry["why"]]
        if "reading" in entry:
            lines += [
                "",
                entry["reading"],
                "",
                f"{len(entry['runs'])} runs, {entry['gpu_hours']:g} GPU-hours of training against an estimate of "
                f"{entry['estimated_gpu_hours']:g}; ledger rows written: " + ", ".join(f"`{key}`" for key in entry["rows"]) + ".",
            ]
        else:
            lines += ["", "Not finished."]
        if "note" in entry:
            lines += ["", entry["note"]]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) or "No round has run."


def later_cells(ledger: Ledger, steps: int) -> list[Row]:
    return [
        row
        for row in ledger.rows
        if row.cell[STEPS] == steps and row.losses is not None and ledger.name(row.cell) not in ROUNDS_ONE_AND_TWO
    ]


@dataclass(frozen=True)
class Counterpart:
    words: str
    stored: int
    losses: list[float]


def classic_counterpart(ledger: Ledger, row: Row) -> Counterpart | None:
    twin = ledger.row({**row.cell, ARCHITECTURE: ledger.axis(ARCHITECTURE).default})
    if twin is not None and twin.losses is not None:
        return Counterpart(f"`{ledger.key(twin.cell)}`", twin.stored_weights, twin.losses)
    steps = row.cell[STEPS]
    assert isinstance(steps, int)
    plain = [ledger.row(ledger.parse(name, steps)) for name in (*NARROWER, FULL)]
    curve = [(narrower.stored_weights, narrower.losses) for narrower in plain if narrower is not None and narrower.losses is not None]
    width_alone = ledger.sparse(row.cell).keys() <= {ARCHITECTURE, "width", STEPS}
    if width_alone and len(curve) == len(plain) and covers(curve, row.stored_weights):
        return Counterpart("the narrower plain curve, read at the same stored weights", row.stored_weights, by_seed(curve, row.stored_weights))
    return None


def beside_classic_table(ledger: Ledger, steps: int) -> str:
    classic = ledger.axis(ARCHITECTURE).default
    rows = []
    for row in later_cells(ledger, steps):
        if row.cell[ARCHITECTURE] == classic:
            continue
        losses = row.losses
        assert losses is not None
        cells = [f"`{ledger.key(row.cell)}`", f"{row.stored_weights:,}", f"{fmean(losses):.4f}"]
        counterpart = classic_counterpart(ledger, row)
        if counterpart is None:
            cells += ["none with runs", "", "", "", ""]
        else:
            differences = [ours - theirs for ours, theirs in zip(losses, counterpart.losses, strict=True)]
            cells += [
                counterpart.words,
                f"{counterpart.stored:,}",
                f"{fmean(counterpart.losses):.4f}",
                ", ".join(f"{difference:+.4f}" for difference in differences),
                f"{fmean(differences):+.4f}",
            ]
        rows.append(cells)
    return table(
        [
            "cell",
            "stored weights",
            "mean loss",
            f"{classic} counterpart",
            "its stored weights",
            "its mean loss",
            f"cell minus {classic}, by seed",
            "mean",
        ],
        rows,
    )


@dataclass
class Mark:
    label: str
    color: str
    rows: list[Row]


def marks(ledger: Ledger, later: Sequence[Row]) -> list[Mark]:
    no_draw = ledger.axis(DRAW).default
    grouped: dict[str, list[Row]] = {}
    for row in later:
        grouped.setdefault(ledger.key({**row.cell, DRAW: no_draw}), []).append(row)
    if len(grouped) > len(LATER_COLORS):
        raise ValueError(f"{len(grouped)} later cells and {len(LATER_COLORS)} colours to tell them apart")
    draws = ledger.axis(DRAW).words
    return [
        Mark(
            plain_words(rows[0].cell).replace(draws[rows[0].cell[DRAW]], ", ".join(draws[row.cell[DRAW]] for row in rows)),
            color,
            rows,
        )
        for rows, color in zip(grouped.values(), LATER_COLORS)
    ]


def chunk_map(cell: Cell) -> tuple[int, ...]:
    config = model_config(cell)
    bank = next(bank for bank in (config.feed_forward, config.attention) if isinstance(bank, Fixed))
    first_use: dict[int, int] = {}
    return tuple(first_use.setdefault(chunk, len(first_use)) for chunk in bank.map(config.modules))


def maps_by_twin(ledger: Ledger, steps: int) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = {}
    for row in ledger.rows:
        if row.cell[STEPS] != steps or row.losses is None or not other_map(row.cell):
            continue
        twin = ledger.row({**row.cell, PICK: ledger.axis(PICK).default, DRAW: ledger.axis(DRAW).default})
        if twin is None or twin.losses is None:
            continue
        if twin.rate != row.rate:
            raise ValueError(f"{ledger.key(row.cell)} trained at rate {row.rate}, its mod map at {twin.rate}")
        groups.setdefault(ledger.key(twin.cell), [twin]).append(row)
    return groups


def maps_table(ledger: Ledger, steps: int) -> str:
    rows = []
    for twin, *others in maps_by_twin(ledger, steps).values():
        assert twin.losses is not None
        for row in (twin, *others):
            chunks = chunk_map(row.cell)
            assert row.losses is not None
            differences = [ours - theirs for ours, theirs in zip(row.losses, twin.losses, strict=True)]
            rows.append(
                [
                    f"`{ledger.key(row.cell)}`",
                    " ".join(map(str, chunks)),
                    " ".join(str(chunks.count(chunk)) for chunk in sorted(set(chunks))),
                    str(sum(a == b for a, b in zip(chunks, chunks[1:]))),
                    *(f"{loss:.4f}" for loss in row.losses),
                    f"{fmean(row.losses):.4f}",
                    f"{stdev(row.losses):.4f}",
                    "" if row is twin else ", ".join(f"{difference:+.4f}" for difference in differences),
                    "" if row is twin else f"{fmean(differences):+.4f}",
                ]
            )
    return table(
        [
            "cell",
            "chunk of each module, first to last",
            "modules on each chunk",
            "neighbouring modules on one chunk",
            *(f"seed {seed}" for seed in SEEDS),
            "mean",
            "seed sd",
            "minus the mod map, by seed",
            "mean difference",
        ],
        rows,
    )


def map_spread(ledger: Ledger, steps: int) -> str:
    sentences = []
    for key, rows in maps_by_twin(ledger, steps).items():
        distinct = {chunk_map(row.cell): row.losses for row in reversed(rows) if row.losses is not None}
        losses = list(distinct.values())
        if len(losses) < 2:
            sentences.append(f"Beside `{key}`: every map trained lays the chunks out as the mod map does.")
            continue
        means = [fmean(seeds) for seeds in losses]
        spreads = [stdev(seeds) for seeds in losses]
        by_seed = [stdev(seeds[seed] for seeds in losses) for seed in range(len(SEEDS))]
        sentences.append(
            f"Beside `{key}`: {len(distinct)} distinct maps were trained, the mod map among them. Their mean losses run from "
            f"{min(means):.4f} to {max(means):.4f}, a standard deviation across maps of {stdev(means):.4f}"
            f" ({', '.join(f'{spread:.4f}' for spread in by_seed)} at training seeds {', '.join(map(str, SEEDS))} taken one at a time)."
            f" Across the {len(SEEDS)} training seeds of one map the standard deviation is {fmean(spreads):.4f} on average,"
            f" from {min(spreads):.4f} to {max(spreads):.4f}."
        )
    return "\n\n".join(sentences) or "No map but the mod map has runs."


def lead_cells(ledger: Ledger, steps: int) -> list[Row]:
    classic = ledger.axis(ARCHITECTURE).default
    return [row for row in later_cells(ledger, steps) if row.cell[ARCHITECTURE] != classic and not other_map(row.cell)]


def runs_of(ledger: Ledger, row: Row) -> list[Run]:
    rate, steps = row.rate, row.cell[STEPS]
    assert rate is not None and isinstance(steps, int)
    return [Plan(ledger.name(row.cell), seed, steps, rate).read() for seed in SEEDS]


def run_end_table(ledger: Ledger, names: Sequence[str], first: int, long: int) -> str:
    rows = []
    for name in names:
        row = ledger.row(ledger.parse(name, long))
        short = finished(short_sweep(name))
        assert row is not None and short is not None
        runs = runs_of(ledger, row)
        (ended,) = [run.validation_loss for run in runs if run.seed == SWEEP_SEED]
        falls = [fmean(logged(run, long) - logged(run, long - stretch) for run in runs) for stretch in LAST_STRETCHES]
        rows.append(
            [
                f"`{name}`",
                plain_words(row.cell),
                f"{min(short.values()):.4f}",
                f"{ended:.4f}",
                f"{ended - min(short.values()):+.4f}",
                *(f"{fall:+.4f}" for fall in falls),
            ]
        )
    return table(
        [
            "cell",
            "in plain words",
            f"seed {SWEEP_SEED}, stopped at {first:,} steps, best of {len(SHORT_RATES)} rates",
            f"seed {SWEEP_SEED}, stopped at {long:,} steps",
            f"what {long // first} times the steps bought",
            *(f"change over the last {stretch:,} of {long:,} steps, mean of seeds" for stretch in LAST_STRETCHES),
        ],
        rows,
    )


def curves(ledger: Ledger, steps: int, lead: Sequence[Mark]) -> None:
    figure, (whole, end) = plt.subplots(1, 2, figsize=(13, 5.6), dpi=160, facecolor=SURFACE)
    last = []
    for mark in lead:
        (row,) = mark.rows
        runs = runs_of(ledger, row)
        at = [int(step) for step, _ in runs[0].curve]
        mean = [fmean(logged(run, step) for run in runs) for step in at]
        last += [loss for run in runs for step, loss in run.curve if step >= steps * 3 // 4]
        for axes in (whole, end):
            for run in runs:
                axes.plot(at, [loss for _, loss in run.curve], color=mark.color, linewidth=0.8, alpha=0.35, zorder=2)
            axes.plot(at, mean, color=mark.color, linewidth=2, label=mark.label, zorder=3)
    margin = 0.08 * (max(last) - min(last))
    whole.set_xlim(0, steps)
    whole.set_ylim(min(last) - margin, min(last) + 0.35)
    end.set_xlim(steps * 3 // 4, steps)
    end.set_ylim(min(last) - margin, max(last) + margin)
    whole.set_title("the whole run", color=MUTED, fontsize=10, loc="left")
    end.set_title("the last quarter, enlarged", color=MUTED, fontsize=10, loc="left")
    for axes in (whole, end):
        axes.set_facecolor(SURFACE)
        axes.set_xlabel("optimizer steps", color=MUTED)
        axes.set_ylabel("validation loss (nats per byte)", color=MUTED)
        axes.grid(color=GRID, linewidth=1)
        axes.set_axisbelow(True)
        axes.tick_params(colors=MUTED)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(GRID)
    figure.suptitle(
        f"Validation loss along training, enwik8, {steps:,} steps, logged on the first {LOG_WINDOWS:,} validation windows"
        f" (heavy line: mean of {len(SEEDS)} seeds; light lines: each seed)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    handles, labels = whole.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=LEGEND_COLUMNS, frameon=False, labelcolor=INK, fontsize=10)
    figure.tight_layout(rect=(0, 0.04 + 0.035 * math.ceil(len(handles) / LEGEND_COLUMNS), 1, 0.95))
    figure.savefig(CURVES, facecolor=SURFACE)
    plt.close(figure)


def mebibytes(size: int) -> str:
    return f"{size / 2**20:,.0f}"


def memory_table(ledger: Ledger, measured: dict[str, memory.Memory], rows: Sequence[Row]) -> str:
    missing = [ledger.name(row.cell) for row in rows if ledger.name(row.cell) not in measured]
    if missing:
        raise ValueError(f"{memory.PATH} has no measurement of {missing}: python -m weightbank.memory {' '.join(missing)}")
    return table(
        [
            "cell",
            "in plain words",
            "stored weights",
            "weights alone, MiB",
            "inference, 1 window, MiB",
            f"inference, {EVAL_BATCH} windows, MiB",
            f"training, {BATCH} windows, MiB",
        ],
        [
            [
                f"`{ledger.name(row.cell)}`",
                plain_words(row.cell),
                f"{row.stored_weights:,}",
                *(mebibytes(size) for size in vars(measured[ledger.name(row.cell)]).values()),
            ]
            for row in rows
        ],
    )


def panel(
    axes: Axes, budget: Budget, later: Sequence[Mark], ledger: Ledger, low: float, high: float, fan: bool
) -> None:
    summaries = budget.summaries
    axes.set_facecolor(SURFACE)
    axes.set_xscale("log")
    plain = plain_words(ledger.cell({}))
    for curve, color, label in (
        (budget.narrower, NARROWER_LINE, f"{plain}, narrower ({ledger.axis('modules').words[MODULES]})"),
        (budget.shallower, SHALLOWER_LINE, f"{plain}, fewer modules ({ledger.axis('width').words[WIDTH]})"),
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
    fixed = [summaries[name] for name in FIXED_MAP.values()]
    axes.plot(
        [summary.stored / 1e6 for summary in fixed],
        [summary.loss for summary in fixed],
        color=FIXED_LINE,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        label=f"{ledger.words(ledger.parse(NO_ROUTER, budget.steps), (PICK,))} (squares, chunks beside each)",
        zorder=3,
    )
    points: list[tuple[Summary, str, str, str, str | None]] = [
        (summaries[name], FIXED_LINE, SURFACE, "s", None) for name in FIXED_MAP.values()
    ]
    points += [
        (summaries[name], COLORS[name], COLORS[name], MARKERS[name], plain_words(ledger.parse(name, budget.steps)))
        for name in PICKING
    ]
    for summary, color, fill, marker, legend in points:
        dodge = ScaledTranslation(DODGE.get(summary.name, 0) / 72, 0, axes.figure.dpi_scale_trans)
        axes.errorbar(
            [summary.stored / 1e6],
            [summary.loss],
            yerr=[[summary.loss - min(summary.losses)], [max(summary.losses) - summary.loss]],
            color=color,
            marker=marker,
            markersize=7,
            markerfacecolor=fill,
            markeredgecolor=SURFACE if fill == color else color,
            markeredgewidth=1.5,
            capsize=5,
            linewidth=1.5,
            linestyle="none",
            label=legend,
            transform=axes.transData + dodge,
            zorder=4,
        )
    for mark in later:
        for index, row in enumerate(mark.rows):
            losses = row.losses
            assert losses is not None
            mean = fmean(losses)
            grouped = len(mark.rows) > 1
            axes.errorbar(
                [row.stored_weights / 1e6],
                [mean],
                yerr=[[mean - min(losses)], [max(losses) - mean]],
                color=mark.color,
                marker="*",
                markersize=11 if grouped else 14,
                markeredgecolor=SURFACE,
                capsize=3 if grouped else 5,
                linewidth=1 if grouped else 1.5,
                linestyle="none",
                label=mark.label if index == 0 else None,
                transform=axes.transData
                + ScaledTranslation((LATER_DODGE + grouped * DRAW_DODGE * (index + DRAW_CLEAR if fan else 1)) / 72, 0, axes.figure.dpi_scale_trans),
                zorder=5,
            )
    visible = [loss for summary in summaries.values() if low <= summary.stored / 1e6 <= high for loss in summary.losses]
    visible += [loss for mark in later for row in mark.rows if low <= row.stored_weights / 1e6 <= high for loss in row.losses or ()]
    margin = 0.08 * (max(visible) - min(visible))
    axes.set_xlim(low, high)
    axes.set_ylim(min(visible) - margin, max(visible) + margin)
    for chunks, name in FIXED_MAP.items():
        summary = summaries[name]
        if low <= summary.stored / 1e6 <= high:
            axes.annotate(str(chunks), (summary.stored / 1e6, summary.loss), xytext=(DODGE.get(name, 0) - 9, -4), textcoords="offset points", color=INK, fontsize=10, ha="right")
    ticks = [tick for tick in (1, 2, 3, 4, 5, 6) if low <= tick <= high]
    axes.set_xticks(ticks, [str(tick) for tick in ticks])
    axes.set_xticks([], minor=True)
    axes.set_xlabel("stored weights (millions, log scale)", color=MUTED)
    axes.grid(color=GRID, linewidth=1)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)


def chart(budget: Budget, ledger: Ledger) -> None:
    later = marks(ledger, later_cells(ledger, budget.steps))
    figure, (whole, detail) = plt.subplots(1, 2, figsize=(13, 6.4), dpi=160, facecolor=SURFACE, width_ratios=(1, 1.25))
    panel(whole, budget, later, ledger, 0.85, 7.4, fan=False)
    panel(detail, budget, later, ledger, 3.0, 7.0, fan=True)
    for axes in (whole, detail):
        axes.set_ylabel("validation loss (nats per byte)", color=MUTED)
    whole.set_title("every size", color=MUTED, fontsize=10, loc="left")
    detail.set_title("3 to 7 million stored weights", color=MUTED, fontsize=10, loc="left")
    figure.suptitle(
        f"Validation loss against stored weights, enwik8, {budget.steps:,} steps, {len(SEEDS)} seeds (points and lines: mean; bars and bands: lowest to highest)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    handles, labels = whole.get_legend_handles_labels()
    labels = [textwrap.fill(label, LEGEND_WRAP) for label in labels]
    lines = sum(label.count("\n") + 1 for label in labels)
    architecture = ledger.axis(ARCHITECTURE)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=LEGEND_COLUMNS, frameon=False, labelcolor=INK, fontsize=9)
    figure.text(
        0.5,
        0.008,
        "The fixed map at 4 chunks and the three points with both kinds picked per token store the same weights to within 1.2% and are drawn side by side.\n"
        "A star is drawn just right of its stored weights; in the enlarged panel the small stars of several draws are spread out to its left, in the order the legend names them.\n"
        f"Lines and every point whose label names no architecture are the {architecture.words[architecture.default]};"
        " stars that name another architecture are compared with one another.",
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.095 + 0.028 * math.ceil(lines / LEGEND_COLUMNS), 1, 0.96))
    figure.savefig(CHART, facecolor=SURFACE)
    plt.close(figure)
    lead = lead_cells(ledger, budget.steps)
    curves(ledger, budget.steps, [mark for mark in later if mark.rows[0] in lead])


def main() -> None:
    ledger = Ledger.read()
    ledger.refresh(RUNS)
    first = read_budget(FIRST_STEPS, {name: FIRST_RATE for name in ROUNDS_ONE_AND_TWO}, SEEDS)
    short = {name: complete(name, short_sweep(name)) for name in ARMS}
    long_sweeps = {name: complete(name, long_sweep(name)) for name in ARMS}
    rates = {name: lowest(sweep) for name, sweep in long_sweeps.items()}
    long = read_budget(LONG_STEPS, {name: rates[rate_source(name)] for name in ROUNDS_ONE_AND_TWO}, SEEDS)
    chart(long, ledger)
    sections = {
        "axes": axes_table(ledger),
        "hypotheses": hypotheses_table(ledger),
        "rounds": rounds_text(ledger),
        "cells": cells_table(ledger),
        "beside-classic": beside_classic_table(ledger, LONG_STEPS),
        "memory": memory_table(ledger, memory.read(), lead_cells(ledger, LONG_STEPS)),
        "maps": maps_table(ledger, LONG_STEPS),
        "map-spread": map_spread(ledger, LONG_STEPS),
        "run-end": run_end_table(
            ledger, [*(ledger.name(row.cell) for row in lead_cells(ledger, LONG_STEPS)), FULL, NO_ROUTER, BOTH_ROUTED], FIRST_STEPS, LONG_STEPS
        ),
        "both-budgets": both_budgets_table(first, long),
        "rate-alone": rate_alone_table(
            first,
            read_budget(FIRST_STEPS, {name: FIRST_RATE for name in ARMS}, (SWEEP_SEED,)),
            read_budget(FIRST_STEPS, {name: lowest(sweep) for name, sweep in short.items()}, (SWEEP_SEED,)),
        ),
        "short-sweep": sweep_table(short),
        "long-sweep": sweep_table(long_sweeps),
        "plain": plain_table(first, long),
        "gaps-long": gap_table(long),
        "gaps-first": gap_table(first),
        "sanity-first": paired_table(first.summaries, OWN_CHUNK, FULL),
        "sanity-long": paired_table(long.summaries, OWN_CHUNK, FULL),
        "routers-long": versus_table(long.summaries, (NO_ROUTER, "both-routed", *ROUTER_VARIANTS), NO_ROUTER),
        "routers-first": versus_table(first.summaries, (NO_ROUTER, "both-routed", *ROUTER_VARIANTS), NO_ROUTER),
        "fixed-map": fixed_map_table(first, long),
        "during-training": during_training_table(long),
        "sampled": sampled_table(long.summaries),
        "log-windows": f"{LOG_WINDOWS:,}",
        "pick-variation": pick_variation_table(first, long),
        "runs": runs_table(ledger),
        "summary-long": summary_table(long.summaries),
        "summary-first": summary_table(first.summaries),
        **{f"picks:{name}": picks_table(long.summaries[name]) for name in PICKING},
    }
    text = TEMPLATE.read_text()
    for key, section in sections.items():
        if "{{" + key + "}}" not in text:
            raise ValueError(f"{TEMPLATE} has no place for the section {key}")
        text = text.replace("{{" + key + "}}", section)
    if "{{" in text:
        raise ValueError(f"{TEMPLATE} names a section this report does not write")
    REPORT.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
