from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

PATH = Path("ledger.json")
STEPS = "steps"
SEEDS = (0, 1, 2)
LINE = 110
OPEN = "open"
SUPPORTED = "supported"
REFUTED = "refuted"
SIDES = ("lower", "higher")
ABANDONED = "abandoned"

Value = int | str
Cell = dict[str, Value]


@dataclass(frozen=True)
class Axis:
    name: str
    meaning: str
    default: Value
    tags: dict[Value, str]
    words: dict[Value, str]

    def __post_init__(self) -> None:
        if self.default not in self.tags:
            raise ValueError(f"axis {self.name}: the default {self.default!r} is not one of its values")
        if len(set(self.tags.values())) != len(self.tags) or "" in self.tags.values():
            raise ValueError(f"axis {self.name}: every value needs its own tag")
        if len(set(self.words.values())) != len(self.words) or "" in self.words.values():
            raise ValueError(f"axis {self.name}: every value needs its own plain words")


@dataclass
class RunRow:
    file: str
    rate: float
    seed: int
    loss: float
    seconds: float


@dataclass
class Row:
    cell: Cell
    stored_weights: int
    runs: list[RunRow]

    def at(self, rate: float) -> dict[int, float]:
        return {run.seed: run.loss for run in self.runs if run.rate == rate}

    @property
    def rate(self) -> float | None:
        complete = [
            rate
            for rate in sorted({run.rate for run in self.runs})
            if all(math.isfinite(self.at(rate).get(seed, math.inf)) for seed in SEEDS)
        ]
        return min(complete, key=lambda rate: fmean(self.at(rate).values()), default=None)

    @property
    def losses(self) -> list[float] | None:
        rate = self.rate
        return None if rate is None else [self.at(rate)[seed] for seed in SEEDS]

    def written(self) -> dict[str, Any]:
        losses = self.losses
        result = {} if losses is None else {"rate": self.rate, "loss": round(fmean(losses), 4), "seed_sd": round(stdev(losses), 4)}
        return {"cell": self.cell, "stored_weights": self.stored_weights, **result, "runs": [vars(run) for run in self.runs]}


def verdict(lower: list[float], higher: list[float]) -> str:
    differences = [ours - theirs for ours, theirs in zip(lower, higher, strict=True)]
    if abs(fmean(differences)) <= max(stdev(lower), stdev(higher)):
        return OPEN
    if all(difference < 0 for difference in differences):
        return SUPPORTED
    if all(difference > 0 for difference in differences):
        return REFUTED
    return OPEN


def render(value: Any, depth: int = 0, room: int = LINE) -> str:
    flat = json.dumps(value)
    if not isinstance(value, dict | list) or not value or len(flat) <= room:
        return flat
    pad = " " * (depth + 1)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(key)}: {render(item, depth + 1, LINE - len(pad) - len(key) - 5)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * depth + "}"
    return "[\n" + ",\n".join(pad + render(item, depth + 1, LINE - len(pad) - 1) for item in value) + "\n" + " " * depth + "]"


class Ledger:
    def __init__(self, fields: dict[str, Any]) -> None:
        self.default_name: str = fields["default_name"]
        self.axes = tuple(
            Axis(
                axis["name"],
                axis["meaning"],
                axis["default"],
                {value: tag for value, tag, _ in axis["values"]},
                {value: words for value, _, words in axis["values"]},
            )
            for axis in fields["axes"]
        )
        if [axis.name for axis in self.axes].count(STEPS) != 1 or len({axis.name for axis in self.axes}) != len(self.axes):
            raise ValueError(f"the axes need distinct names and one named {STEPS}")
        tags = [tag for axis in self.model_axes for tag in axis.tags.values()]
        if len(set(tags)) != len(tags):
            raise ValueError("two axes share a tag, so a name could spell two cells")
        self.hypotheses: list[dict[str, Any]] = [
            {
                **hypothesis,
                "settle": [
                    {**comparison, **{side: self.sparse(self.cell(comparison[side])) for side in SIDES}}
                    for comparison in hypothesis["settle"]
                ],
            }
            for hypothesis in fields["hypotheses"]
        ]
        if len({hypothesis["id"] for hypothesis in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("two hypotheses share an id")
        self.rounds: list[dict[str, Any]] = [dict(entry) for entry in fields["rounds"]]
        self.rows = [
            Row(self.cell(row["cell"]), row["stored_weights"], [RunRow(**run) for run in row["runs"]])
            for row in fields["cells"]
        ]

    @staticmethod
    def read(path: Path = PATH) -> Ledger:
        return Ledger(json.loads(path.read_text()))

    def write(self, path: Path = PATH) -> None:
        fields = {
            "default_name": self.default_name,
            "axes": [
                {
                    "name": axis.name,
                    "meaning": axis.meaning,
                    "default": axis.default,
                    "values": [[value, tag, axis.words[value]] for value, tag in axis.tags.items()],
                }
                for axis in self.axes
            ],
            "hypotheses": self.hypotheses,
            "rounds": self.rounds,
            "cells": [{**row.written(), "cell": self.sparse(row.cell)} for row in self.rows],
        }
        path.write_text(render(fields) + "\n")

    @property
    def model_axes(self) -> tuple[Axis, ...]:
        return tuple(axis for axis in self.axes if axis.name != STEPS)

    def axis(self, name: str) -> Axis:
        (axis,) = [axis for axis in self.axes if axis.name == name]
        return axis

    def cell(self, changed: Cell) -> Cell:
        for name, value in changed.items():
            if value not in self.axis(name).tags:
                raise ValueError(f"axis {name} has no value {value!r}")
        return {axis.name: changed.get(axis.name, axis.default) for axis in self.axes}

    def sparse(self, cell: Cell) -> Cell:
        return {axis.name: cell[axis.name] for axis in self.axes if cell[axis.name] != axis.default or axis.name == STEPS}

    def name(self, cell: Cell) -> str:
        tags = [axis.tags[cell[axis.name]] for axis in self.model_axes if cell[axis.name] != axis.default]
        return "-".join(tags) or self.default_name

    def words(self, cell: Cell, spelled: tuple[str, ...] = ()) -> str:
        return ", ".join(
            axis.words[cell[axis.name]]
            for axis in self.model_axes
            if cell[axis.name] != axis.default or axis.name in spelled
        )

    def key(self, cell: Cell) -> str:
        return f"{self.axis(STEPS).tags[cell[STEPS]]}/{self.name(cell)}"

    def parse(self, name: str, steps: int) -> Cell:
        def walk(rest: str, axes: tuple[Axis, ...]) -> Cell | None:
            if not axes:
                return {} if rest == "" else None
            axis, later = axes[0], axes[1:]
            options = [(axis.default, rest)]
            options += [
                (value, rest[len(tag) :].removeprefix("-"))
                for value, tag in axis.tags.items()
                if value != axis.default and (rest == tag or rest.startswith(tag + "-"))
            ]
            for value, remaining in options:
                found = walk(remaining, later)
                if found is not None:
                    return {axis.name: value, **found}
            return None

        found = walk("" if name == self.default_name else name, self.model_axes)
        if found is None or self.name(self.cell(found)) != name:
            raise ValueError(f"{name!r} names no cell of the ledger's axes")
        return self.cell({**found, STEPS: steps})

    def run_path(self, runs: Path, cell: Cell, rate: float, seed: int) -> Path:
        return runs / self.axis(STEPS).tags[cell[STEPS]] / f"{self.name(cell)}-lr{rate!r}-s{seed}.json"

    def row(self, cell: Cell) -> Row | None:
        return next((row for row in self.rows if row.cell == cell), None)

    def losses(self, comparison: dict[str, Any]) -> tuple[list[float], list[float]] | None:
        lower, higher = (self.row(self.cell(comparison[side])) for side in SIDES)
        if lower is None or higher is None or lower.losses is None or higher.losses is None:
            return None
        return lower.losses, higher.losses

    def refresh(self, runs: Path) -> None:
        rows: dict[str, Row] = {}
        budgets = self.axis(STEPS).tags.values()
        for path in sorted(path for budget in budgets for path in (runs / budget).glob("*.json")):
            run = json.loads(path.read_text())
            cell = self.parse(run["config"], run["steps"])
            if path != self.run_path(runs, cell, run["peak_rate"], run["seed"]):
                raise ValueError(f"{path} holds a run of another cell, rate or seed")
            row = rows.setdefault(self.key(cell), Row(cell, run["stored_weights"], []))
            if row.stored_weights != run["stored_weights"]:
                raise ValueError(f"{path} stores {run['stored_weights']} weights, other runs of its cell {row.stored_weights}")
            row.runs.append(
                RunRow(
                    path.relative_to(runs.parent).as_posix(),
                    run["peak_rate"],
                    run["seed"],
                    round(run["validation_loss"], 4),
                    round(run["training_seconds"]),
                )
            )
        self.rows = list(rows.values())
        for hypothesis in self.hypotheses:
            verdicts = []
            for comparison in hypothesis["settle"]:
                comparison.pop("differences", None)
                losses = self.losses(comparison)
                if losses is not None:
                    comparison["differences"] = [round(lower - higher, 4) for lower, higher in zip(*losses)]
                    verdicts.append(verdict(*losses))
            settled = len(verdicts) == len(hypothesis["settle"]) and len(set(verdicts)) == 1
            hypothesis["status"] = verdicts[0] if settled else OPEN
