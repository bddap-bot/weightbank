from __future__ import annotations

import json
from statistics import fmean
from typing import Any

import pytest

from weightbank import report
from weightbank.configs import ARCHITECTURE, named
from weightbank.ledger import PATH, STEPS, Cell, Ledger, Row
from weightbank.sweep import LONG_STEPS

SHARED: Cell = {"shared": "both", STEPS: LONG_STEPS}
NARROWED: Cell = {"width": 184, STEPS: LONG_STEPS}
BELOW: Cell = {"width": 160, STEPS: LONG_STEPS}
ABOVE: Cell = {"width": 192, STEPS: LONG_STEPS}
MODERN_CELLS: tuple[Cell, ...] = ({"shared": "both"}, {"modules": 4}, {"width": 184})


def modern(cell: Cell) -> Cell:
    return {**cell, ARCHITECTURE: "modern"}


def ledger_without(*dropped: Cell) -> Ledger:
    fields: dict[str, Any] = json.loads(PATH.read_text())
    return Ledger({**fields, "cells": [row for row in fields["cells"] if row["cell"] not in dropped]})


def row(ledger: Ledger, cell: Cell) -> Row:
    found = ledger.row(ledger.cell(cell))
    assert found is not None and found.losses is not None
    return found


def losses(found: Row) -> list[float]:
    assert found.losses is not None
    return found.losses


def test_a_modern_cell_is_set_beside_the_same_cell_in_the_classic_architecture() -> None:
    ledger = Ledger.read()
    classic = row(ledger, SHARED)
    assert report.classic_counterpart(ledger, row(ledger, modern(SHARED))) == report.Counterpart(
        "`32000/both`", classic.stored_weights, losses(classic)
    )


def test_a_narrowed_modern_cell_without_a_classic_twin_is_read_off_the_narrower_plain_curve_seed_by_seed() -> None:
    ledger = Ledger.read()
    narrowed = row(ledger, modern(NARROWED))
    below, above = row(ledger, BELOW), row(ledger, ABOVE)
    assert below.stored_weights < narrowed.stored_weights < above.stored_weights
    counterpart = report.classic_counterpart(ledger, narrowed)
    assert counterpart is not None and counterpart.stored == narrowed.stored_weights
    assert counterpart.losses == [
        report.interpolate(
            [(below.stored_weights, losses(below)[seed]), (above.stored_weights, losses(above)[seed])],
            narrowed.stored_weights,
        )
        for seed in range(3)
    ]
    assert len(set(counterpart.losses)) == 3


def test_a_modern_cell_that_is_not_narrowed_alone_has_no_counterpart_without_its_classic_twin() -> None:
    ledger = ledger_without(SHARED)
    assert report.classic_counterpart(ledger, row(ledger, modern(SHARED))) is None


def test_a_narrowed_cell_outside_the_plain_curve_has_no_counterpart() -> None:
    ledger = Ledger.read()
    narrowed = row(ledger, modern(NARROWED))
    smallest = row(ledger, {"width": 96, STEPS: LONG_STEPS})
    assert report.classic_counterpart(ledger, Row(narrowed.cell, smallest.stored_weights - 1, narrowed.runs)) is None


def test_a_plain_curve_with_a_width_missing_is_not_read_across_the_hole() -> None:
    ledger = ledger_without(ABOVE)
    assert report.classic_counterpart(ledger, row(ledger, modern(NARROWED))) is None


def test_the_table_beside_classic_has_one_row_per_modern_cell_and_none_for_a_classic_one() -> None:
    ledger = Ledger.read()
    lines = report.beside_classic_table(ledger, LONG_STEPS).splitlines()[2:]
    keys = [line.split(" | ")[0] for line in lines]
    assert keys == [f"| `32000/{named(**modern(cell))}`" for cell in MODERN_CELLS]


def test_a_row_beside_classic_gives_both_cells_and_the_modern_one_minus_the_classic_by_seed() -> None:
    ledger = Ledger.read()
    ours, theirs = row(ledger, modern(SHARED)), row(ledger, SHARED)
    differences = [a - b for a, b in zip(losses(ours), losses(theirs), strict=True)]
    assert len({round(difference, 4) for difference in differences}) == 3 and ours.stored_weights != theirs.stored_weights
    expected = [
        "`32000/both-modern`",
        f"{ours.stored_weights:,}",
        f"{fmean(losses(ours)):.4f}",
        "`32000/both`",
        f"{theirs.stored_weights:,}",
        f"{fmean(losses(theirs)):.4f}",
        ", ".join(f"{difference:+.4f}" for difference in differences),
        f"{fmean(differences):+.4f}",
    ]
    assert "| " + " | ".join(expected) + " |" in report.beside_classic_table(ledger, LONG_STEPS).splitlines()


def test_a_modern_cell_with_no_counterpart_keeps_its_row() -> None:
    ledger = ledger_without(SHARED)
    (line,) = [line for line in report.beside_classic_table(ledger, LONG_STEPS).splitlines() if "`32000/both-modern`" in line]
    assert line.endswith("| none with runs |  |  |  |  |")


def test_the_gap_by_seed_reads_the_plain_curve_of_the_same_seed() -> None:
    curve = [(100, [1.0, 2.0]), (400, [0.0, 4.0])]
    assert report.by_seed(curve, 200) == pytest.approx([0.5, 3.0])
