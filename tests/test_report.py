from __future__ import annotations

import json
from statistics import fmean
from typing import Any

import pytest

from weightbank import memory, report
from weightbank.configs import ARCHITECTURE, named
from weightbank.ledger import PATH, STEPS, Cell, Ledger, Row, RunRow
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
    tried = [
        f"| `{ledger.key(row.cell)}`"
        for row in ledger.rows
        if row.cell[STEPS] == LONG_STEPS and row.cell[ARCHITECTURE] == "modern" and row.losses is not None
    ]
    assert keys == tried and set(tried) >= {f"| `32000/{named(**modern(cell))}`" for cell in MODERN_CELLS}


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


def with_rows(*cells: tuple[Cell, list[float]]) -> Ledger:
    fields: dict[str, Any] = json.loads(PATH.read_text())
    made = Ledger({**fields, "cells": []})
    for cell, seeds in cells:
        full = made.cell({**cell, STEPS: LONG_STEPS})
        runs = [RunRow(f"runs/{made.key(full)}-s{seed}.json", 2e-3, seed, loss, 1.0) for seed, loss in enumerate(seeds)]
        made.rows.append(Row(full, 7, runs))
    return made


MOD: Cell = modern({"shared": "both"})


def drawn(number: int) -> Cell:
    return {**MOD, "pick": "drawn", "draw": number}


def test_a_map_is_numbered_by_first_use_so_equal_maps_read_alike() -> None:
    ledger = Ledger.read()
    assert report.chunk_map(ledger.cell(MOD)) == (0, 1, 2, 3, 0, 1, 2, 3)
    assert report.chunk_map(ledger.cell(drawn(2))) == (0, 0, 1, 1, 0, 2, 2, 3)
    assert report.chunk_map(ledger.cell({"shared": "both", "pick": "ends"})) == (0, 1, 2, 1, 2, 1, 2, 3)
    assert report.chunk_map(ledger.cell({"shared": "attn", "pick": "drawn", "draw": 2})) == (0, 0, 1, 1, 0, 2, 2, 3)


def test_the_maps_table_sets_each_drawn_map_against_the_mod_map_seed_by_seed() -> None:
    ledger = with_rows(
        (MOD, [1.0, 1.1, 1.2]), (drawn(2), [1.05, 1.1, 1.1]), ({"shared": "both"}, [1.3, 1.3, 1.3]), ({**MOD, "pick": "ends"}, [1.0, 1.0, 1.0])
    )
    _, _, mod, second, ends = report.maps_table(ledger, LONG_STEPS).splitlines()
    assert mod == "| `32000/both-modern` | 0 1 2 3 0 1 2 3 | 2 2 2 2 | 0 | 1.0000 | 1.1000 | 1.2000 | 1.1000 | 0.1000 |  |  |"
    assert second == (
        "| `32000/both-drawn-d2-modern` | 0 0 1 1 0 2 2 3 | 3 2 2 1 | 3 | 1.0500 | 1.1000 | 1.1000 | 1.0833 | 0.0289 "
        "| +0.0500, +0.0000, -0.1000 | -0.0167 |"
    )
    assert ends.startswith("| `32000/both-ends-modern` | 0 1 2 1 2 1 2 3 | 1 3 3 1 | 0 | 1.0000 | 1.0000 | 1.0000 |")
    assert report.maps_by_twin(with_rows((drawn(2), [1.05, 1.1, 1.1])), LONG_STEPS) == {}


def test_a_map_trained_at_another_rate_than_its_mod_map_is_refused() -> None:
    ledger = with_rows((MOD, [1.0, 1.1, 1.2]), (drawn(2), [1.05, 1.1, 1.1]))
    for run in ledger.rows[1].runs:
        run.rate = 1e-3
    with pytest.raises(ValueError, match="its mod map at 0.002"):
        report.maps_by_twin(ledger, LONG_STEPS)


def test_the_spread_across_maps_is_given_beside_the_spread_across_training_seeds() -> None:
    ledger = with_rows((MOD, [1.0, 1.1, 1.2]), (drawn(1), [1.2, 1.3, 1.4]), (drawn(2), [1.4, 1.5, 1.6]))
    text = report.map_spread(ledger, LONG_STEPS)
    assert "3 distinct maps were trained, the mod map among them" in text
    assert "from 1.1000 to 1.5000, a standard deviation across maps of 0.2000" in text
    assert "(0.2000, 0.2000, 0.2000 at training seeds 0, 1, 2 taken one at a time)" in text
    assert "one map the standard deviation is 0.1000 on average, from 0.1000 to 0.1000" in text
    assert report.map_spread(with_rows((MOD, [1.0, 1.1, 1.2])), LONG_STEPS) == "No map but the mod map has runs."


def test_two_draws_of_one_map_count_once_in_the_spread() -> None:
    one: Cell = {"shared": "both", "chunks": 1, "architecture": "modern"}
    ledger = with_rows(
        (one, [1.0, 1.1, 1.2]), ({**one, "pick": "drawn", "draw": 1}, [1.2, 1.3, 1.4]), ({**one, "pick": "drawn", "draw": 2}, [9.0, 9.0, 9.0])
    )
    assert report.map_spread(ledger, LONG_STEPS) == "Beside `32000/both-c1-modern`: every map trained lays the chunks out as the mod map does."
    two: Cell = {"shared": "both", "chunks": 2, "modules": 4, "architecture": "modern"}
    ledger = with_rows(
        (two, [1.0, 1.1, 1.2]), ({**two, "pick": "drawn", "draw": 3}, [9.0, 9.0, 9.0]), ({**two, "pick": "drawn", "draw": 1}, [1.2, 1.3, 1.4])
    )
    text = report.map_spread(ledger, LONG_STEPS)
    assert "2 distinct maps were trained" in text and "from 1.1000 to 1.3000" in text


def test_draws_of_one_cell_share_a_legend_entry_that_names_each_draw() -> None:
    ledger = with_rows(
        (MOD, [1.0, 1.1, 1.2]), (drawn(1), [1.2, 1.3, 1.4]), (drawn(3), [1.4, 1.5, 1.6]), (modern({"modules": 4}), [1.0, 1.0, 1.0])
    )
    found = report.marks(ledger, report.later_cells(ledger, LONG_STEPS))
    assert [(mark.label, len(mark.rows)) for mark in found] == [
        ("attention and feed-forward shared, fixed map, 4 chunks, modern architecture", 1),
        ("attention and feed-forward shared, fixed map drawn at random, draw 1, draw 3, 4 chunks, modern architecture", 2),
        ("nothing shared, 4 modules, modern architecture", 1),
    ]
    assert len({mark.color for mark in found}) == 3


def test_the_lead_cells_are_the_modern_ones_with_a_sweep_of_their_own() -> None:
    ledger = with_rows(
        (MOD, [1.0, 1.1, 1.2]),
        (drawn(1), [1.2, 1.3, 1.4]),
        ({**MOD, "pick": "ends"}, [1.0, 1.0, 1.0]),
        ({"shared": "both", "chunks": 1, "rank": 24}, [1.0, 1.0, 1.0]),
    )
    assert [ledger.key(row.cell) for row in report.lead_cells(ledger, LONG_STEPS)] == ["32000/both-modern"]


def test_the_memory_table_wants_a_measurement_of_every_lead_cell() -> None:
    ledger = with_rows((MOD, [1.0, 1.1, 1.2]))
    rows = report.lead_cells(ledger, LONG_STEPS)
    measured = {"both-modern": memory.Memory(5 * 2**20, 6 * 2**20, 30 * 2**20, 40 * 2**20)}
    header, _, row = report.memory_table(ledger, measured, rows).splitlines()
    assert header == (
        "| cell | in plain words | stored weights | weights alone, MiB | inference, 1 window, MiB "
        "| inference, 256 windows, MiB | training, 64 windows, MiB |"
    )
    assert row == "| `both-modern` | attention and feed-forward shared, fixed map, 4 chunks, modern architecture | 7 | 5 | 6 | 30 | 40 |"
    with pytest.raises(ValueError, match="no measurement of .'both-modern'."):
        report.memory_table(ledger, {}, rows)
