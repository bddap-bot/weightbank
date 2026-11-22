from __future__ import annotations

from statistics import fmean

import pytest

from weightbank import report
from weightbank.configs import ROUNDS_ONE_AND_TWO
from weightbank.ledger import SEEDS, Ledger, Row
from weightbank.sweep import LONG_STEPS


def long_budget(ledger: Ledger) -> report.Budget:
    rates = {}
    for name in ROUNDS_ONE_AND_TWO:
        row = ledger.row(ledger.parse(name, LONG_STEPS))
        assert row is not None and row.rate is not None
        rates[name] = row.rate
    return report.read_budget(LONG_STEPS, rates, SEEDS)


def modern(ledger: Ledger, **changed: int | str) -> Row:
    row = ledger.row(ledger.cell({**changed, "steps": LONG_STEPS, "architecture": "modern"}))
    assert row is not None
    return row


def test_a_modern_cell_is_set_beside_the_same_cell_in_the_classic_architecture() -> None:
    ledger = Ledger.read()
    counterpart = report.classic_counterpart(ledger, long_budget(ledger), modern(ledger, shared="both"))
    classic = ledger.row(ledger.cell({"shared": "both", "steps": LONG_STEPS}))
    assert classic is not None and classic.losses is not None
    assert counterpart == ("`32000/both`", classic.stored_weights, classic.losses)


def test_a_narrowed_modern_cell_without_a_classic_twin_is_read_against_the_narrower_plain_curve() -> None:
    ledger = Ledger.read()
    budget = long_budget(ledger)
    row = modern(ledger, width=184)
    counterpart = report.classic_counterpart(ledger, budget, row)
    assert counterpart is not None
    words, stored, losses = counterpart
    assert "curve" in words and stored == row.stored_weights
    below, above = (budget.summaries[name] for name in ("plain-w160", "plain-w192"))
    assert below.stored < stored < above.stored
    for seed, loss in enumerate(losses):
        assert above.losses[seed] < loss < below.losses[seed]
    assert fmean(losses) == pytest.approx(report.interpolate(report.means(budget.narrower), stored), abs=1e-9)


def test_every_modern_cell_with_three_seeds_has_a_row_in_the_table_of_architectures() -> None:
    ledger = Ledger.read()
    text = report.architectures_table(ledger, long_budget(ledger))
    cells = [row for row in report.later_cells(ledger, LONG_STEPS) if row.cell["architecture"] == "modern"]
    assert len(cells) >= 3
    for row in cells:
        assert f"| `{ledger.key(row.cell)}` |" in text
