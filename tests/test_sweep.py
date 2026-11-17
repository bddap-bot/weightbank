from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from weightbank import train
from weightbank.configs import SEEDS
from weightbank.sweep import ARMS, FIRST_STEPS, LONG_STEPS, chosen, planned, rate_sweep, sweep_plan
from weightbank.train import Plan, Run

ARM = "ff-routed"


@pytest.fixture(autouse=True)
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "RUNS", tmp_path)


def record(plan: Plan, loss: float) -> None:
    run = Run(plan.config, plan.seed, plan.steps, plan.peak_rate, 1, loss, loss, 0.0, None, None, [])
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(json.dumps(asdict(run)))


def test_a_run_file_reads_back_under_the_path_of_its_plan() -> None:
    plan = Plan(ARM, 2, 32000, 5e-4)
    record(plan, 1.25)
    assert plan.path.name == "ff-routed-lr0.0005-s2.json"
    assert plan.path.parent.name == "32000"
    assert Run.read(plan.path).peak_rate == 5e-4
    assert plan.command == [ARM, "--seed", "2", "--steps", "32000", "--rate", "0.0005"]


def test_no_rate_is_chosen_before_every_candidate_has_run() -> None:
    record(sweep_plan(ARM, 5e-4), 1.2)
    record(sweep_plan(ARM, 1e-3), 1.1)
    assert rate_sweep(ARM) == ([5e-4, 1e-3, 2e-3], None)
    assert chosen(ARM) is None


def test_the_lowest_loss_inside_the_tried_rates_is_chosen() -> None:
    for rate, loss in ((5e-4, 1.2), (1e-3, 1.1), (2e-3, 1.15)):
        record(sweep_plan(ARM, rate), loss)
    assert rate_sweep(ARM) == ([5e-4, 1e-3, 2e-3], {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.15})
    assert chosen(ARM) == 1e-3


def test_a_lowest_loss_on_the_edge_adds_the_next_rate_until_it_is_bracketed() -> None:
    for rate, loss in ((5e-4, 1.2), (1e-3, 1.1), (2e-3, 1.05)):
        record(sweep_plan(ARM, rate), loss)
    assert rate_sweep(ARM) == ([5e-4, 1e-3, 2e-3, 4e-3], None)
    record(sweep_plan(ARM, 4e-3), 1.0)
    assert rate_sweep(ARM) == ([5e-4, 1e-3, 2e-3, 4e-3, 8e-3], None)
    record(sweep_plan(ARM, 8e-3), 1.3)
    assert chosen(ARM) == 4e-3


def test_a_lowest_loss_on_the_low_edge_adds_a_lower_rate() -> None:
    for rate, loss in ((5e-4, 1.0), (1e-3, 1.1), (2e-3, 1.2)):
        record(sweep_plan(ARM, rate), loss)
    assert rate_sweep(ARM)[0] == [5e-4, 1e-3, 2e-3, 2.5e-4]
    record(sweep_plan(ARM, 2.5e-4), 1.3)
    assert chosen(ARM) == 5e-4


def test_long_runs_are_planned_only_at_a_chosen_rate() -> None:
    assert not [plan for plan in planned() if plan.steps == LONG_STEPS]
    for rate, loss in ((5e-4, 1.2), (1e-3, 1.1), (2e-3, 1.05), (4e-3, 1.3)):
        record(sweep_plan(ARM, rate), loss)
    long = [plan for plan in planned() if plan.steps == LONG_STEPS]
    assert long == [Plan(ARM, seed, LONG_STEPS, 2e-3) for seed in SEEDS]
    assert LONG_STEPS == 4 * FIRST_STEPS
    assert "own-chunk" not in ARMS


def test_a_run_file_of_another_plan_is_refused() -> None:
    plan = Plan(ARM, 0, 8000, 1e-3)
    record(plan, 1.0)
    other = Plan(ARM, 0, 8000, 2e-3)
    plan.path.rename(other.path)
    with pytest.raises(ValueError):
        other.read()
