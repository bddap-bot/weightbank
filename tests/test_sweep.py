from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from weightbank import train
from weightbank.configs import CONFIGS, SEEDS
from weightbank.sweep import (
    LONG_STEPS,
    RATE_GRID,
    check_runs,
    chosen,
    first_runs,
    long_runs,
    planned,
    rate_sweep,
    sweep_plan,
    sweep_runs,
)
from weightbank.train import Plan, Run

ARM = "ff-routed"
SWEPT = tuple(name for name in CONFIGS if name not in ("own-chunk", "plain-w128"))


@pytest.fixture(autouse=True)
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(train, "RUNS", tmp_path)


def record(plan: Plan, loss: float) -> None:
    run = Run(plan.config, plan.seed, plan.steps, plan.peak_rate, 1, loss, 9.0 - loss, 0.0, None, None, [])
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(json.dumps(asdict(run)))


def sweep(config: str, losses: dict[float, float]) -> None:
    for rate, loss in losses.items():
        record(sweep_plan(config, rate), loss)


def test_a_run_file_reads_back_under_the_path_of_its_plan() -> None:
    plan = Plan(ARM, 2, 32000, 5e-4)
    record(plan, 1.25)
    assert plan.path.name == "ff-routed-lr0.0005-s2.json"
    assert plan.path.parent.name == "32000"
    assert plan.read().validation_loss == 1.25
    assert plan.command == [ARM, "--seed", "2", "--steps", "32000", "--rate", "0.0005"]


def test_every_grid_rate_survives_the_command_line_and_the_file_name() -> None:
    for rate in RATE_GRID:
        plan = Plan(ARM, 0, 8000, rate)
        assert float(plan.command[-1]) == rate
        assert float(plan.path.name.removeprefix(f"{ARM}-lr").removesuffix("-s0.json")) == rate


@pytest.mark.parametrize(
    "other",
    [Plan("attn-routed", 0, 8000, 1e-3), Plan(ARM, 1, 8000, 1e-3), Plan(ARM, 0, 32000, 1e-3), Plan(ARM, 0, 8000, 2e-3)],
)
def test_a_run_file_of_another_plan_is_refused(other: Plan) -> None:
    plan = Plan(ARM, 0, 8000, 1e-3)
    record(plan, 1.0)
    other.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.rename(other.path)
    with pytest.raises(ValueError):
        other.read()


def test_the_sweep_is_the_first_budget_at_seed_zero() -> None:
    assert sweep_plan(ARM, 1e-3) == Plan(ARM, 0, 8000, 1e-3)
    assert sweep_plan(ARM, 1e-3) in first_runs()


def test_the_first_experiment_is_every_configuration_at_one_rate() -> None:
    assert first_runs() == [Plan(config, seed, 8000, 1e-3) for seed in SEEDS for config in CONFIGS]


def test_no_rate_is_chosen_before_every_candidate_has_run() -> None:
    sweep(ARM, {5e-4: 1.2, 1e-3: 1.1})
    assert rate_sweep(ARM) == {5e-4: 1.2, 1e-3: 1.1, 2e-3: None}
    assert chosen(ARM) is None


def test_the_lowest_loss_inside_the_tried_rates_is_chosen() -> None:
    sweep(ARM, {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.15})
    assert rate_sweep(ARM) == {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.15}
    assert chosen(ARM) == 1e-3


def test_a_lowest_loss_on_the_edge_adds_the_next_rate_until_it_is_bracketed() -> None:
    sweep(ARM, {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.05})
    assert list(rate_sweep(ARM).items())[-1] == (4e-3, None)
    assert chosen(ARM) is None
    assert sweep_plan(ARM, 4e-3) in planned()
    sweep(ARM, {4e-3: 1.3})
    assert chosen(ARM) == 2e-3


def test_a_lowest_loss_on_the_low_edge_adds_a_lower_rate() -> None:
    sweep(ARM, {5e-4: 1.0, 1e-3: 1.1, 2e-3: 1.2})
    assert list(rate_sweep(ARM).items())[0] == (2.5e-4, None)
    sweep(ARM, {2.5e-4: 1.3})
    assert chosen(ARM) == 5e-4


@pytest.mark.parametrize("end", [RATE_GRID[0], RATE_GRID[-1]])
def test_a_lowest_loss_at_the_end_of_the_grid_is_refused(end: float) -> None:
    sweep(ARM, {rate: 1.0 + abs(RATE_GRID.index(rate) - RATE_GRID.index(end)) for rate in RATE_GRID})
    with pytest.raises(ValueError):
        rate_sweep(ARM)


def test_a_sweep_loss_that_is_not_finite_is_refused() -> None:
    sweep(ARM, {5e-4: 1.2, 1e-3: float("nan"), 2e-3: 1.15})
    with pytest.raises(ValueError):
        rate_sweep(ARM)


def test_only_the_compared_configurations_are_swept() -> None:
    assert sweep_runs() == [sweep_plan(config, rate) for config in SWEPT for rate in (5e-4, 1e-3, 2e-3)]


def test_long_runs_are_planned_only_at_a_chosen_rate() -> None:
    assert long_runs() == []
    sweep(ARM, {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.05, 4e-3: 1.3})
    assert long_runs() == [Plan(ARM, seed, 32000, 2e-3) for seed in SEEDS]
    assert set(long_runs()) <= set(planned())


def test_the_lower_rate_check_waits_for_every_long_run() -> None:
    for config in SWEPT:
        sweep(config, {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.05, 4e-3: 1.3})
    assert len(long_runs()) == len(SWEPT) * len(SEEDS)
    for plan in long_runs()[:-1]:
        record(plan, 1.0)
    assert check_runs() == []
    record(long_runs()[-1], 1.0)
    assert check_runs() == [Plan("attn-routed", 0, LONG_STEPS, 1e-3), Plan("both-routed", 0, LONG_STEPS, 1e-3)]
    assert set(check_runs()) <= set(planned())
