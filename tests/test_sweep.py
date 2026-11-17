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
    chosen,
    first_runs,
    long_plan,
    long_runs,
    long_sweep,
    long_sweep_runs,
    planned,
    short_plan,
    short_runs,
    short_sweep,
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


def short(config: str, losses: dict[float, float]) -> None:
    for rate, loss in losses.items():
        record(short_plan(config, rate), loss)


def long(config: str, losses: dict[float, float]) -> None:
    for rate, loss in losses.items():
        record(long_plan(config, rate), loss)


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


def test_the_short_sweep_is_the_first_budget_at_seed_zero() -> None:
    assert short_plan(ARM, 1e-3) == Plan(ARM, 0, 8000, 1e-3)
    assert short_plan(ARM, 1e-3) in first_runs()
    assert long_plan(ARM, 1e-3) == Plan(ARM, 0, 32000, 1e-3)
    assert LONG_STEPS == 32000


def test_the_first_experiment_is_every_configuration_at_one_rate() -> None:
    assert first_runs() == [Plan(config, seed, 8000, 1e-3) for seed in SEEDS for config in CONFIGS]


def test_only_the_compared_configurations_are_swept() -> None:
    assert short_runs() == [short_plan(config, rate) for config in SWEPT for rate in (5e-4, 1e-3, 2e-3)]
    assert set(short_runs()) <= set(planned())


def test_no_long_run_is_planned_before_the_short_sweep_has_run() -> None:
    short(ARM, {5e-4: 1.2, 1e-3: 1.1})
    assert short_sweep(ARM) == {5e-4: 1.2, 1e-3: 1.1, 2e-3: None}
    assert long_sweep(ARM) is None
    assert long_sweep_runs() == [] and long_runs() == []


@pytest.mark.parametrize(
    "losses, candidates",
    [
        ({5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1}, [1e-3, 2e-3]),
        ({5e-4: 1.3, 1e-3: 1.1, 2e-3: 1.2}, [1e-3, 2e-3]),
        ({5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.3}, [5e-4, 1e-3]),
        ({5e-4: 1.1, 1e-3: 1.2, 2e-3: 1.3}, [5e-4, 1e-3]),
    ],
)
def test_the_long_sweep_starts_at_the_best_short_rate_and_its_better_neighbour(
    losses: dict[float, float], candidates: list[float]
) -> None:
    short(ARM, losses)
    assert long_sweep(ARM) == dict.fromkeys(candidates)
    assert long_sweep_runs() == [long_plan(ARM, rate) for rate in candidates]
    assert set(long_sweep_runs()) <= set(planned())
    assert chosen(ARM) is None


def test_the_lower_long_loss_is_chosen_and_the_other_seeds_follow_it() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.004})
    assert chosen(ARM) is None and long_runs() == []
    long(ARM, {2e-3: 1.0})
    assert chosen(ARM) == 2e-3
    assert long_runs() == [Plan(ARM, seed, 32000, 2e-3) for seed in SEEDS]
    assert set(long_runs()) <= set(planned())
    long(ARM, {1e-3: 0.996})
    assert chosen(ARM) == 1e-3


def test_a_steep_lead_on_the_edge_adds_the_rate_beyond_it() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.006, 2e-3: 1.0})
    assert long_sweep(ARM) == {1e-3: 1.006, 2e-3: 1.0, 4e-3: None}
    assert chosen(ARM) is None
    long(ARM, {4e-3: 0.99})
    assert long_sweep(ARM) == {1e-3: 1.006, 2e-3: 1.0, 4e-3: 0.99, 8e-3: None}
    long(ARM, {4e-3: 0.999})
    assert chosen(ARM) == 4e-3


def test_a_steep_lead_on_the_low_edge_adds_a_lower_rate() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.0, 2e-3: 1.006})
    assert long_sweep(ARM) == {5e-4: None, 1e-3: 1.0, 2e-3: 1.006}
    long(ARM, {5e-4: 1.1})
    assert chosen(ARM) == 1e-3


def test_a_steep_lead_at_the_end_of_the_grid_is_refused() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.3, 2e-3: 1.2, 4e-3: 1.1, 8e-3: 1.0})
    with pytest.raises(ValueError):
        long_sweep(ARM)


def test_a_sweep_loss_that_is_not_finite_is_refused() -> None:
    short(ARM, {5e-4: 1.2, 1e-3: float("nan"), 2e-3: 1.15})
    with pytest.raises(ValueError):
        short_sweep(ARM)
