from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from weightbank import train
from weightbank.ledger import SEEDS
from weightbank.sweep import (
    LONG_STEPS,
    RATE_GRID,
    chosen,
    least_steps,
    long_plan,
    long_sweep,
    protocol,
    short_plan,
    short_sweep,
)
from weightbank.train import Plan, Run

ARM = "ff-routed"
SHORT = [Plan(ARM, 0, 8000, rate) for rate in (5e-4, 1e-3, 2e-3)]


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




def test_a_run_of_a_name_the_axes_do_not_spell_is_refused() -> None:
    with pytest.raises(ValueError):
        Plan("ff-routed-twice", 0, 8000, 1e-3).path
    with pytest.raises(ValueError):
        Plan(ARM, 0, 9000, 1e-3).path


def test_the_short_sweep_is_the_first_budget_at_seed_zero() -> None:
    assert short_plan(ARM, 1e-3) == Plan(ARM, 0, 8000, 1e-3)
    assert long_plan(ARM, 1e-3, 128000) == Plan(ARM, 0, 128000, 1e-3)
    assert long_plan(ARM, 1e-3) == Plan(ARM, 0, 32000, 1e-3)
    assert LONG_STEPS == 32000


def test_the_first_budget_is_swept_at_seed_zero_and_run_at_one_rate_for_every_seed() -> None:
    assert protocol(ARM, 8000) == [*SHORT, Plan(ARM, 1, 8000, 1e-3), Plan(ARM, 2, 8000, 1e-3)]
    assert least_steps(ARM, 8000) == 5 * 8000


def test_a_longer_budget_starts_from_the_short_sweep() -> None:
    assert protocol(ARM, 32000) == SHORT == protocol(ARM, 128000)
    assert least_steps(ARM, 32000) == 3 * 8000 + 4 * 32000


def test_no_long_run_is_planned_before_the_short_sweep_has_run() -> None:
    short(ARM, {5e-4: 1.2, 1e-3: 1.1})
    assert short_sweep(ARM) == {5e-4: 1.2, 1e-3: 1.1, 2e-3: None}
    assert long_sweep(ARM) is None
    assert protocol(ARM, 32000) == SHORT


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
    assert protocol(ARM, 32000) == [*SHORT, *(long_plan(ARM, rate) for rate in candidates)]
    assert chosen(ARM) is None


def test_the_lower_long_loss_is_chosen_and_the_other_seeds_follow_it() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.004})
    assert chosen(ARM) is None and len(protocol(ARM, 32000)) == 5
    long(ARM, {2e-3: 1.0})
    assert chosen(ARM) == 2e-3
    assert protocol(ARM, 32000) == [
        *SHORT,
        long_plan(ARM, 1e-3),
        *(Plan(ARM, seed, 32000, 2e-3) for seed in SEEDS),
    ]


def test_a_lead_of_the_steep_margin_or_less_on_the_short_best_ends_the_sweep() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.005, 2e-3: 1.0})
    assert chosen(ARM) == 2e-3


def test_a_steep_lead_on_the_short_best_adds_the_rate_beyond_it() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.006, 2e-3: 1.0})
    assert long_sweep(ARM) == {1e-3: 1.006, 2e-3: 1.0, 4e-3: None}
    assert chosen(ARM) is None
    assert protocol(ARM, 32000)[-1] == long_plan(ARM, 4e-3)
    long(ARM, {4e-3: 1.2})
    assert chosen(ARM) == 2e-3


def test_a_winner_that_is_not_the_short_best_is_followed_until_bracketed() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.0, 2e-3: 1.001})
    assert long_sweep(ARM) == {5e-4: None, 1e-3: 1.0, 2e-3: 1.001}
    long(ARM, {5e-4: 0.999})
    assert long_sweep(ARM) == {2.5e-4: None, 5e-4: 0.999, 1e-3: 1.0, 2e-3: 1.001}
    long(ARM, {2.5e-4: 1.1})
    assert chosen(ARM) == 5e-4


def test_a_winner_past_the_short_best_is_followed_until_bracketed() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.1, 2e-3: 1.0, 4e-3: 0.999})
    assert long_sweep(ARM) == {1e-3: 1.1, 2e-3: 1.0, 4e-3: 0.999, 8e-3: None}
    long(ARM, {8e-3: float("inf")})
    assert chosen(ARM) == 4e-3


def test_a_winner_at_the_end_of_the_grid_cannot_be_bracketed() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.3, 2e-3: 1.2, 4e-3: 1.1, 8e-3: 1.0})
    with pytest.raises(ValueError):
        long_sweep(ARM)


def test_a_diverged_run_loses_and_a_loss_that_is_not_a_number_is_refused() -> None:
    short(ARM, {5e-4: 1.2, 1e-3: 1.1, 2e-3: float("inf")})
    assert long_sweep(ARM) == {5e-4: None, 1e-3: None}
    short(ARM, {2e-3: float("nan")})
    with pytest.raises(ValueError):
        short_sweep(ARM)


def test_each_budget_keeps_its_own_long_sweep() -> None:
    short(ARM, {5e-4: 1.3, 1e-3: 1.2, 2e-3: 1.1})
    long(ARM, {1e-3: 1.004, 2e-3: 1.0})
    assert chosen(ARM, 32000) == 2e-3
    assert chosen(ARM, 128000) is None
    assert protocol(ARM, 128000) == [*SHORT, long_plan(ARM, 1e-3, 128000), long_plan(ARM, 2e-3, 128000)]


def test_a_fixed_map_that_is_not_the_mod_map_trains_at_the_mod_maps_rate_with_no_sweep_of_its_own() -> None:
    mod, drawn, ends = "both-modern", "both-drawn-d1-modern", "both-ends-modern"
    assert protocol(drawn, 32000) == protocol(mod, 32000) == [short_plan(mod, rate) for rate in (5e-4, 1e-3, 2e-3)]
    short(mod, {5e-4: 1.2, 1e-3: 1.1, 2e-3: 1.0})
    long(mod, {2e-3: 0.9})
    assert protocol(ends, 32000) == protocol(mod, 32000) == [*protocol(mod, 8000)[:3], long_plan(mod, 1e-3), long_plan(mod, 2e-3)]
    long(mod, {1e-3: 0.904})
    for name in (drawn, ends):
        assert protocol(name, 32000) == [Plan(name, seed, 32000, 2e-3) for seed in SEEDS]
        assert protocol(name, 8000) == [Plan(name, seed, 8000, 1e-3) for seed in SEEDS]
        assert least_steps(name, 32000) == 3 * 32000
    assert protocol("both-c8", 8000) == [Plan("both-c8", seed, 8000, 1e-3) for seed in SEEDS]
    assert len(protocol(mod, 32000)) == 7
