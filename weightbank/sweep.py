from __future__ import annotations

import math
import subprocess
import sys

from weightbank.configs import CONFIGS, FULL, OWN_CHUNK, SEEDS
from weightbank.train import Plan

FIRST_STEPS = 8000
FIRST_RATE = 1e-3
LONG_STEPS = 4 * FIRST_STEPS
RATE_GRID = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
SHORT_RATES = RATE_GRID[2:5]
SWEEP_SEED = 0
STEEP = 0.005
ARMS = tuple(name for name in CONFIGS if name != OWN_CHUNK)
RATE_FROM = {OWN_CHUNK: FULL}

Sweep = dict[float, float | None]


def read(plans: dict[float, Plan]) -> Sweep:
    losses = {rate: plan.read().validation_loss if plan.path.exists() else None for rate, plan in sorted(plans.items())}
    if any(loss is not None and math.isnan(loss) for loss in losses.values()):
        raise ValueError(f"a sweep run of {next(iter(plans.values())).config} ended with a loss that is not a number")
    return losses


def finished(sweep: Sweep) -> dict[float, float] | None:
    losses = {rate: loss for rate, loss in sweep.items() if loss is not None}
    return losses if len(losses) == len(sweep) else None


def lowest(losses: dict[float, float]) -> float:
    return min(losses, key=losses.__getitem__)


def short_plan(config: str, rate: float) -> Plan:
    return Plan(config, SWEEP_SEED, FIRST_STEPS, rate)


def long_plan(config: str, rate: float) -> Plan:
    return Plan(config, SWEEP_SEED, LONG_STEPS, rate)


def short_sweep(config: str) -> Sweep:
    return read({rate: short_plan(config, rate) for rate in SHORT_RATES})


def neighbours(rate: float) -> list[float]:
    index = RATE_GRID.index(rate)
    if index in (0, len(RATE_GRID) - 1):
        raise ValueError(f"the rate grid ends at {rate:g}")
    return [RATE_GRID[index - 1], RATE_GRID[index + 1]]


def long_sweep(config: str) -> Sweep | None:
    short = finished(short_sweep(config))
    if short is None:
        return None
    short_best = lowest(short)
    rates = [short_best, lowest({rate: short[rate] for rate in neighbours(short_best) if rate in short})]
    while True:
        sweep = read({rate: long_plan(config, rate) for rate in rates})
        losses = finished(sweep)
        if losses is None:
            return sweep
        best = lowest(losses)
        lead = min(loss for rate, loss in losses.items() if rate != best) - losses[best]
        if best in (min(losses), max(losses)) and (best != short_best or lead > STEEP):
            rates += [rate for rate in neighbours(best) if rate not in losses]
        else:
            return sweep


def chosen(config: str) -> float | None:
    sweep = long_sweep(RATE_FROM.get(config, config))
    losses = None if sweep is None else finished(sweep)
    return None if losses is None else lowest(losses)


def first_runs() -> list[Plan]:
    return [Plan(config, seed, FIRST_STEPS, FIRST_RATE) for seed in SEEDS for config in CONFIGS]


def short_runs() -> list[Plan]:
    return [short_plan(config, rate) for config in ARMS for rate in SHORT_RATES]


def long_sweep_runs() -> list[Plan]:
    return [long_plan(config, rate) for config in ARMS for rate in long_sweep(config) or ()]


def long_runs() -> list[Plan]:
    rates = {config: chosen(config) for config in CONFIGS}
    return [
        Plan(config, seed, LONG_STEPS, rate)
        for seed in SEEDS
        for config, rate in rates.items()
        if rate is not None
    ]


def planned() -> list[Plan]:
    return list(dict.fromkeys([*first_runs(), *short_runs(), *long_sweep_runs(), *long_runs()]))


def main() -> None:
    while missing := [plan for plan in planned() if not plan.path.exists()]:
        subprocess.run([sys.executable, "-m", "weightbank.train", *missing[0].command], check=True)


if __name__ == "__main__":
    main()
