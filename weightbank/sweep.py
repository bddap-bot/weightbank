from __future__ import annotations

import math

from weightbank.configs import default, rate_source
from weightbank.ledger import SEEDS, STEPS
from weightbank.train import Plan

FIRST_STEPS = default(STEPS)
FIRST_RATE = 1e-3
LONG_STEPS = 4 * FIRST_STEPS
RATE_GRID = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
SHORT_RATES = RATE_GRID[2:5]
SWEEP_SEED = 0
STEEP = 0.005

Sweep = dict[float, float | None]


class Stuck(ValueError):
    pass


def read(plans: dict[float, Plan]) -> Sweep:
    losses = {rate: plan.read().validation_loss if plan.path.exists() else None for rate, plan in sorted(plans.items())}
    if any(loss is not None and math.isnan(loss) for loss in losses.values()):
        raise Stuck(f"a sweep run of {next(iter(plans.values())).config} ended with a loss that is not a number")
    return losses


def finished(sweep: Sweep) -> dict[float, float] | None:
    losses = {rate: loss for rate, loss in sweep.items() if loss is not None}
    return losses if len(losses) == len(sweep) else None


def lowest(losses: dict[float, float]) -> float:
    return min(losses, key=losses.__getitem__)


def short_plan(config: str, rate: float) -> Plan:
    return Plan(config, SWEEP_SEED, FIRST_STEPS, rate)


def long_plan(config: str, rate: float, steps: int = LONG_STEPS) -> Plan:
    return Plan(config, SWEEP_SEED, steps, rate)


def short_sweep(config: str) -> Sweep:
    return read({rate: short_plan(config, rate) for rate in SHORT_RATES})


def neighbours(rate: float) -> list[float]:
    index = RATE_GRID.index(rate)
    if index in (0, len(RATE_GRID) - 1):
        raise Stuck(f"the rate grid ends at {rate:g}")
    return [RATE_GRID[index - 1], RATE_GRID[index + 1]]


def long_sweep(config: str, steps: int = LONG_STEPS) -> Sweep | None:
    short = finished(short_sweep(config))
    if short is None:
        return None
    short_best = lowest(short)
    rates = [short_best, lowest({rate: short[rate] for rate in neighbours(short_best) if rate in short})]
    while True:
        sweep = read({rate: long_plan(config, rate, steps) for rate in rates})
        losses = finished(sweep)
        if losses is None:
            return sweep
        best = lowest(losses)
        lead = min(loss for rate, loss in losses.items() if rate != best) - losses[best]
        if best in (min(losses), max(losses)) and (best != short_best or lead > STEEP):
            rates += [rate for rate in neighbours(best) if rate not in losses]
        else:
            return sweep


def chosen(config: str, steps: int = LONG_STEPS) -> float | None:
    sweep = long_sweep(config, steps)
    losses = None if sweep is None else finished(sweep)
    return None if losses is None else lowest(losses)


def protocol(config: str, steps: int) -> list[Plan]:
    source = rate_source(config)
    if source != config:
        rate = FIRST_RATE if steps == FIRST_STEPS else chosen(source, steps)
        return protocol(source, steps) if rate is None else [Plan(config, seed, steps, rate) for seed in SEEDS]
    plans = [short_plan(config, rate) for rate in SHORT_RATES]
    if steps == FIRST_STEPS:
        plans += [Plan(config, seed, FIRST_STEPS, FIRST_RATE) for seed in SEEDS]
    else:
        rate = chosen(config, steps)
        plans += [long_plan(config, swept, steps) for swept in long_sweep(config, steps) or ()]
        plans += [] if rate is None else [Plan(config, seed, steps, rate) for seed in SEEDS]
    return list(dict.fromkeys(plans))


def least_steps(config: str, steps: int) -> int:
    if rate_source(config) != config:
        return len(SEEDS) * steps
    short = len(SHORT_RATES) * FIRST_STEPS
    if steps == FIRST_STEPS:
        return short + (len(SEEDS) - 1) * FIRST_STEPS
    return short + (2 + len(SEEDS) - 1) * steps
