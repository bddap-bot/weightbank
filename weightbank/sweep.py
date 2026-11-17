from __future__ import annotations

import math
import subprocess
import sys

from weightbank.configs import CONFIGS, OWN_CHUNK, SEEDS
from weightbank.train import Plan

FIRST_STEPS = 8000
FIRST_RATE = 1e-3
LONG_STEPS = 4 * FIRST_STEPS
RATE_GRID = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
FIRST_CANDIDATES = RATE_GRID[2:5]
SWEEP_SEED = 0
BELOW_EVERY_ARM = "plain-w128"
ARMS = tuple(name for name in CONFIGS if name not in (OWN_CHUNK, BELOW_EVERY_ARM))
LOWER_RATE_CHECK = ("attn-routed", "both-routed")


def sweep_plan(config: str, rate: float) -> Plan:
    return Plan(config, SWEEP_SEED, FIRST_STEPS, rate)


def rate_sweep(config: str) -> dict[float, float | None]:
    rates = list(FIRST_CANDIDATES)
    while True:
        plans = {rate: sweep_plan(config, rate) for rate in sorted(rates)}
        losses = {rate: plan.read().validation_loss if plan.path.exists() else None for rate, plan in plans.items()}
        finished = {rate: loss for rate, loss in losses.items() if loss is not None}
        if len(finished) < len(losses):
            return losses
        if not all(math.isfinite(loss) for loss in finished.values()):
            raise ValueError(f"{config}: a sweep run ended with a loss that is not finite")
        best = RATE_GRID.index(min(finished, key=finished.__getitem__))
        if best in (0, len(RATE_GRID) - 1):
            raise ValueError(f"{config}: the lowest loss is at the end of the rate grid, {RATE_GRID[best]:g}")
        beyond = [RATE_GRID[index] for index in (best - 1, best + 1) if RATE_GRID[index] not in losses]
        if not beyond:
            return losses
        rates += beyond


def chosen(config: str) -> float | None:
    losses = rate_sweep(config)
    finished = {rate: loss for rate, loss in losses.items() if loss is not None}
    if len(finished) < len(losses):
        return None
    return min(finished, key=finished.__getitem__)


def first_runs() -> list[Plan]:
    return [Plan(config, seed, FIRST_STEPS, FIRST_RATE) for seed in SEEDS for config in CONFIGS]


def sweep_runs() -> list[Plan]:
    return [sweep_plan(config, rate) for config in ARMS for rate in rate_sweep(config)]


def long_runs() -> list[Plan]:
    rates = {config: chosen(config) for config in ARMS}
    return [
        Plan(config, seed, LONG_STEPS, rate)
        for seed in SEEDS
        for config, rate in rates.items()
        if rate is not None
    ]


def lower_rate_check(config: str) -> Plan | None:
    rate = chosen(config)
    return None if rate is None else Plan(config, SWEEP_SEED, LONG_STEPS, RATE_GRID[RATE_GRID.index(rate) - 1])


def check_runs() -> list[Plan]:
    long = long_runs()
    if len(long) < len(ARMS) * len(SEEDS) or not all(plan.path.exists() for plan in long):
        return []
    return [plan for plan in map(lower_rate_check, LOWER_RATE_CHECK) if plan is not None]


def planned() -> list[Plan]:
    return list(dict.fromkeys([*first_runs(), *sweep_runs(), *long_runs(), *check_runs()]))


def main() -> None:
    while missing := [plan for plan in planned() if not plan.path.exists()]:
        for plan in missing:
            subprocess.run([sys.executable, "-m", "weightbank.train", *plan.command], check=True)


if __name__ == "__main__":
    main()
