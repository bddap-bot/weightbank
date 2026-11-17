from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from weightbank.configs import CONFIGS, OWN_CHUNK, SEEDS
from weightbank.train import Plan

FIRST_STEPS = 8000
FIRST_RATE = 1e-3
LONG_STEPS = 4 * FIRST_STEPS
RATE_GRID = (1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
FIRST_CANDIDATES = (5e-4, 1e-3, 2e-3)
SWEEP_STEPS = FIRST_STEPS
SWEEP_SEED = 0
ARMS = tuple(name for name in CONFIGS if name != OWN_CHUNK)


def sweep_plan(config: str, rate: float) -> Plan:
    return Plan(config, SWEEP_SEED, SWEEP_STEPS, rate)


def rate_sweep(config: str) -> tuple[list[float], dict[float, float] | None]:
    rates = list(FIRST_CANDIDATES)
    while True:
        plans = [sweep_plan(config, rate) for rate in rates]
        if not all(plan.path.exists() for plan in plans):
            return rates, None
        losses = {rate: plan.read().validation_loss for rate, plan in zip(rates, plans)}
        best = RATE_GRID.index(min(losses, key=losses.__getitem__))
        beyond = [
            RATE_GRID[index]
            for index in (best - 1, best + 1)
            if 0 <= index < len(RATE_GRID) and RATE_GRID[index] not in losses
        ]
        if not beyond:
            return rates, dict(sorted(losses.items()))
        rates += beyond


def chosen(config: str) -> float | None:
    _, losses = rate_sweep(config)
    return None if losses is None else min(losses, key=losses.__getitem__)


def first() -> list[Plan]:
    return [Plan(config, seed, FIRST_STEPS, FIRST_RATE) for seed in SEEDS for config in CONFIGS]


def long() -> list[Plan]:
    rates = {config: chosen(config) for config in ARMS}
    return [
        Plan(config, seed, LONG_STEPS, rate)
        for seed in SEEDS
        for config, rate in rates.items()
        if rate is not None
    ]


def planned() -> list[Plan]:
    sweep = [sweep_plan(config, rate) for config in ARMS for rate in rate_sweep(config)[0]]
    return [*first(), *sweep, *long()]


def run(plan: Plan) -> None:
    subprocess.run([sys.executable, "-m", "weightbank.train", *plan.command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train every planned run that runs/ lacks.")
    parser.add_argument("--jobs", type=int, default=1, help="runs trained at once on the GPU")
    arguments = parser.parse_args()
    while missing := [plan for plan in dict.fromkeys(planned()) if not plan.path.exists()]:
        with ThreadPoolExecutor(arguments.jobs) as pool:
            list(pool.map(run, missing))


if __name__ == "__main__":
    main()
