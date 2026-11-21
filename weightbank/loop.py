from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from weightbank import report, train
from weightbank.configs import model_config
from weightbank.ledger import ABANDONED, OPEN, SIDES, STEPS, Cell, Ledger
from weightbank.sweep import Stuck, least_steps, protocol

FREE_MIB = 4096
POLL_SECONDS = 60


@dataclass
class Choice:
    hypothesis: dict[str, Any]
    comparison: dict[str, Any]
    hours: float
    passed_over: list[str]


def plans(ledger: Ledger, cell: Cell) -> list[train.Plan]:
    steps = cell[STEPS]
    assert isinstance(steps, int)
    return protocol(ledger.name(cell), steps)


def seconds_per_step(ledger: Ledger, cell: Cell) -> float:
    def distance(other: Cell) -> int:
        return sum(cell[axis.name] != other[axis.name] for axis in ledger.model_axes)

    nearest = min(distance(row.cell) for row in ledger.rows)
    return max(row.seconds_per_step for row in ledger.rows if distance(row.cell) == nearest)


def hours(ledger: Ledger, cell: Cell) -> float:
    if complete(ledger, cell):
        return 0.0
    steps = cell[STEPS]
    assert isinstance(steps, int)
    planned = plans(ledger, cell)
    trained = sum(plan.steps for plan in planned if plan.path.exists())
    missing = sum(plan.steps for plan in planned if not plan.path.exists())
    return max(least_steps(steps) - trained, missing) * seconds_per_step(ledger, cell) / 3600


def sides(ledger: Ledger, comparison: dict[str, Any]) -> list[Cell]:
    return [ledger.cell(comparison[side]) for side in SIDES]


def unfinished(hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        comparison
        for comparison in hypothesis["settle"]
        if "differences" not in comparison and ABANDONED not in comparison
    ]


def choose(ledger: Ledger, budget: float) -> Choice | None:
    passed_over: list[str] = []
    for hypothesis in sorted(ledger.hypotheses, key=lambda hypothesis: int(hypothesis["priority"])):
        unsettled = unfinished(hypothesis)
        if hypothesis["status"] != OPEN or not unsettled:
            continue
        costs = {}
        for index, comparison in enumerate(unsettled):
            try:
                costs[index] = sum(hours(ledger, cell) for cell in sides(ledger, comparison))
            except Stuck as error:
                passed_over.append(f"{hypothesis['id']} (priority {hypothesis['priority']}, a sweep that cannot go on: {error})")
        if not costs:
            continue
        cheapest = min(costs, key=costs.__getitem__)
        if costs[cheapest] <= budget:
            return Choice(hypothesis, unsettled[cheapest], costs[cheapest], passed_over)
        passed_over.append(f"{hypothesis['id']} (priority {hypothesis['priority']}, {costs[cheapest]:.1f} GPU-hours)")
    return None


def why(ledger: Ledger, choice: Choice, budget: float) -> str:
    lower, higher = (ledger.key(cell) for cell in sides(ledger, choice.comparison))
    text = (
        f"{choice.hypothesis['id']} (priority {choice.hypothesis['priority']}) is the highest-priority open hypothesis "
        f"with an unfinished comparison that fits {budget:g} GPU-hours; the cheapest is {lower} against {higher}, "
        f"estimated at {choice.hours:.1f} GPU-hours."
    )
    if choice.passed_over:
        text += " Passed over: " + "; ".join(choice.passed_over) + "."
    return text


def complete(ledger: Ledger, cell: Cell) -> bool:
    row = ledger.row(cell)
    return row is not None and row.losses is not None


def lacking(ledger: Ledger, entry: dict[str, Any]) -> list[train.Plan]:
    ledger.refresh(train.RUNS)
    return [
        plan
        for cell in reversed(sides(ledger, entry))
        if not complete(ledger, cell)
        for plan in plans(ledger, cell)
        if not plan.path.exists()
    ]


def wait_for_gpu() -> None:
    while True:
        free = int(
            subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.split()[0]
        )
        if free >= FREE_MIB:
            return
        print(f"{free} MiB of GPU memory free, waiting for {FREE_MIB}", flush=True)
        time.sleep(POLL_SECONDS)


def begin(ledger: Ledger, budget: float) -> dict[str, Any] | None:
    choice = choose(ledger, budget)
    if choice is None:
        return None
    for cell in sides(ledger, choice.comparison):
        model_config(cell)
    entry = {
        "hypothesis": choice.hypothesis["id"],
        "lower": choice.comparison["lower"],
        "higher": choice.comparison["higher"],
        "budget_gpu_hours": budget,
        "estimated_gpu_hours": round(choice.hours, 1),
        "why": why(ledger, choice, budget),
    }
    ledger.rounds.append(entry)
    ledger.write()
    return entry


def finish(ledger: Ledger, entry: dict[str, Any], stuck: str | None = None) -> None:
    known = {run.file for row in ledger.rows for run in row.runs}
    before = {hypothesis["id"]: hypothesis["status"] for hypothesis in ledger.hypotheses}
    ledger.refresh(train.RUNS)
    made = [run for row in ledger.rows for run in row.runs if run.file not in known]
    changed = {
        hypothesis["id"]: [before[hypothesis["id"]], hypothesis["status"]]
        for hypothesis in ledger.hypotheses
        if hypothesis["status"] != before[hypothesis["id"]]
    }
    entry["runs"] = [run.file for run in made]
    entry["rows"] = sorted({ledger.key(row.cell) for row in ledger.rows for run in row.runs if run.file not in known})
    entry["gpu_hours"] = round(sum(run.seconds for run in made) / 3600, 1)
    entry["statuses"] = changed
    named = [
        comparison
        for hypothesis in ledger.hypotheses
        if hypothesis["id"] == entry["hypothesis"]
        for comparison in hypothesis["settle"]
        if sides(ledger, comparison) == sides(ledger, entry)
    ]
    if not named:
        raise ValueError(f"the unfinished round's comparison is no longer under {entry['hypothesis']} in the ledger")
    comparison = named[0]
    if "differences" in comparison:
        lower, higher = (ledger.row(cell) for cell in sides(ledger, entry))
        assert lower is not None and higher is not None and lower.losses is not None and higher.losses is not None
        entry["differences"] = comparison["differences"]
        entry["reading"] = (
            f"{ledger.key(lower.cell)} reached {fmean(lower.losses):.4f} nats per byte with {lower.stored_weights:,} "
            f"stored weights at rate {lower.rate:g}; {ledger.key(higher.cell)} reached {fmean(higher.losses):.4f} with "
            f"{higher.stored_weights:,} at rate {higher.rate:g}. Lower minus higher by seed: "
            + ", ".join(f"{difference:+.4f}" for difference in comparison["differences"])
            + ". "
            + ("; ".join(f"{name} went from {was} to {now}" for name, (was, now) in changed.items()) or "No status changed")
            + "."
        )
    else:
        comparison[ABANDONED] = stuck or "its runs ended without three finite seeds at one rate on both sides"
        entry["reading"] = f"The comparison is abandoned: {comparison[ABANDONED]}. No round picks it again while the ledger marks it so."
    ledger.write()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one round: pick a cell by the ledger's hypotheses, train it, record it.")
    parser.add_argument("--gpu-hours", type=float, required=True, help="the most this round may be estimated to cost")
    arguments = parser.parse_args()
    ledger = Ledger.read()
    entry = ledger.rounds[-1] if ledger.rounds and "reading" not in ledger.rounds[-1] else None
    if entry is None:
        ledger.refresh(train.RUNS)
        entry = begin(ledger, arguments.gpu_hours)
    if entry is None:
        print(f"no open hypothesis has an unfinished comparison that fits {arguments.gpu_hours:g} GPU-hours")
        return
    print(json.dumps(entry, indent=1), flush=True)
    stuck = None
    while stuck is None:
        try:
            missing = lacking(ledger, entry)
        except Stuck as error:
            stuck = f"its rate sweep cannot go on: {error}"
            break
        if not missing:
            break
        wait_for_gpu()
        subprocess.run([sys.executable, "-m", "weightbank.train", *missing[0].command], check=True)
    ledger = Ledger.read()
    finish(ledger, ledger.rounds[-1], stuck)
    report.main()


if __name__ == "__main__":
    main()
