from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from weightbank import loop, report, train
from weightbank.ledger import OPEN, PATH, REFUTED, STEPS, SUPPORTED, Cell, Ledger
from weightbank.train import Plan, Run

REPO = Path.cwd()
SHARED: Cell = {"shared": "both", STEPS: 32000}
CORRECTED: Cell = {"shared": "both", "chunks": 1, "rank": 24, STEPS: 32000}
TWO: Cell = {"shared": "both", "chunks": 2, STEPS: 32000}
LONGEST: Cell = {"shared": "both", "pick": "routed", STEPS: 128000}


def hypothesis(name: str, priority: int, *settle: tuple[Cell, Cell]) -> dict[str, Any]:
    return {
        "id": name,
        "priority": priority,
        "status": OPEN,
        "statement": "",
        "settle": [{"lower": lower, "higher": higher} for lower, higher in settle],
    }


def record(plan: Plan, loss: float, seconds: float = 0.0) -> None:
    run = Run(plan.config, plan.seed, plan.steps, plan.peak_rate, 7, loss, loss, seconds, None, None, [])
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(json.dumps(asdict(run)))


@pytest.fixture(autouse=True)
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for seed in (0, 1, 2):
        record(Plan("both-c2", seed, 32000, 2e-3), 1.2 + seed / 100, seconds=3200.0)
    for rate in (5e-4, 1e-3, 2e-3):
        record(Plan("both-c2", 0, 8000, rate), 1.3 + rate, seconds=800.0)
    record(Plan("both-c2", 0, 32000, 1e-3), 1.3, seconds=3200.0)


def ledger(*hypotheses: dict[str, Any]) -> Ledger:
    fields = json.loads((REPO / PATH).read_text())
    made = Ledger({**fields, "hypotheses": list(hypotheses), "rounds": [], "cells": []})
    made.refresh(train.RUNS)
    return made


def test_the_cost_of_a_cell_is_its_protocol_at_the_speed_of_the_nearest_tried_cell() -> None:
    record(Plan("full", 0, 8000, 1e-3), 1.3, seconds=4000.0)
    made = ledger()
    assert loop.seconds_per_step(made, made.cell(CORRECTED)) == 0.1
    record(Plan("both-c2", 1, 8000, 1e-3), 1.3, seconds=1600.0)
    assert loop.seconds_per_step(ledger(), made.cell(CORRECTED)) == 0.2
    (train.RUNS / "8000" / "both-c2-lr0.001-s1.json").unlink()
    assert loop.hours(made, made.cell(TWO)) == 0.0
    assert loop.hours(made, made.cell(CORRECTED)) == pytest.approx((3 * 8000 + 4 * 32000) * 0.1 / 3600)
    record(Plan("both-c1-r24", 0, 8000, 5e-4), 1.5)
    assert loop.hours(made, made.cell(CORRECTED)) == pytest.approx((2 * 8000 + 4 * 32000) * 0.1 / 3600)


def test_the_highest_priority_open_hypothesis_that_fits_is_chosen_at_its_cheapest_comparison() -> None:
    made = ledger(
        hypothesis("costly", 1, (LONGEST, SHARED)),
        hypothesis("settled", 1, (TWO, TWO)),
        hypothesis("nothing-to-run", 1),
        hypothesis("later", 3, (SHARED, TWO)),
        hypothesis("wanted", 2, ({**CORRECTED, STEPS: 128000}, TWO), (CORRECTED, TWO)),
        hypothesis("tied", 2, (SHARED, TWO)),
    )
    choice = loop.choose(made, 12)
    assert choice is not None
    assert (choice.hypothesis["id"], choice.comparison["lower"]) == ("wanted", CORRECTED)
    assert choice.hours == pytest.approx(4.2, abs=0.05)
    assert len(choice.passed_over) == 1 and choice.passed_over[0].startswith("costly (priority 1, ")
    assert "wanted (priority 2)" in loop.why(made, choice, 12) and "32000/both-c1-r24 against 32000/both-c2" in loop.why(made, choice, 12)
    assert loop.choose(made, 4) is None
    made.hypotheses[4]["settle"][1]["abandoned"] = "said so"
    assert loop.choose(made, 12).hypothesis["id"] == "tied"  # type: ignore[union-attr]
    del made.hypotheses[4]["settle"][1]["abandoned"]
    for row in made.hypotheses:
        row["priority"] = {"later": 1}.get(row["id"], 5)
    cheaper = loop.choose(made, 5)
    assert cheaper is not None and cheaper.hypothesis["id"] == "later"


def test_a_round_trains_what_the_comparison_lacks_and_records_what_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    trained: list[Plan] = []

    def fake(command: list[str], check: bool, **_: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="8000\n")
        plan = Plan(command[3], int(command[5]), int(command[7]), float(command[9]))
        trained.append(plan)
        record(plan, 1.1 + abs(plan.peak_rate - 1e-3) + plan.seed / 100 + (plan.steps == 8000), seconds=1800.0)
        return subprocess.CompletedProcess(command, 0)

    reported: list[bool] = []
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(report, "main", lambda: reported.append(True))
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    ledger(hypothesis("wanted", 1, (CORRECTED, TWO)), hypothesis("other", 2, (TWO, CORRECTED))).write()
    loop.main()
    assert trained == [
        *(Plan("both-c1-r24", 0, 8000, rate) for rate in (5e-4, 1e-3, 2e-3)),
        Plan("both-c1-r24", 0, 32000, 5e-4),
        Plan("both-c1-r24", 0, 32000, 1e-3),
        Plan("both-c1-r24", 1, 32000, 1e-3),
        Plan("both-c1-r24", 2, 32000, 1e-3),
    ]
    after = Ledger.read()
    (entry,) = after.rounds
    assert entry["hypothesis"] == "wanted" and entry["lower"] == CORRECTED and entry["higher"] == TWO
    assert entry["runs"] == sorted(plan.path.as_posix() for plan in trained)
    assert entry["rows"] == ["32000/both-c1-r24", "8000/both-c1-r24"]
    assert entry["gpu_hours"] == 3.5 and entry["estimated_gpu_hours"] == 4.2
    assert entry["differences"] == [-0.1, -0.1, -0.1]
    assert entry["statuses"] == {"wanted": [OPEN, SUPPORTED], "other": [OPEN, REFUTED]}
    assert "32000/both-c1-r24 reached 1.1100" in entry["reading"] and "wanted went from open to supported" in entry["reading"]
    assert [row["status"] for row in after.hypotheses] == [SUPPORTED, REFUTED]
    assert reported == [True]
    loop.main()
    assert len(Ledger.read().rounds) == 1 and len(trained) == 7


def test_a_comparison_that_cannot_finish_is_abandoned_and_the_next_round_moves_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(command: list[str], check: bool, **_: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="8000\n")
        plan = Plan(command[3], int(command[5]), int(command[7]), float(command[9]))
        diverged = plan.config == "both-c1-r24" and plan.seed == 2
        record(plan, float("inf") if diverged else 1.1 + abs(plan.peak_rate - 1e-3) + (plan.steps == 8000), seconds=1800.0)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(report, "main", lambda: None)
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    ledger(hypothesis("wanted", 1, (CORRECTED, TWO)), hypothesis("later", 2, (SHARED, TWO))).write()
    loop.main()
    after = Ledger.read()
    assert "abandoned" in after.rounds[0]["reading"] and "differences" not in after.rounds[0]
    assert "abandoned" in after.hypotheses[0]["settle"][0] and after.hypotheses[0]["status"] == OPEN
    assert len(after.rounds[0]["runs"]) == 7 and after.rounds[0]["statuses"] == {}
    loop.main()
    assert [entry["hypothesis"] for entry in Ledger.read().rounds] == ["wanted", "later"]


def test_a_sweep_that_runs_off_the_rate_grid_abandons_the_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(command: list[str], check: bool, **_: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="8000\n")
        plan = Plan(command[3], int(command[5]), int(command[7]), float(command[9]))
        record(plan, 1.0 + plan.peak_rate * 100, seconds=1800.0)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(report, "main", lambda: None)
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    full: Cell = {"shared": "both", "pick": "fixed", "chunks": 1, "rank": 24, "width": 256, STEPS: 32000}
    made = ledger(hypothesis("wanted", 1, (full, TWO)))
    made.write()
    loop.main()
    after = Ledger.read()
    assert "the rate grid ends" in after.hypotheses[0]["settle"][0]["abandoned"]
    assert "abandoned" in after.rounds[0]["reading"] and after.hypotheses[0]["status"] == OPEN
    loop.main()
    assert len(Ledger.read().rounds) == 1
    again = Ledger.read()
    again.hypotheses.append(hypothesis("shares-the-cell", 2, (full, SHARED)))
    again.hypotheses.append(hypothesis("fine", 3, (SHARED, TWO)))
    again.write()
    loop.main()
    last = Ledger.read().rounds[-1]
    assert last["hypothesis"] == "fine" and "shares-the-cell (priority 2, a sweep that cannot go on" in last["why"]


def test_a_gpu_reading_that_is_no_number_stops_the_loop_without_abandoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda command, **_: subprocess.CompletedProcess(command, 0, stdout="[N/A]\n"))
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    ledger(hypothesis("wanted", 1, (CORRECTED, TWO))).write()
    with pytest.raises(ValueError):
        loop.main()
    after = Ledger.read()
    assert "reading" not in after.rounds[0] and "abandoned" not in after.hypotheses[0]["settle"][0]


def test_a_round_refuses_a_cell_the_model_cannot_build_before_recording_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    ledger(hypothesis("wrong", 1, ({"pick": "routed", STEPS: 32000}, TWO))).write()
    with pytest.raises(ValueError):
        loop.main()
    assert Ledger.read().rounds == []


def test_an_unfinished_round_is_resumed_instead_of_chosen_again(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(command: list[str], check: bool, **_: Any) -> subprocess.CompletedProcess[str]:
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout="8000\n")
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(report, "main", lambda: None)
    monkeypatch.setattr("sys.argv", ["loop", "--gpu-hours", "12"])
    made = ledger(hypothesis("wanted", 1, (CORRECTED, TWO)), hypothesis("first-now", 0, (SHARED, TWO)))
    made.hypotheses[1]["priority"] = 5
    made.write()
    with pytest.raises(KeyboardInterrupt):
        loop.main()
    waiting = Ledger.read()
    assert [entry["hypothesis"] for entry in waiting.rounds] == ["wanted"] and "reading" not in waiting.rounds[0]
    waiting.hypotheses[1]["priority"] = 0
    waiting.write()
    with pytest.raises(KeyboardInterrupt):
        loop.main()
    assert [entry["hypothesis"] for entry in Ledger.read().rounds] == ["wanted"]


def test_the_loop_waits_while_the_gpu_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    free = iter(["100\n", "3000\n", "5000\n"])
    slept: list[float] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_: subprocess.CompletedProcess(command, 0, stdout=next(free)))
    monkeypatch.setattr(time, "sleep", slept.append)
    loop.wait_for_gpu()
    assert slept == [60, 60]


