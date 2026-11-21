from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from weightbank.compress import bank, lm
from weightbank.compress.heal import heal
from weightbank.data import windows

RUNS = Path("runs/compress")
STEPS = 2000
SWEEP_STEPS = 500
LADDER = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
FIRST_RATES = LADDER[1:5]
BITS = (16, 8, 4)
BANKS = ((8, 16), (8, 64), (15, 16), (15, 64), (23, 16), (23, 64))
SWEEP_BANK = (15, 64)
EXTRA_SEEDS = (1, 2)
ROUTED_BANK = (8, 64)
ROUTED_MICRO = 1

Kind = Literal["bank", "low-rank", "low-rank-kept", "routed"]
FIXED: tuple[Kind, ...] = ("bank", "low-rank", "low-rank-kept")


@dataclass(frozen=True)
class Trial:
    """A short run that tries one learning rate."""

    rate: float


@dataclass(frozen=True)
class Full:
    seed: int = 0


@dataclass(frozen=True)
class Arm:
    kind: Kind
    chunks: int
    rank: int
    budget: Trial | Full = Full()

    @property
    def path(self) -> Path:
        name = f"{self.kind}-n{self.chunks}-r{self.rank}"
        if isinstance(self.budget, Trial):
            return RUNS / "sweep" / f"{name}-lr{self.budget.rate:g}.json"
        return RUNS / (name + (f"-s{self.budget.seed}" if self.budget.seed else "") + ".json")

    @property
    def steps(self) -> int:
        return SWEEP_STEPS if isinstance(self.budget, Trial) else STEPS

    @property
    def rate(self) -> float:
        return self.budget.rate if isinstance(self.budget, Trial) else chosen_rate(self.kind)

    @property
    def seed(self) -> int:
        return 0 if isinstance(self.budget, Trial) else self.budget.seed

    def read(self) -> dict[str, Any]:
        record = read(self.path)
        planned = (self.kind, self.chunks, self.rank, self.seed, self.steps, self.rate)
        held = tuple(record[key] for key in ("kind", "chunks", "rank", "seed", "steps", "rate"))
        if held != planned:
            raise ValueError(f"{self.path} holds {held}, the plan is {planned}")
        return record


def read(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text())
    return record


def write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


def trials(kind: Kind) -> dict[float, Arm]:
    tried = {rate: Arm(kind, *SWEEP_BANK, Trial(rate)) for rate in LADDER}
    return {rate: arm for rate, arm in tried.items() if arm.path.exists()}


def sweep(kind: Kind) -> list[Arm]:
    """The learning rates still to try for a kind: the first four, then one more beyond
    whichever end of the tried range holds the best, until the best has a worse rate on both
    sides or the ladder ends."""
    done = list(trials(kind))
    missing = [Arm(kind, *SWEEP_BANK, Trial(rate)) for rate in FIRST_RATES if rate not in done]
    if missing:
        return missing
    best = LADDER.index(chosen_rate(kind))
    lowest, highest = LADDER.index(done[0]), LADDER.index(done[-1])
    if best == lowest and lowest > 0:
        return [Arm(kind, *SWEEP_BANK, Trial(LADDER[lowest - 1]))]
    if best == highest and highest < len(LADDER) - 1:
        return [Arm(kind, *SWEEP_BANK, Trial(LADDER[highest + 1]))]
    return []


def chosen_rate(kind: Kind) -> float:
    """The tried rate with the lowest validation perplexity; the routed bank takes the fixed
    bank's."""
    scores = {
        rate: float(arm.read()["validation_perplexity"])
        for rate, arm in trials("bank" if kind == "routed" else kind).items()
    }
    finite = {rate: score for rate, score in scores.items() if math.isfinite(score)}
    return min(finite, key=lambda rate: finite[rate])


class Bench:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = lm.load(device)
        self.train = lm.tokens(lm.TRAIN, lm.TRAIN_TOKENS)
        self.validation = lm.tokens(lm.VALIDATION)
        self.test = lm.tokens(lm.TEST)
        self.layers = [bank.stack(step) for step in self.model.feed_forwards()]

    def original(self) -> dict[str, Any]:
        return {
            "bytes": {str(bits): bank.stored_bytes(self.model, bits) for bits in BITS},
            "perplexity": {str(bits): lm.perplexity(self.squeezed(self.model, bits), self.test) for bits in BITS},
            "validation_perplexity": lm.perplexity(self.model, self.validation),
        }

    def squeezed(self, model: lm.Lm, bits: int) -> lm.Lm:
        return model if bits == bank.FLOAT_BITS else bank.quantized(model, bits)

    def similarity(self) -> dict[str, Any]:
        generator = torch.Generator(device=self.device).manual_seed(0)
        scrambled = bank.scrambled(self.layers, generator)
        attention = {
            name: bank.pairwise(
                [getattr(self.model.block(layer), name).weight.detach() for layer in range(lm.LAYERS)], matched=False
            ).tolist()
            for name in ("query", "key", "value", "out")
        }
        return {
            "raw": bank.pairwise(self.layers, matched=False).tolist(),
            "matched": bank.pairwise(self.layers, matched=True).tolist(),
            "scrambled_matched": bank.pairwise(scrambled, matched=True).tolist(),
            "shared_energy": bank.shared_energy(self.layers),
            "attention_raw": attention,
        }

    @cached_property
    def matched(self) -> np.ndarray:
        return np.asarray(read((RUNS / "similarity.json"))["matched"])

    def fitted(self, arm: Arm, matched: bool = True) -> bank.Fit:
        return bank.fit(self.layers, bank.groups(self.matched, arm.chunks), arm.rank, matched)

    def student(self, arm: Arm) -> tuple[lm.Lm, dict[str, Any]]:
        fitted = self.fitted(arm)
        facts: dict[str, Any] = {"assignment": fitted.assignment}
        if arm.kind == "bank":
            facts["weight_error"] = fitted.error()
            facts["unmatched_weight_error"] = self.fitted(arm, matched=False).error()
            return bank.banked(self.model, fitted), facts
        if arm.kind == "routed":
            return bank.routed(self.model, fitted), facts
        if arm.kind == "low-rank-kept":
            student = bank.kept_low_rank(self.model, fitted)
            facts["ranks"] = [bank.kept_rank(fitted, layer) if fitted.shares(layer) else None for layer in range(lm.LAYERS)]
        else:
            rank = bank.equal_rank(bank.stored_bytes(bank.banked(self.model, fitted)), self.model)
            student = bank.low_rank(self.model, rank)
            facts["ranks"] = [rank] * lm.LAYERS
        facts["weight_error"] = bank.weight_error(self.model, student)
        return student, facts

    def off_assignment(self, student: lm.Lm, assignment: list[int]) -> float:
        """Share of test tokens, over all layers, whose routed pick is not the fit's bank matrix."""
        inputs, _ = windows(self.test, lm.CONTEXT)
        moved = total = 0
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            for start in range(0, len(inputs), 8):
                student.hidden(inputs[start : start + 8].to(self.device))
                for layer in range(lm.LAYERS):
                    step = student.block(layer).feed_forward
                    assert isinstance(step, bank.RoutedFeedForward) and step.picked is not None
                    picked = step.picked
                    moved += int((picked != assignment[layer]).sum())
                    total += picked.numel()
        return moved / total

    def run(self, arm: Arm) -> dict[str, Any]:
        started = time.time()
        torch.manual_seed(arm.seed)
        student, facts = self.student(arm)
        rate, steps = arm.rate, arm.steps
        record: dict[str, Any] = {
            **facts,
            "kind": arm.kind,
            "chunks": arm.chunks,
            "rank": arm.rank,
            "seed": arm.seed,
            "rate": rate,
            "steps": steps,
            "bytes": {str(bits): bank.stored_bytes(student, bits) for bits in BITS},
            "fit_perplexity": lm.perplexity(student, self.test),
        }
        micro = ROUTED_MICRO if arm.kind == "routed" else 4
        record["loss"] = heal(student, self.model, self.train, steps, rate, arm.seed, micro)
        record["validation_perplexity"] = lm.perplexity(student, self.validation)
        if isinstance(arm.budget, Full):
            record["healed_perplexity"] = {
                str(bits): lm.perplexity(self.squeezed(student, bits), self.test) for bits in BITS
            }
        if arm.kind == "routed":
            record["off_assignment"] = self.off_assignment(student, facts["assignment"])
        record["seconds"] = time.time() - started
        return record


def plan() -> list[Arm]:
    """Every arm of the report in the order it is run, the learning-rate sweeps apart."""
    arms = [Arm(kind, chunks, rank) for chunks, rank in BANKS for kind in FIXED]
    arms.append(Arm("routed", *ROUTED_BANK))
    arms += [Arm(kind, *SWEEP_BANK, Full(seed)) for seed in EXTRA_SEEDS for kind in FIXED]
    return arms


def missing() -> bool:
    paths = [RUNS / "original.json", (RUNS / "similarity.json"), *(arm.path for arm in plan())]
    return not all(path.exists() for path in paths) or any(sweep(kind) for kind in FIXED)


def main() -> None:
    if not missing():
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = Bench(device)
    if not (RUNS / "original.json").exists():
        write(RUNS / "original.json", bench.original())
    if not (RUNS / "similarity.json").exists():
        write((RUNS / "similarity.json"), bench.similarity())
    for kind in FIXED:
        while pending := sweep(kind):
            for arm in pending:
                write(arm.path, bench.run(arm))
                print(arm.path, flush=True)
    for arm in plan():
        if not arm.path.exists():
            write(arm.path, bench.run(arm))
            print(arm.path, flush=True)
