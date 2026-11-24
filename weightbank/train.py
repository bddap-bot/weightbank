from __future__ import annotations

import argparse
import json
import math
import os
import time
from statistics import median
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from weightbank import data
from weightbank.configs import LEDGER, config
from weightbank.model import CONTEXT, Picks, Transformer

RUNS = Path("runs")
BATCH = 64
EVAL_BATCH = 256
WARMUP = 200
WEIGHT_DECAY = 0.1
LOG_EVERY = 500
LOG_WINDOWS = 8 * EVAL_BATCH
SETTLE_PART = 4
PACE_REACH = 2
GPU_MIB = "WEIGHTBANK_GPU_MIB"


@dataclass
class PickStats:
    counts: list[list[int]]
    confidence: list[float]


@dataclass
class Evaluation:
    loss: float
    feed_forward: PickStats | None
    attention: PickStats | None


@dataclass
class Run:
    config: str
    seed: int
    steps: int
    peak_rate: float
    stored_weights: int
    validation_loss: float
    sampled_picks_validation_loss: float
    training_seconds: float
    feed_forward: PickStats | None
    attention: PickStats | None
    curve: list[list[float]]

    @staticmethod
    def read(path: Path) -> Run:
        fields: dict[str, Any] = json.loads(path.read_text())
        for kind in ("feed_forward", "attention"):
            if fields[kind] is not None:
                fields[kind] = PickStats(**fields[kind])
        return Run(**fields)


@dataclass(frozen=True)
class Plan:
    config: str
    seed: int
    steps: int
    peak_rate: float

    @property
    def path(self) -> Path:
        return LEDGER.run_path(RUNS, LEDGER.parse(self.config, self.steps), self.peak_rate, self.seed)

    @property
    def command(self) -> list[str]:
        return [
            self.config,
            "--seed",
            str(self.seed),
            "--steps",
            str(self.steps),
            "--rate",
            repr(self.peak_rate),
        ]

    def read(self) -> Run:
        run = Run.read(self.path)
        if Plan(run.config, run.seed, run.steps, run.peak_rate) != self:
            raise ValueError(f"{self.path} holds a run of another plan")
        return run


def rate(step: int, steps: int, peak: float) -> float:
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, steps - WARMUP)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def training_seconds(paces: list[float], steps: int) -> float:
    span = 2 * PACE_REACH + 1
    starts = [max(0, min(window - PACE_REACH, len(paces) - span)) for window in range(len(paces))]
    steady = [median(paces[start : start + span]) for start in starts]
    return sum(pace * min(LOG_EVERY, steps - window * LOG_EVERY) for window, pace in enumerate(steady))


class PickTally:
    def __init__(self) -> None:
        self.counts: Tensor | None = None
        self.confidence: Tensor | None = None

    def add(self, trace: list[Picks]) -> None:
        if not trace:
            return
        chunks = trace[0].hard.shape[-1]
        counts = torch.stack([torch.bincount(picks.chunk.flatten(), minlength=chunks) for picks in trace])
        confidence = torch.stack([picks.confidence.sum() for picks in trace])
        self.counts = counts if self.counts is None else self.counts + counts
        self.confidence = confidence if self.confidence is None else self.confidence + confidence

    def stats(self) -> PickStats | None:
        if self.counts is None or self.confidence is None:
            return None
        return PickStats(
            counts=self.counts.tolist(),
            confidence=(self.confidence / self.counts.sum(-1)).tolist(),
        )


@torch.no_grad()
def evaluate(
    model: Transformer, text: Tensor, windows: int | None = None, sample_picks: bool = False
) -> Evaluation:
    model.train(sample_picks)
    inputs, targets = data.windows(text, CONTEXT)
    total = 0.0
    tokens = 0
    feed_forward = PickTally()
    attention = PickTally()
    for start in range(0, len(inputs) if windows is None else min(windows, len(inputs)), EVAL_BATCH):
        batch = inputs[start : start + EVAL_BATCH]
        with torch.autocast(batch.device.type, dtype=torch.float16):
            logits, trace = model(batch)
        total += F.cross_entropy(
            logits.float().flatten(0, 1), targets[start : start + EVAL_BATCH].flatten(), reduction="sum"
        ).item()
        tokens += batch.numel()
        feed_forward.add(trace.feed_forward)
        attention.add(trace.attention)
    model.train()
    return Evaluation(total / tokens, feed_forward.stats(), attention.stats())


def settle(model: Transformer, text: Tensor) -> None:
    banks = [bank for bank in (model.feed_forward, model.attention) if bank.settles()]
    if not banks:
        return
    seen = evaluate(model, text, windows=LOG_WINDOWS)
    for bank in banks:
        stats = seen.feed_forward if bank is model.feed_forward else seen.attention
        assert stats is not None
        bank.settle([max(range(len(row)), key=row.__getitem__) for row in stats.counts])


def optimizer_for(model: Transformer, peak_rate: float) -> torch.optim.AdamW:
    matrices = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    vectors = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": WEIGHT_DECAY},
            {"params": vectors, "weight_decay": 0.0},
        ],
        lr=peak_rate,
        betas=(0.9, 0.95),
    )


def step_once(
    model: Transformer, optimizer: torch.optim.AdamW, scaler: GradScaler, inputs: Tensor, targets: Tensor
) -> Tensor:
    with torch.autocast(inputs.device.type, dtype=torch.float16):
        logits, _ = model(inputs)
    loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(scaler.scale(loss))
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss


def train(plan: Plan) -> Run:
    name, seed, steps = plan.config, plan.seed, plan.steps
    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    if GPU_MIB in os.environ:
        torch.cuda.set_per_process_memory_fraction(int(os.environ[GPU_MIB]) * 2**20 / total)
    print(f"{name} seed {seed}: {free / 2**20:.0f} MiB of GPU memory free", flush=True)
    corpus = data.load(device)
    torch.manual_seed(seed)
    generator = torch.Generator(device).manual_seed(seed)
    model = Transformer(config(name)).to(device)
    stored = model.stored_weights()
    print(f"stored weights {stored}", flush=True)
    optimizer = optimizer_for(model, plan.peak_rate)
    scaler = GradScaler()
    curve: list[list[float]] = []
    paces: list[float] = []
    started = time.monotonic()
    for step in range(steps):
        if step == steps // SETTLE_PART:
            settle(model, corpus.train[: LOG_WINDOWS * CONTEXT + 1])
        for group in optimizer.param_groups:
            group["lr"] = rate(step, steps, plan.peak_rate)
        inputs, targets = data.sample(corpus.train, BATCH, CONTEXT, generator)
        loss = step_once(model, optimizer, scaler, inputs, targets)
        if (step + 1) % LOG_EVERY == 0:
            logged = loss.item()
            paces.append((time.monotonic() - started) / LOG_EVERY)
            if not math.isfinite(logged):
                print(f"training loss {logged} at step {step + 1}: diverged", flush=True)
                seconds = training_seconds(paces, step + 1)
                return Run(name, seed, steps, plan.peak_rate, stored, math.inf, math.inf, seconds, None, None, curve)
            validation = evaluate(model, corpus.validation, windows=LOG_WINDOWS).loss
            curve.append([step + 1, validation])
            print(
                f"step {step + 1} train {logged:.4f} validation {validation:.4f} "
                f"{paces[-1] * 1000:.0f} ms a step",
                flush=True,
            )
            started = time.monotonic()
    if steps % LOG_EVERY:
        paces.append((time.monotonic() - started) / (steps % LOG_EVERY))
    seconds = training_seconds(paces, steps)
    final = evaluate(model, corpus.validation)
    sampled = evaluate(model, corpus.validation, sample_picks=True)
    print(
        f"final validation loss {final.loss:.4f} nats/byte "
        f"({sampled.loss:.4f} with sampled picks), training {seconds:.0f}s",
        flush=True,
    )
    return Run(
        config=name,
        seed=seed,
        steps=steps,
        peak_rate=plan.peak_rate,
        stored_weights=stored,
        validation_loss=final.loss,
        sampled_picks_validation_loss=sampled.loss,
        training_seconds=seconds,
        feed_forward=final.feed_forward,
        attention=final.attention,
        curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one configuration at one seed.")
    parser.add_argument("config", help="a cell's name, as the ledger's axes spell it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--rate", type=float, required=True, help="peak learning rate")
    arguments = parser.parse_args()
    plan = Plan(arguments.config, arguments.seed, arguments.steps, arguments.rate)
    config(plan.config)
    run = train(plan)
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    partial = plan.path.with_suffix(".partial")
    partial.write_text(json.dumps(asdict(run), indent=1) + "\n")
    partial.replace(plan.path)


if __name__ == "__main__":
    main()
