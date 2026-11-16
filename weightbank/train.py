from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from weightbank import data
from weightbank.configs import CONFIGS
from weightbank.model import ModelConfig, Transformer

RUNS = Path("runs")
STEPS = 8000
BATCH = 64
EVAL_BATCH = 256
PEAK_RATE = 1e-3
WARMUP = 200
WEIGHT_DECAY = 0.1
LOG_EVERY = 500


@dataclass
class Evaluation:
    loss: float
    feed_forward_picks: list[list[int]]
    attention_picks: list[list[int]]


@dataclass
class Run:
    config: str
    seed: int
    steps: int
    stored_weights: int
    validation_loss: float
    wall_seconds: float
    feed_forward_picks: list[list[int]]
    attention_picks: list[list[int]]
    curve: list[tuple[int, float]]


def run_path(config: str, seed: int) -> Path:
    return RUNS / f"{config}-s{seed}.json"


def rate(step: int, steps: int) -> float:
    if step < WARMUP:
        return PEAK_RATE * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, steps - WARMUP)
    return PEAK_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def count(picks: list[Tensor], chunks: int) -> Tensor:
    return torch.stack([torch.bincount(pick.flatten(), minlength=chunks) for pick in picks])


@torch.no_grad()
def evaluate(model: Transformer, text: Tensor, limit: int | None = None) -> Evaluation:
    model.eval()
    config: ModelConfig = model.config
    inputs, targets = data.windows(text, config.context)
    total = 0.0
    tokens = 0
    feed_forward: Tensor | None = None
    attention: Tensor | None = None
    for start in range(0, len(inputs) if limit is None else limit, EVAL_BATCH):
        batch = inputs[start : start + EVAL_BATCH]
        with torch.autocast(batch.device.type, dtype=torch.float16):
            logits, picks = model(batch)
        total += F.cross_entropy(
            logits.float().flatten(0, 1), targets[start : start + EVAL_BATCH].flatten(), reduction="sum"
        ).item()
        tokens += batch.numel()
        if picks.feed_forward:
            counts = count(picks.feed_forward, config.chunks(config.feed_forward))
            feed_forward = counts if feed_forward is None else feed_forward + counts
        if picks.attention:
            counts = count(picks.attention, config.chunks(config.attention))
            attention = counts if attention is None else attention + counts
    model.train()
    return Evaluation(
        loss=total / tokens,
        feed_forward_picks=[] if feed_forward is None else feed_forward.tolist(),
        attention_picks=[] if attention is None else attention.tolist(),
    )


def train(name: str, seed: int, steps: int) -> Run:
    device = torch.device("cuda")
    free, _ = torch.cuda.mem_get_info()
    print(f"{name} seed {seed}: {free / 2**20:.0f} MiB of GPU memory free", flush=True)
    corpus = data.load(device)
    torch.manual_seed(seed)
    generator = torch.Generator(device).manual_seed(seed)
    model = Transformer(CONFIGS[name]).to(device)
    stored = model.stored_weights()
    print(f"stored weights {stored}", flush=True)
    matrices = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    vectors = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": matrices, "weight_decay": WEIGHT_DECAY},
            {"params": vectors, "weight_decay": 0.0},
        ],
        lr=PEAK_RATE,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler()
    curve: list[tuple[int, float]] = []
    started = time.monotonic()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = rate(step, steps)
        inputs, targets = data.sample(corpus.train, BATCH, model.config.context, generator)
        with torch.autocast(device.type, dtype=torch.float16):
            logits, _ = model(inputs)
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(scaler.scale(loss))
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % LOG_EVERY == 0:
            validation = evaluate(model, corpus.validation, limit=8 * EVAL_BATCH).loss
            curve.append((step + 1, validation))
            print(
                f"step {step + 1} train {loss.item():.4f} validation {validation:.4f} "
                f"elapsed {time.monotonic() - started:.0f}s",
                flush=True,
            )
    final = evaluate(model, corpus.validation)
    wall = time.monotonic() - started
    print(f"final validation loss {final.loss:.4f} nats/byte, wall {wall:.0f}s", flush=True)
    return Run(
        config=name,
        seed=seed,
        steps=steps,
        stored_weights=stored,
        validation_loss=final.loss,
        wall_seconds=wall,
        feed_forward_picks=final.feed_forward_picks,
        attention_picks=final.attention_picks,
        curve=curve,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one configuration at one seed.")
    parser.add_argument("config", choices=sorted(CONFIGS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    run = train(arguments.config, arguments.seed, arguments.steps)
    out: Path = arguments.out or run_path(arguments.config, arguments.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(run), indent=1) + "\n")


if __name__ == "__main__":
    main()
