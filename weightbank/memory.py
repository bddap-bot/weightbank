from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler

from weightbank.configs import config
from weightbank.model import CONTEXT, VOCAB, Transformer
from weightbank.sweep import FIRST_RATE
from weightbank.train import BATCH, EVAL_BATCH, RUNS, optimizer_for, step_once

PATH = RUNS / "memory.json"
TRAINING_STEPS = 3


@dataclass
class Memory:
    weights_bytes: int
    inference_window_bytes: int
    inference_batch_bytes: int
    training_bytes: int


def tokens(batch: int, device: torch.device) -> Tensor:
    return torch.randint(VOCAB, (batch, CONTEXT), device=device)


def inference(model: Transformer, batch: int, device: torch.device) -> int:
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        model(tokens(batch, device))
    return torch.cuda.max_memory_allocated(device)


def measure(name: str, device: torch.device) -> Memory:
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated(device)
    model = Transformer(config(name)).to(device)
    weights = torch.cuda.memory_allocated(device) - before
    model.eval()
    window, batch = (inference(model, size, device) - before for size in (1, EVAL_BATCH))
    model.train()
    optimizer = optimizer_for(model, FIRST_RATE)
    scaler = GradScaler()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(TRAINING_STEPS):
        step_once(model, optimizer, scaler, tokens(BATCH, device), tokens(BATCH, device))
    return Memory(weights, window, batch, torch.cuda.max_memory_allocated(device) - before)


def read(path: Path = PATH) -> dict[str, Memory]:
    return {name: Memory(**fields) for name, fields in json.loads(path.read_text()).items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the peak GPU memory PyTorch's allocator holds for a cell, training and at inference."
    )
    parser.add_argument("cells", nargs="+", help="cell names, as the ledger's axes spell them")
    arguments = parser.parse_args()
    device = torch.device("cuda")
    measured = read() if PATH.exists() else {}
    # The first pass in a process allocates library workspaces that stay; they are no cell's.
    measure(arguments.cells[0], device)
    for name in arguments.cells:
        measured[name] = measure(name, device)
        print(name, json.dumps(asdict(measured[name])), flush=True)
    PATH.write_text(json.dumps({name: asdict(memory) for name, memory in measured.items()}, indent=1) + "\n")


if __name__ == "__main__":
    main()
