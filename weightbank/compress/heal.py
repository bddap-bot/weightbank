from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler

from weightbank.compress.lm import CONTEXT, Lm

BATCH = 4
WARMUP = 0.05
LOG_EVERY = 50
HEAD_TOKENS = 512


def schedule(step: int, steps: int) -> float:
    warm = max(1, int(WARMUP * steps))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))


def heal(student: Lm, teacher: Lm, text: Tensor, steps: int, rate: float, seed: int, micro: int = BATCH
) -> list[float]:
    """Train the student's unfrozen weights to match the teacher's next-token distribution. A
    step's batch passes through the student micro rows at a time, gradients summed."""
    device = next(teacher.parameters()).device
    amp = device.type == "cuda"
    trained = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=rate, weight_decay=0.0)
    scaler = GradScaler(enabled=amp)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    offsets = torch.arange(CONTEXT)
    curve: list[float] = []
    running = 0.0
    student.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = rate * schedule(step, steps)
        starts = torch.randint(len(text) - CONTEXT, (BATCH,), generator=generator)
        tokens = text[starts[:, None] + offsets].long().to(device)
        optimizer.zero_grad(set_to_none=True)
        for rows in (tokens[start : start + micro] for start in range(0, BATCH, micro)):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                with torch.no_grad():
                    target = teacher.hidden(rows).flatten(0, 1)
                output = student.hidden(rows)
            head = output.detach().flatten(0, 1).requires_grad_()
            for start in range(0, len(head), HEAD_TOKENS):
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    predicted = student.logits(head[start : start + HEAD_TOKENS])
                    wanted = teacher.logits(target[start : start + HEAD_TOKENS])
                loss = F.kl_div(predicted.float().log_softmax(-1), wanted.float().softmax(-1), reduction="sum")
                loss = loss / tokens.numel()
                torch.autograd.backward(scaler.scale(loss))
                running += loss.item()
            assert head.grad is not None
            torch.autograd.backward(output, head.grad.view_as(output))
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(trained, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % LOG_EVERY == 0:
            curve.append(running / LOG_EVERY)
            running = 0.0
    student.eval()
    return curve
