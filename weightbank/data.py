from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

URL = "https://mattmahoney.net/dc/enwik8.zip"
SHA256 = "547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc"
PATH = Path("data/enwik8")
TRAIN_BYTES = 90_000_000
VALIDATION_BYTES = 5_000_000


@dataclass
class Corpus:
    train: Tensor
    validation: Tensor


def load(device: torch.device) -> Corpus:
    if not PATH.exists():
        with urllib.request.urlopen(URL) as response:
            archive = response.read()
        digest = hashlib.sha256(archive).hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"enwik8.zip sha256 {digest}, expected {SHA256}")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_bytes(zipfile.ZipFile(io.BytesIO(archive)).read("enwik8"))
    text = torch.from_numpy(np.fromfile(PATH, dtype=np.uint8)).to(device)
    return Corpus(
        train=text[:TRAIN_BYTES],
        validation=text[TRAIN_BYTES : TRAIN_BYTES + VALIDATION_BYTES],
    )


def sample(text: Tensor, batch: int, context: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    starts = torch.randint(len(text) - context, (batch,), device=text.device, generator=generator)
    drawn = text[starts[:, None] + torch.arange(context + 1, device=text.device)].long()
    return drawn[:, :-1], drawn[:, 1:]


def windows(text: Tensor, context: int) -> tuple[Tensor, Tensor]:
    count = (len(text) - 1) // context
    inputs = text[: count * context].view(count, context).long()
    targets = text[1 : count * context + 1].view(count, context).long()
    return inputs, targets
