from __future__ import annotations

import hashlib
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor, nn

from weightbank.data import windows

MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_URL = f"https://huggingface.co/{MODEL}/resolve/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/"
TEXT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/"
DATA = Path("data/compress")

LAYERS = 30
HEADS = 9
KEY_VALUE_HEADS = 3
ROTARY_BASE = 100000.0
NORM_EPSILON = 1e-5
CONTEXT = 512
TRAIN_TOKENS = 16_000_000


@dataclass(frozen=True)
class Download:
    url: str
    sha256: str

    @property
    def path(self) -> Path:
        return DATA / self.url.rsplit("/", 1)[1]

    def fetch(self) -> Path:
        if not self.path.exists():
            with urllib.request.urlopen(self.url) as response:
                content = response.read()
            digest = hashlib.sha256(content).hexdigest()
            if digest != self.sha256:
                raise RuntimeError(f"{self.url} sha256 {digest}, expected {self.sha256}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(content)
        return self.path


WEIGHTS = Download(MODEL_URL + "model.safetensors", "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1")
TOKENIZER = Download(MODEL_URL + "tokenizer.json", "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c")
TRAIN = Download(
    TEXT_URL + "wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212",
)
VALIDATION = Download(
    TEXT_URL + "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
    "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
)
TEST = Download(
    TEXT_URL + "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
)


class Matrix(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Dense(Matrix):
    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class FeedForward(nn.Module):
    def __init__(self, gate: Matrix, up: Matrix, down: Matrix) -> None:
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down(F.silu(self.gate(x)) * self.up(x))
        return out


def normalise(x: Tensor, weight: Tensor) -> Tensor:
    wide = x.float()
    return (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + NORM_EPSILON)).to(x.dtype) * weight


def rotate(x: Tensor) -> Tensor:
    length, half = x.shape[-2], x.shape[-1] // 2
    frequency = ROTARY_BASE ** (-torch.arange(half, device=x.device) / half)
    angle = torch.arange(length, device=x.device)[:, None] * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, weights: dict[str, Tensor]) -> None:
        super().__init__()
        self.attention_norm = nn.Parameter(weights["input_layernorm.weight"])
        self.query = Dense(weights["self_attn.q_proj.weight"])
        self.key = Dense(weights["self_attn.k_proj.weight"])
        self.value = Dense(weights["self_attn.v_proj.weight"])
        self.out = Dense(weights["self_attn.o_proj.weight"])
        self.feed_forward_norm = nn.Parameter(weights["post_attention_layernorm.weight"])
        self.feed_forward: nn.Module = FeedForward(
            Dense(weights["mlp.gate_proj.weight"]),
            Dense(weights["mlp.up_proj.weight"]),
            Dense(weights["mlp.down_proj.weight"]),
        )

    def attend(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        query = self.query(x).view(batch, length, HEADS, -1).transpose(1, 2)
        key = self.key(x).view(batch, length, KEY_VALUE_HEADS, -1).transpose(1, 2)
        value = self.value(x).view(batch, length, KEY_VALUE_HEADS, -1).transpose(1, 2)
        repeat = HEADS // KEY_VALUE_HEADS
        attended = F.scaled_dot_product_attention(
            rotate(query),
            rotate(key).repeat_interleave(repeat, dim=1),
            value.repeat_interleave(repeat, dim=1),
            is_causal=True,
        )
        out: Tensor = self.out(attended.transpose(1, 2).reshape(batch, length, -1))
        return out

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attend(normalise(x, self.attention_norm))
        stepped: Tensor = self.feed_forward(normalise(x, self.feed_forward_norm))
        return x + stepped


class Lm(nn.Module):
    def __init__(self, weights: dict[str, Tensor]) -> None:
        super().__init__()
        self.embedding = nn.Parameter(weights["model.embed_tokens.weight"])
        self.blocks = nn.ModuleList(
            Block({name.removeprefix(prefix): weight for name, weight in weights.items() if name.startswith(prefix)})
            for prefix in (f"model.layers.{layer}." for layer in range(LAYERS))
        )
        self.norm = nn.Parameter(weights["model.norm.weight"])

    def hidden(self, tokens: Tensor) -> Tensor:
        x = self.embedding[tokens]
        for block in self.blocks:
            x = block(x)
        return x

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(normalise(hidden, self.norm), self.embedding)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.logits(self.hidden(tokens))

    def block(self, layer: int) -> Block:
        block = self.blocks[layer]
        assert isinstance(block, Block)
        return block

    def feed_forwards(self) -> list[FeedForward]:
        steps = [self.block(layer).feed_forward for layer in range(LAYERS)]
        return [step for step in steps if isinstance(step, FeedForward)]


def load(device: torch.device) -> Lm:
    weights = {name: weight.float() for name, weight in load_file(WEIGHTS.fetch()).items()}
    model = Lm(weights).to(device)
    model.requires_grad_(False)
    return model.eval()


def tokens(source: Download, limit: int | None = None) -> Tensor:
    cache = source.path.with_suffix(f".{limit}.npy")
    if not cache.exists():
        tokenizer = Tokenizer.from_file(str(TOKENIZER.fetch()))
        lines: list[str] = pyarrow.parquet.read_table(source.fetch()).column("text").to_pylist()
        pieces: list[np.ndarray] = []
        count = 0
        for start in range(0, len(lines), 2000):
            piece = np.asarray(tokenizer.encode("".join(lines[start : start + 2000])).ids, dtype=np.int32)
            pieces.append(piece)
            count += len(piece)
            if limit is not None and count >= limit:
                break
        np.save(cache, np.concatenate(pieces)[:limit])
    return torch.from_numpy(np.load(cache))


@torch.no_grad()
def perplexity(model: Lm, text: Tensor, batch: int = 8) -> float:
    model.eval()
    device = next(model.parameters()).device
    inputs, targets = windows(text, CONTEXT)
    total = 0.0
    for start in range(0, len(inputs), batch):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs[start : start + batch].to(device))
        total += F.cross_entropy(
            logits.float().flatten(0, 1), targets[start : start + batch].to(device).flatten(), reduction="sum"
        ).item()
    return math.exp(total / targets.numel())
