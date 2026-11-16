from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HEADS = 4
CONTEXT = 256
VOCAB = 256
ROUTER_WIDTH = 64


@dataclass(frozen=True)
class Own:
    pass


@dataclass(frozen=True)
class Fixed:
    chunks: int


@dataclass(frozen=True)
class Routed:
    chunks: int


Sharing = Own | Fixed | Routed


@dataclass(frozen=True)
class ModelConfig:
    modules: int
    width: int
    feed_forward: Sharing = Own()
    attention: Sharing = Own()


def linear(inputs: int, outputs: int, std: float = 0.02) -> nn.Linear:
    layer = nn.Linear(inputs, outputs)
    nn.init.normal_(layer.weight, std=std)
    nn.init.zeros_(layer.bias)
    return layer


def residual_std(config: ModelConfig) -> float:
    return 0.02 / math.sqrt(2 * config.modules)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = linear(config.width, 4 * config.width)
        self.down = linear(4 * config.width, config.width, residual_std(config))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down(F.gelu(self.up(x)))
        return out


class AttentionWeights(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.qkv = linear(config.width, 3 * config.width)
        self.down = linear(config.width, config.width, residual_std(config))


def attend(qkv: Tensor) -> Tensor:
    batch, length, _ = qkv.shape
    q, k, v = (part.view(batch, length, HEADS, -1).transpose(1, 2) for part in qkv.chunk(3, dim=-1))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(batch, length, -1)


@dataclass
class Picks:
    hard: Tensor
    confidence: Tensor

    @property
    def chunk(self) -> Tensor:
        return self.hard.argmax(-1)


class Router(nn.Module):
    def __init__(self, config: ModelConfig, chunks: int) -> None:
        super().__init__()
        self.data = linear(config.width, ROUTER_WIDTH)
        self.asker = nn.Parameter(torch.randn(config.modules, ROUTER_WIDTH) * 0.02)
        self.score = linear(ROUTER_WIDTH, chunks)

    def forward(self, x: Tensor, module: int) -> Picks:
        logits: Tensor = self.score(torch.tanh(self.data(x) + self.asker[module])).float()
        confidence = logits.detach().softmax(-1).amax(-1)
        if self.training:
            return Picks(F.gumbel_softmax(logits, tau=1.0, hard=True), confidence)
        return Picks(F.one_hot(logits.argmax(-1), logits.shape[-1]).float(), confidence)


def fixed(like: Tensor, chunk: int, chunks: int) -> Picks:
    hard = torch.zeros(*like.shape[:-1], chunks, device=like.device)
    hard[..., chunk] = 1.0
    return Picks(hard, torch.ones(like.shape[:-1], device=like.device))


def mix(chunks: Sequence[Callable[[Tensor], Tensor]], picks: Tensor, x: Tensor) -> Tensor:
    out: Tensor | None = None
    for index, chunk in enumerate(chunks):
        pick = picks[..., index : index + 1]
        if not pick.requires_grad and not bool(pick.any()):
            continue
        computed = chunk(x)
        term = pick.to(computed.dtype) * computed
        out = term if out is None else out + term
    assert out is not None
    return out


Chunk = TypeVar("Chunk", bound=nn.Module)


class Bank(nn.Module, Generic[Chunk]):
    def __init__(self, config: ModelConfig, sharing: Sharing, chunk: Callable[[ModelConfig], Chunk]) -> None:
        super().__init__()
        self.sharing = sharing
        count = config.modules if isinstance(sharing, Own) else sharing.chunks
        self.chunks = tuple(chunk(config) for _ in range(count))
        self.registered = nn.ModuleList(self.chunks)
        self.router = Router(config, count) if isinstance(sharing, Routed) else None

    def picks(self, x: Tensor, module: int) -> Picks | None:
        if self.router is not None:
            picks: Picks = self.router(x, module)
            return picks
        if isinstance(self.sharing, Fixed):
            return fixed(x, module % self.sharing.chunks, self.sharing.chunks)
        return None


@dataclass
class Trace:
    feed_forward: list[Picks] = field(default_factory=list)
    attention: list[Picks] = field(default_factory=list)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(VOCAB, config.width)
        self.position = nn.Embedding(CONTEXT, config.width)
        for embedding in (self.token, self.position):
            nn.init.normal_(embedding.weight, std=0.02)
        self.attention_norm = nn.ModuleList(nn.LayerNorm(config.width) for _ in range(config.modules))
        self.feed_forward_norm = nn.ModuleList(nn.LayerNorm(config.width) for _ in range(config.modules))
        self.attention = Bank(config, config.attention, AttentionWeights)
        self.feed_forward = Bank(config, config.feed_forward, FeedForward)
        self.final_norm = nn.LayerNorm(config.width)
        self.head = linear(config.width, VOCAB)

    def stored_weights(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _attention(self, x: Tensor, module: int, trace: list[Picks]) -> Tensor:
        bank = self.attention
        picks = bank.picks(x, module)
        if picks is None:
            own = bank.chunks[module]
            out: Tensor = own.down(attend(own.qkv(x)))
            return out
        trace.append(picks)
        attended = attend(mix([chunk.qkv for chunk in bank.chunks], picks.hard, x))
        return mix([chunk.down for chunk in bank.chunks], picks.hard, attended)

    def _feed_forward(self, x: Tensor, module: int, trace: list[Picks]) -> Tensor:
        bank = self.feed_forward
        picks = bank.picks(x, module)
        if picks is None:
            out: Tensor = bank.chunks[module](x)
            return out
        trace.append(picks)
        return mix(bank.chunks, picks.hard, x)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Trace]:
        trace = Trace()
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x: Tensor = self.token(tokens) + self.position(positions)
        for module in range(self.config.modules):
            x = x + self._attention(self.attention_norm[module](x), module, trace.attention)
            x = x + self._feed_forward(self.feed_forward_norm[module](x), module, trace.feed_forward)
        logits: Tensor = self.head(self.final_norm(x))
        return logits, trace
