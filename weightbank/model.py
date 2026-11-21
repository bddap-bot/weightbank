from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Generic, TypeVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HEADS = 4
CONTEXT = 256
VOCAB = 256
ROUTER_WIDTH = 64
CLASSIC = "classic"
MODERN = "modern"
ARCHITECTURES = (CLASSIC, MODERN)
ROTARY_BASE = 10000.0


@dataclass(frozen=True)
class Own:
    pass


@dataclass(frozen=True)
class Fixed:
    chunks: int


class Settled(Enum):
    NOTHING = "nothing"
    PICKS = "picks"
    ROUTER = "router"


@dataclass(frozen=True)
class Routed:
    chunks: int
    noise: bool = True
    keys: bool = False
    settled: Settled = Settled.NOTHING


Sharing = Own | Fixed | Routed


@dataclass(frozen=True)
class ModelConfig:
    modules: int
    width: int
    feed_forward: Sharing = Own()
    attention: Sharing = Own()
    rank: int = 0
    architecture: str = CLASSIC

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture {self.architecture!r} is none of {ARCHITECTURES}")
        if self.rank and isinstance(self.feed_forward, Own) and isinstance(self.attention, Own):
            raise ValueError("a correction needs a shared bank to correct")


def linear(inputs: int, outputs: int, std: float = 0.02) -> nn.Linear:
    layer = nn.Linear(inputs, outputs)
    nn.init.normal_(layer.weight, std=std)
    nn.init.zeros_(layer.bias)
    return layer


def residual_std(config: ModelConfig) -> float:
    return 0.02 / math.sqrt(2 * config.modules)


class LowRank(nn.Module):
    def __init__(self, corrected: nn.Linear, rank: int) -> None:
        super().__init__()
        self.read = nn.Parameter(torch.randn(rank, corrected.in_features) * 0.02)
        self.write = nn.Parameter(torch.zeros(corrected.out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.read.T @ self.write.T


class Correction(nn.Module):
    def __init__(self, first: nn.Linear, second: nn.Linear, rank: int) -> None:
        super().__init__()
        self.first = LowRank(first, rank)
        self.second = LowRank(second, rank)


class RootMeanSquareNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        scale = x.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()
        return (x.float() * scale * self.weight).to(x.dtype)


def norm(config: ModelConfig) -> nn.Module:
    return RootMeanSquareNorm(config.width) if config.architecture == MODERN else nn.LayerNorm(config.width)


class Chunk(nn.Module):
    def maps(self) -> tuple[nn.Linear, nn.Linear]:
        raise NotImplementedError


class FeedForward(Chunk):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gated = config.architecture == MODERN
        hidden = 8 * config.width // 3 if self.gated else 4 * config.width
        self.up = linear(config.width, 2 * hidden if self.gated else hidden)
        self.down = linear(hidden, config.width, residual_std(config))

    def maps(self) -> tuple[nn.Linear, nn.Linear]:
        return self.up, self.down

    def forward(self, x: Tensor, correction: Correction | None = None) -> Tensor:
        hidden: Tensor = self.up(x)
        if correction is not None:
            hidden = hidden + correction.first(x)
        if self.gated:
            gate, value = hidden.chunk(2, dim=-1)
            hidden = F.silu(gate) * value
        else:
            hidden = F.gelu(hidden)
        out: Tensor = self.down(hidden)
        if correction is not None:
            out = out + correction.second(hidden)
        return out


class AttentionWeights(Chunk):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.qkv = linear(config.width, 3 * config.width)
        self.down = linear(config.width, config.width, residual_std(config))

    def maps(self) -> tuple[nn.Linear, nn.Linear]:
        return self.qkv, self.down


def rotate(x: Tensor) -> Tensor:
    length, width = x.shape[-2:]
    frequencies = ROTARY_BASE ** (-torch.arange(0, width, 2, device=x.device) / width)
    angles = torch.outer(torch.arange(length, device=x.device), frequencies)
    even, odd = x[..., 0::2], x[..., 1::2]
    turned = torch.stack((even * angles.cos() - odd * angles.sin(), even * angles.sin() + odd * angles.cos()), dim=-1)
    return turned.flatten(-2).to(x.dtype)


def attend(qkv: Tensor, rotary: bool = False) -> Tensor:
    batch, length, _ = qkv.shape
    q, k, v = (part.view(batch, length, HEADS, -1).transpose(1, 2) for part in qkv.chunk(3, dim=-1))
    if rotary:
        q, k = rotate(q), rotate(k)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(batch, length, -1)


@dataclass
class Picks:
    hard: Tensor
    confidence: Tensor

    @property
    def chunk(self) -> Tensor:
        return self.hard.argmax(-1)


def pick(logits: Tensor, sample: bool, straight_through: bool) -> Picks:
    confidence = logits.detach().softmax(-1).amax(-1)
    if sample:
        return Picks(F.gumbel_softmax(logits, tau=1.0, hard=True), confidence)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    if straight_through:
        soft = logits.softmax(-1)
        hard = hard - soft.detach() + soft
    return Picks(hard, confidence)


class Router(nn.Module):
    def __init__(self, sharing: Routed) -> None:
        super().__init__()
        self.noise = sharing.noise

    def logits(self, x: Tensor, module: int) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, module: int) -> Picks:
        return pick(self.logits(x, module).float(), self.training and self.noise, self.training)


class ScoreRouter(Router):
    def __init__(self, config: ModelConfig, sharing: Routed) -> None:
        super().__init__(sharing)
        self.data = linear(config.width, ROUTER_WIDTH)
        self.asker = nn.Parameter(torch.randn(config.modules, ROUTER_WIDTH) * 0.02)
        self.score = linear(ROUTER_WIDTH, sharing.chunks)

    def logits(self, x: Tensor, module: int) -> Tensor:
        logits: Tensor = self.score(torch.tanh(self.data(x) + self.asker[module]))
        return logits


class KeyRouter(Router):
    def __init__(self, config: ModelConfig, sharing: Routed) -> None:
        super().__init__(sharing)
        self.query = linear(config.width, ROUTER_WIDTH)
        self.asker = nn.Parameter(torch.randn(config.modules, config.width) * 0.02)
        self.keys = nn.Parameter(torch.randn(sharing.chunks, ROUTER_WIDTH) * 0.02)

    def logits(self, x: Tensor, module: int) -> Tensor:
        logits: Tensor = self.query(x + self.asker[module]) @ self.keys.T
        return logits


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


Stored = TypeVar("Stored", bound=Chunk)


class Bank(nn.Module, Generic[Stored]):
    def __init__(self, config: ModelConfig, sharing: Sharing, chunk: Callable[[ModelConfig], Stored]) -> None:
        super().__init__()
        self.sharing = sharing
        count = config.modules if isinstance(sharing, Own) else sharing.chunks
        self.chunks = tuple(chunk(config) for _ in range(count))
        self.registered = nn.ModuleList(self.chunks)
        self.router: Router | None = None
        if isinstance(sharing, Routed):
            self.router = KeyRouter(config, sharing) if sharing.keys else ScoreRouter(config, sharing)
        self.corrections: tuple[Correction, ...] | None = None
        if config.rank and not isinstance(sharing, Own):
            first, second = self.chunks[0].maps()
            self.corrections = tuple(Correction(first, second, config.rank) for _ in range(config.modules))
            self.registered_corrections = nn.ModuleList(self.corrections)
        self.settled_picks: list[int] | None = None

    def correction(self, module: int) -> Correction | None:
        return None if self.corrections is None else self.corrections[module]

    def settles(self) -> bool:
        return isinstance(self.sharing, Routed) and self.sharing.settled is not Settled.NOTHING

    def settle(self, common: list[int]) -> None:
        assert isinstance(self.sharing, Routed) and self.router is not None
        if self.sharing.settled is Settled.PICKS:
            self.settled_picks = common
        elif self.sharing.settled is Settled.ROUTER:
            self.router.requires_grad_(False)

    def picks(self, x: Tensor, module: int) -> Picks | None:
        if self.settled_picks is not None:
            return fixed(x, self.settled_picks[module], len(self.chunks))
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
        self.rotary = config.architecture == MODERN
        self.position = None if self.rotary else nn.Embedding(CONTEXT, config.width)
        for embedding in (self.token, self.position):
            if embedding is not None:
                nn.init.normal_(embedding.weight, std=0.02)
        self.attention_norm = nn.ModuleList(norm(config) for _ in range(config.modules))
        self.feed_forward_norm = nn.ModuleList(norm(config) for _ in range(config.modules))
        self.attention = Bank(config, config.attention, AttentionWeights)
        self.feed_forward = Bank(config, config.feed_forward, FeedForward)
        self.final_norm = norm(config)
        self.head = linear(config.width, VOCAB)

    def stored_weights(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _attention(self, x: Tensor, module: int, trace: list[Picks]) -> Tensor:
        bank = self.attention
        picks = bank.picks(x, module)
        if picks is None:
            own = bank.chunks[module]
            out: Tensor = own.down(attend(own.qkv(x), self.rotary))
            return out
        trace.append(picks)
        correction = bank.correction(module)
        qkv = mix([chunk.qkv for chunk in bank.chunks], picks.hard, x)
        if correction is not None:
            qkv = qkv + correction.first(x)
        attended = attend(qkv, self.rotary)
        out = mix([chunk.down for chunk in bank.chunks], picks.hard, attended)
        if correction is not None:
            out = out + correction.second(attended)
        return out

    def _feed_forward(self, x: Tensor, module: int, trace: list[Picks]) -> Tensor:
        bank = self.feed_forward
        picks = bank.picks(x, module)
        if picks is None:
            out: Tensor = bank.chunks[module](x)
            return out
        trace.append(picks)
        correction = bank.correction(module)
        return mix([partial(chunk, correction=correction) for chunk in bank.chunks], picks.hard, x)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Trace]:
        trace = Trace()
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x: Tensor = self.token(tokens)
        if self.position is not None:
            x = x + self.position(positions)
        for module in range(self.config.modules):
            x = x + self._attention(self.attention_norm[module](x), module, trace.attention)
            x = x + self._feed_forward(self.feed_forward_norm[module](x), module, trace.feed_forward)
        logits: Tensor = self.head(self.final_norm(x))
        return logits, trace
