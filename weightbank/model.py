from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Weights(Enum):
    OWN = "own"
    ROUTED = "routed"
    PINNED = "pinned"


@dataclass(frozen=True)
class ModelConfig:
    modules: int
    width: int
    feed_forward: Weights = Weights.OWN
    attention: Weights = Weights.OWN
    bank: int = 0
    heads: int = 4
    context: int = 256
    vocab: int = 256
    router_width: int = 64

    def chunks(self, weights: Weights) -> int:
        return self.bank if weights is Weights.ROUTED else self.modules


class FeedForward(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(4 * width, width)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down(F.gelu(self.up(x)))
        return out


class AttentionWeights(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(width, 3 * width)
        self.down = nn.Linear(width, width)


class Router(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.data = nn.Linear(config.width, config.router_width)
        self.asker = nn.Embedding(config.modules, config.router_width)
        self.score = nn.Linear(config.router_width, config.bank)

    def forward(self, x: Tensor, module: int) -> Tensor:
        logits: Tensor = self.score(torch.tanh(self.data(x) + self.asker.weight[module])).float()
        if not self.training:
            return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        return F.gumbel_softmax(logits, tau=1.0, hard=True)


def pinned(like: Tensor, module: int, chunks: int) -> Tensor:
    picks = torch.zeros(*like.shape[:-1], chunks, device=like.device)
    picks[..., module] = 1.0
    return picks


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


def attend(qkv: Tensor, heads: int) -> Tensor:
    batch, length, _ = qkv.shape
    q, k, v = (part.view(batch, length, heads, -1).transpose(1, 2) for part in qkv.chunk(3, dim=-1))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(batch, length, -1)


@dataclass
class Picks:
    feed_forward: list[Tensor]
    attention: list[Tensor]


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.attention_norm = nn.ModuleList(nn.LayerNorm(config.width) for _ in range(config.modules))
        self.feed_forward_norm = nn.ModuleList(nn.LayerNorm(config.width) for _ in range(config.modules))
        self.attention = nn.ModuleList(
            AttentionWeights(config.width) for _ in range(config.chunks(config.attention))
        )
        self.feed_forward = nn.ModuleList(
            FeedForward(config.width) for _ in range(config.chunks(config.feed_forward))
        )
        self.attention_router = Router(config) if config.attention is Weights.ROUTED else None
        self.feed_forward_router = Router(config) if config.feed_forward is Weights.ROUTED else None
        self.final_norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab)
        self.apply(self._init)
        for name, parameter in self.named_parameters():
            if name.endswith("down.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.modules))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def stored_weights(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _picks(self, weights: Weights, router: Router | None, x: Tensor, module: int) -> Tensor | None:
        if router is not None:
            picks: Tensor = router(x, module)
            return picks
        if weights is Weights.PINNED:
            return pinned(x, module, self.config.modules)
        return None

    def _attention(self, x: Tensor, module: int, record: list[Tensor]) -> Tensor:
        picks = self._picks(self.config.attention, self.attention_router, x, module)
        if picks is None:
            own = self.attention[module]
            assert isinstance(own, AttentionWeights)
            out: Tensor = own.down(attend(own.qkv(x), self.config.heads))
            return out
        record.append(picks.argmax(-1))
        chunks = [chunk for chunk in self.attention if isinstance(chunk, AttentionWeights)]
        mixed = attend(mix([chunk.qkv for chunk in chunks], picks, x), self.config.heads)
        return mix([chunk.down for chunk in chunks], picks, mixed)

    def _feed_forward(self, x: Tensor, module: int, record: list[Tensor]) -> Tensor:
        picks = self._picks(self.config.feed_forward, self.feed_forward_router, x, module)
        if picks is None:
            out: Tensor = self.feed_forward[module](x)
            return out
        record.append(picks.argmax(-1))
        return mix(list(self.feed_forward), picks, x)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Picks]:
        picks = Picks([], [])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x: Tensor = self.token(tokens) + self.position(positions)
        for module in range(self.config.modules):
            x = x + self._attention(self.attention_norm[module](x), module, picks.attention)
            x = x + self._feed_forward(self.feed_forward_norm[module](x), module, picks.feed_forward)
        logits: Tensor = self.head(self.final_norm(x))
        return logits, picks
