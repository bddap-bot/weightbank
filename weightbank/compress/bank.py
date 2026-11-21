from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from torch import Tensor, nn

from weightbank.compress.lm import Dense, FeedForward, Lm, Matrix
from weightbank.model import ModelConfig, Routed, ScoreRouter

FLOAT_BITS = 16
GROUP = 64
PRIOR = 8.0
FIT_ROUNDS = 4

Module = TypeVar("Module", bound=nn.Module)


class LowRank(Matrix):
    def __init__(self, up: Tensor, down: Tensor) -> None:
        super().__init__()
        self.up = nn.Parameter(up)
        self.down = nn.Parameter(down)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.down), self.up)

    def dense(self) -> Tensor:
        return self.up @ self.down


class Banked(Matrix):
    def __init__(self, shared: Dense, correction: LowRank | None) -> None:
        super().__init__()
        self.shared = shared
        self.correction = correction

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.shared(x)
        if self.correction is not None:
            out = out + self.correction(x)
        return out


def truncate(matrix: Tensor, rank: int) -> LowRank:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    root = s[:rank].sqrt()
    return LowRank(u[:, :rank] * root, root[:, None] * vh[:rank])


def stack(step: FeedForward) -> Tensor:
    """One feed-forward step as a matrix with a row per hidden unit, so reordering the hidden
    units, which leaves the step's function unchanged, is a reordering of rows."""
    gate, up, down = step.gate, step.up, step.down
    assert isinstance(gate, Dense) and isinstance(up, Dense) and isinstance(down, Dense)
    return torch.cat([gate.weight, up.weight, down.weight.T], dim=1).detach()


def unstack(rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    gate, up, down = rows.chunk(3, dim=1)
    return gate, up, down.T


def match(rows: Tensor, reference: Tensor) -> Tensor:
    _, order = linear_sum_assignment((rows @ reference.T).T.cpu().numpy(), maximize=True)
    return torch.from_numpy(order).to(rows.device)


def cosine(a: Tensor, b: Tensor) -> float:
    return float((a * b).sum() / (a.norm() * b.norm()))


def pairwise(layers: list[Tensor], matched: bool) -> np.ndarray:
    count = len(layers)
    similarity = np.eye(count)
    for first in range(count):
        for second in range(first):
            other = layers[second][match(layers[second], layers[first])] if matched else layers[second]
            similarity[first, second] = similarity[second, first] = cosine(layers[first], other)
    return similarity


def scrambled(layers: list[Tensor], generator: torch.Generator) -> list[Tensor]:
    """Each layer with every column's entries shuffled among the hidden units: the same values,
    no relation between layers left. What matching hidden units finds between these is the
    similarity it manufactures from unrelated matrices."""
    return [
        torch.stack(
            [column[torch.randperm(len(column), generator=generator, device=column.device)] for column in layer.T],
            dim=1,
        )
        for layer in layers
    ]


def shared_energy(layers: list[Tensor]) -> list[float]:
    """Share of the layers' summed squared weights that the best k shared directions hold, for
    every k: the most a scheme that writes each layer, hidden units in the order given, as a mix
    of k shared matrices can keep."""
    flat = torch.stack([layer.flatten() for layer in layers]).double()
    spectrum = torch.linalg.eigvalsh(flat @ flat.T).flip(0).clamp(min=0)
    return [float(share) for share in spectrum.cumsum(0) / spectrum.sum()]


def groups(similarity: np.ndarray, count: int) -> list[int]:
    if count >= len(similarity):
        return list(range(len(similarity)))
    distance = squareform(1 - similarity, checks=False)
    labels = [int(label) - 1 for label in fcluster(linkage(distance, method="average"), count, criterion="maxclust")]
    assert len(set(labels)) == count
    return labels


@dataclass
class Fit:
    assignment: list[int]
    centres: list[Tensor]
    layers: list[Tensor]
    rank: int

    def residual(self, layer: int) -> Tensor:
        return self.layers[layer] - self.centres[self.assignment[layer]]

    def shares(self, layer: int) -> bool:
        return self.assignment.count(self.assignment[layer]) > 1

    def corrections(self, layer: int) -> tuple[LowRank, LowRank, LowRank] | None:
        if not self.shares(layer) or self.rank == 0:
            return None
        gate, up, down = (truncate(part, self.rank) for part in unstack(self.residual(layer)))
        return gate, up, down

    def rebuilt(self, layer: int) -> Tensor:
        corrections = self.corrections(layer)
        centre = self.centres[self.assignment[layer]]
        if corrections is None:
            return centre
        gate, up, down = (correction.dense().detach() for correction in corrections)
        return centre + torch.cat([gate, up, down.T], dim=1)

    def error(self) -> float:
        lost = sum(float((self.layers[layer] - self.rebuilt(layer)).pow(2).sum()) for layer in range(len(self.layers)))
        return lost / sum(float(layer.pow(2).sum()) for layer in self.layers)


def fit(layers: list[Tensor], assignment: list[int], rank: int, matched: bool) -> Fit:
    """Bank matrices, each layer's hidden-unit order and each layer's low-rank correction that
    together minimise squared weight error, for a given layer-to-bank assignment."""
    layers = list(layers)
    centres: list[Tensor] = []
    for chunk in range(max(assignment) + 1):
        members = [layer for layer, assigned in enumerate(assignment) if assigned == chunk]
        centre = layers[members[len(members) // 2]]
        for _ in range(FIT_ROUNDS if len(members) > 1 else 0):
            if matched:
                for member in members:
                    layers[member] = layers[member][match(layers[member], centre)]
            fitted = Fit([0] * len(members), [centre], [layers[member] for member in members], rank)
            centre = torch.stack(
                [layers[member] - (fitted.rebuilt(index) - centre) for index, member in enumerate(members)]
            ).mean(0)
        centres.append(centre)
    return Fit(assignment, centres, layers, rank)


def detached(module: Module) -> Module:
    """A copy that shares the original's weights and whose submodules can be replaced without
    touching the original."""
    copied = copy.copy(module)
    copied._modules = dict(module._modules)
    copied._parameters = dict(module._parameters)
    return copied


def scaffold(model: Lm) -> Lm:
    student = detached(model)
    student.blocks = nn.ModuleList(detached(block) for block in model.blocks)
    return student


def banked(model: Lm, fitted: Fit) -> Lm:
    student = scaffold(model)
    shared = [tuple(Dense(part.clone()) for part in unstack(centre)) for centre in fitted.centres]
    for layer, block in enumerate(student.blocks):
        corrections = fitted.corrections(layer) or (None, None, None)
        block.feed_forward = FeedForward(
            *(Banked(dense, correction) for dense, correction in zip(shared[fitted.assignment[layer]], corrections))
        )
    return student


def low_rank(model: Lm, rank: int) -> Lm:
    student = scaffold(model)
    for block, step in zip(student.blocks, model.feed_forwards()):
        block.feed_forward = FeedForward(*(truncate(part, rank) for part in unstack(stack(step))))
    return student


def kept_rank(fitted: Fit, layer: int) -> int:
    """The rank at which the layers of one bank matrix, stored low-rank with nothing shared,
    take the bytes that matrix and their corrections take."""
    hidden, columns = fitted.layers[layer].shape
    members = fitted.assignment.count(fitted.assignment[layer])
    width = columns // 3
    return fitted.rank + hidden * width // (members * (hidden + width))


def kept_low_rank(model: Lm, fitted: Fit) -> Lm:
    """The bank's layout with the sharing taken out: a layer the bank stores whole stays whole,
    and every layer that shares a bank matrix is low-rank at kept_rank instead."""
    student = scaffold(model)
    for layer, (block, step) in enumerate(zip(student.blocks, model.feed_forwards())):
        parts = unstack(stack(step))
        if fitted.shares(layer):
            block.feed_forward = FeedForward(*(truncate(part, kept_rank(fitted, layer)) for part in parts))
        else:
            block.feed_forward = FeedForward(*(Dense(part.clone()) for part in parts))
    return student


def weight_error(model: Lm, student: Lm) -> float:
    lost = kept = 0.0
    for original, compressed in zip(model.feed_forwards(), student.feed_forwards()):
        for name in ("gate", "up", "down"):
            weight, other = getattr(original, name), getattr(compressed, name)
            assert isinstance(weight, Dense) and isinstance(other, Dense | LowRank)
            rebuilt = (other.weight if isinstance(other, Dense) else other.dense()).detach()
            lost += float((weight.weight - rebuilt).pow(2).sum())
            kept += float(weight.weight.pow(2).sum())
    return lost / kept


class Router(ScoreRouter):
    """The score router with a per-layer prior that starts every layer on the bank step the fit
    assigned it."""

    def __init__(self, width: int, assignment: list[int], chunks: int) -> None:
        super().__init__(ModelConfig(len(assignment), width), Routed(chunks))
        self.prior = nn.Parameter(PRIOR * F.one_hot(torch.tensor(assignment), chunks).float())

    def logits(self, x: Tensor, module: int) -> Tensor:
        return super().logits(x, module) + self.prior[module]


class RoutedFeedForward(nn.Module):
    def __init__(
        self,
        chunks: list[tuple[Dense, Dense, Dense]],
        corrections: tuple[LowRank, LowRank, LowRank] | None,
        router: Router,
        layer: int,
    ) -> None:
        super().__init__()
        self.gates = nn.ModuleList(chunk[0] for chunk in chunks)
        self.ups = nn.ModuleList(chunk[1] for chunk in chunks)
        self.downs = nn.ModuleList(chunk[2] for chunk in chunks)
        self.corrections = None if corrections is None else nn.ModuleList(corrections)
        self.router = router
        self.layer = layer
        self.picked: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        picks = self.router(x, self.layer).hard
        self.picked = picks.detach().argmax(-1)
        gate_fix, up_fix, down_fix = self.corrections or (None, None, None)
        out: Tensor | None = None
        hidden_mix: Tensor | None = None
        for index, (gate, up, down) in enumerate(zip(self.gates, self.ups, self.downs)):
            weight = picks[..., index : index + 1]
            if not weight.requires_grad and not bool(weight.any()):
                continue
            gated, raised = gate(x), up(x)
            if gate_fix is not None and up_fix is not None:
                gated, raised = gated + gate_fix(x), raised + up_fix(x)
            hidden = weight.to(gated.dtype) * (F.silu(gated) * raised)
            lowered = down(hidden)
            out = lowered if out is None else out + lowered
            hidden_mix = hidden if hidden_mix is None else hidden_mix + hidden
        assert out is not None and hidden_mix is not None
        return out if down_fix is None else out + down_fix(hidden_mix)


def routed(model: Lm, fitted: Fit) -> Lm:
    student = scaffold(model)
    chunks = [
        (Dense(gate.clone()), Dense(up.clone()), Dense(down.clone()))
        for gate, up, down in (unstack(centre) for centre in fitted.centres)
    ]
    router = Router(model.embedding.shape[1], fitted.assignment, len(chunks)).to(model.embedding.device)
    for layer, block in enumerate(student.blocks):
        block.feed_forward = RoutedFeedForward(chunks, fitted.corrections(layer), router, layer)
    return student


def stored_bytes(model: nn.Module, bits: int = FLOAT_BITS) -> int:
    total = 0
    for parameter in model.parameters():
        if parameter.dim() == 2 and bits < FLOAT_BITS:
            total += -(-parameter.numel() // GROUP) * (GROUP * bits + 2 * FLOAT_BITS) // 8
        else:
            total += parameter.numel() * FLOAT_BITS // 8
    return total


def rounded(weight: Tensor, bits: int) -> Tensor:
    """Round to nearest on a uniform grid between the lowest and highest weight of every run of
    GROUP consecutive weights, the two ends kept as 16-bit floats."""
    flat = F.pad(weight.flatten(), (0, -weight.numel() % GROUP)).view(-1, GROUP)
    low = flat.amin(1, keepdim=True).half().float()
    high = flat.amax(1, keepdim=True).half().float()
    step = (high - low).clamp(min=1e-12) / (2**bits - 1)
    grid = ((flat - low) / step).round().clamp(0, 2**bits - 1)
    out: Tensor = (low + grid * step).flatten()[: weight.numel()].view_as(weight)
    return out


def quantized(model: Module, bits: int) -> Module:
    copied = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in copied.parameters():
            parameter.copy_(rounded(parameter, bits) if parameter.dim() == 2 else parameter.half().float())
    return copied


def equal_rank(target_bytes: int, model: Lm) -> int:
    """The rank at which a low-rank-only model stores no more than target_bytes."""
    base = stored_bytes(low_rank(model, 1))
    per_rank = stored_bytes(low_rank(model, 2)) - base
    return 1 + (target_bytes - base) // per_rank
