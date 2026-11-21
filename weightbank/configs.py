from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

from weightbank.ledger import STEPS, Cell, Ledger
from weightbank.model import Fixed, ModelConfig, Routed, Settled, Sharing

LEDGER = Ledger.read()
SHARED = "shared"
PICK = "pick"
CHUNKS = "chunks"
RANK = "rank"
SHARING_AXES = (SHARED, PICK, CHUNKS)
KINDS = {"none": (), "ff": ("feed_forward",), "attn": ("attention",), "both": ("feed_forward", "attention")}
PICKS: dict[str, Callable[[int], Sharing]] = {
    "fixed": Fixed,
    "routed": Routed,
    "argmax": lambda chunks: Routed(chunks, noise=False),
    "keys": lambda chunks: Routed(chunks, keys=True),
    "frozen": lambda chunks: Routed(chunks, settled=Settled.PICKS),
    "stopped": lambda chunks: Routed(chunks, settled=Settled.ROUTER),
}


def model_config(cell: Cell) -> ModelConfig:
    shared, pick, chunks = (cell[axis] for axis in SHARING_AXES)
    assert isinstance(shared, str) and isinstance(pick, str) and isinstance(chunks, int)
    if shared == "none" and any(cell[axis] != LEDGER.axis(axis).default for axis in (PICK, CHUNKS, RANK)):
        raise ValueError(f"{LEDGER.name(cell)}: with nothing shared, {PICK}, {CHUNKS} and {RANK} stay at their defaults")
    settings: dict[str, Any] = {axis: value for axis, value in cell.items() if axis not in (*SHARING_AXES, STEPS)}
    unknown = settings.keys() - {field.name for field in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"the model has no setting for the axes {sorted(unknown)}")
    if shared not in KINDS or pick not in PICKS:
        raise ValueError(f"the model cannot build {SHARED} {shared!r} with {PICK} {pick!r}")
    settings.update({kind: PICKS[pick](chunks) for kind in KINDS[shared]})
    return ModelConfig(**settings)


def named(**changed: int | str) -> str:
    return LEDGER.name(LEDGER.cell(changed))


def default(axis: str) -> int:
    value = LEDGER.axis(axis).default
    assert isinstance(value, int)
    return value


def config(name: str) -> ModelConfig:
    return model_config(LEDGER.parse(name, default(STEPS)))


MODULES = default("modules")
WIDTH = default("width")
BANK = default(CHUNKS)

FULL = named()
NARROWER = tuple(named(width=width) for width in (96, 128, 160, 192, 224))
SHALLOWER = tuple(named(modules=depth) for depth in (1, 2, 4, 6))
ROUTED = tuple(named(shared=shared, pick="routed") for shared in ("ff", "attn", "both"))
BOTH_ROUTED = named(shared="both", pick="routed")
NOISE_FREE = named(shared="both", pick="argmax")
KEYS = named(shared="both", pick="keys")
ROUTER_VARIANTS = (NOISE_FREE, KEYS)
FIXED_MAP = {chunks: named(shared="both", chunks=chunks) for chunks in (1, 2, BANK, MODULES)}
NO_ROUTER = FIXED_MAP[BANK]
OWN_CHUNK = FIXED_MAP[MODULES]
ROUNDS_ONE_AND_TWO = (FULL, *NARROWER, *SHALLOWER, *ROUTED, *ROUTER_VARIANTS, *FIXED_MAP.values())
