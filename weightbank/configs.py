from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from typing import Any

from weightbank.ledger import STEPS, Cell, Ledger
from weightbank.model import Drawn, Fixed, Layout, ModelConfig, Routed, Settled, Sharing

LEDGER = Ledger.read()
SHARED = "shared"
PICK = "pick"
CHUNKS = "chunks"
DRAW = "draw"
DRAWN = "drawn"
ENDS = "ends"
RANK = "rank"
ARCHITECTURE = "architecture"
SHARING_AXES = (SHARED, PICK, CHUNKS)
BANK_AXES = (*SHARING_AXES, DRAW)
KINDS = {"none": (), "ff": ("feed_forward",), "attn": ("attention",), "both": ("feed_forward", "attention")}
PICKS: dict[str, Callable[[int], Sharing]] = {
    "fixed": Fixed,
    ENDS: lambda chunks: Fixed(chunks, Layout.ENDS),
    "routed": Routed,
    "argmax": lambda chunks: Routed(chunks, noise=False),
    "keys": lambda chunks: Routed(chunks, keys=True),
    "frozen": lambda chunks: Routed(chunks, settled=Settled.PICKS),
    "stopped": lambda chunks: Routed(chunks, settled=Settled.ROUTER),
}


def sharing(pick: str, chunks: int, draw: int) -> Sharing | None:
    if pick == DRAWN:
        return Fixed(chunks, Drawn(draw))
    return PICKS[pick](chunks) if pick in PICKS else None


def model_config(cell: Cell) -> ModelConfig:
    shared, pick, chunks, draw = (cell[axis] for axis in BANK_AXES)
    assert isinstance(shared, str) and isinstance(pick, str) and isinstance(chunks, int) and isinstance(draw, int)
    if shared == "none" and any(cell[axis] != LEDGER.axis(axis).default for axis in (PICK, CHUNKS, RANK, DRAW)):
        raise ValueError(f"{LEDGER.name(cell)}: with nothing shared, {PICK}, {CHUNKS}, {RANK} and {DRAW} stay at their defaults")
    if (pick == DRAWN) != (draw != LEDGER.axis(DRAW).default):
        raise ValueError(f"{LEDGER.name(cell)}: a {DRAW} goes with the {PICK} {DRAWN!r} and with no other")
    settings: dict[str, Any] = {axis: value for axis, value in cell.items() if axis not in (*BANK_AXES, STEPS)}
    unknown = settings.keys() - {field.name for field in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"the model has no setting for the axes {sorted(unknown)}")
    bank = sharing(pick, chunks, draw)
    if shared not in KINDS or bank is None:
        raise ValueError(f"the model cannot build {SHARED} {shared!r} with {PICK} {pick!r}")
    settings.update({kind: bank for kind in KINDS[shared]})
    config = ModelConfig(**settings)
    if isinstance(bank, Fixed) and KINDS[shared]:
        bank.map(config.modules)
    return config


def plain_words(cell: Cell) -> str:
    shares = cell[SHARED] != LEDGER.axis(SHARED).default
    return LEDGER.words(cell, SHARING_AXES if shares else (SHARED,))


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


def other_map(cell: Cell) -> bool:
    return cell[SHARED] != "none" and cell[PICK] in (DRAWN, ENDS)


def rate_source(name: str) -> str:
    if name == OWN_CHUNK:
        return FULL
    cell = LEDGER.parse(name, default(STEPS))
    if other_map(cell):
        return rate_source(LEDGER.name({**cell, PICK: LEDGER.axis(PICK).default, DRAW: LEDGER.axis(DRAW).default}))
    return name
