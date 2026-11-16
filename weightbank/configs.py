from __future__ import annotations

from weightbank.model import Fixed, ModelConfig, Routed

MODULES = 8
WIDTH = 256
BANK = MODULES // 2
SEEDS = (0, 1, 2)

FULL = "full"
NARROWER = {f"plain-w{width}": ModelConfig(MODULES, width) for width in (128, 160, 192, 224)}
SHALLOWER = {f"plain-l{depth}": ModelConfig(depth, WIDTH) for depth in (4, 6)}
ROUTED = {
    "ff-routed": ModelConfig(MODULES, WIDTH, feed_forward=Routed(BANK)),
    "attn-routed": ModelConfig(MODULES, WIDTH, attention=Routed(BANK)),
    "both-routed": ModelConfig(MODULES, WIDTH, feed_forward=Routed(BANK), attention=Routed(BANK)),
}
OWN_CHUNK = "own-chunk"
NO_ROUTER = "both-no-router"

CONFIGS: dict[str, ModelConfig] = {
    FULL: ModelConfig(MODULES, WIDTH),
    **NARROWER,
    **SHALLOWER,
    **ROUTED,
    OWN_CHUNK: ModelConfig(MODULES, WIDTH, feed_forward=Fixed(MODULES), attention=Fixed(MODULES)),
    NO_ROUTER: ModelConfig(MODULES, WIDTH, feed_forward=Fixed(BANK), attention=Fixed(BANK)),
}
