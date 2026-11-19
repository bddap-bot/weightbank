from __future__ import annotations

from weightbank.model import Fixed, ModelConfig, Routed

MODULES = 8
WIDTH = 256
BANK = MODULES // 2
SEEDS = (0, 1, 2)

FULL = "full"
NARROWER = {f"plain-w{width}": ModelConfig(MODULES, width) for width in (96, 128, 160, 192, 224)}
SHALLOWER = {f"plain-l{depth}": ModelConfig(depth, WIDTH) for depth in (1, 2, 4, 6)}
ROUTED = {
    "ff-routed": ModelConfig(MODULES, WIDTH, feed_forward=Routed(BANK)),
    "attn-routed": ModelConfig(MODULES, WIDTH, attention=Routed(BANK)),
    "both-routed": ModelConfig(MODULES, WIDTH, feed_forward=Routed(BANK), attention=Routed(BANK)),
}
NOISE_FREE = "both-routed-argmax"
KEYS = "both-routed-keys"
ROUTER_VARIANTS = {
    NOISE_FREE: ModelConfig(
        MODULES, WIDTH, feed_forward=Routed(BANK, noise=False), attention=Routed(BANK, noise=False)
    ),
    KEYS: ModelConfig(MODULES, WIDTH, feed_forward=Routed(BANK, keys=True), attention=Routed(BANK, keys=True)),
}
OWN_CHUNK = "own-chunk"
NO_ROUTER = "both-no-router"
FIXED_MAP = {
    1: "both-no-router-c1",
    2: "both-no-router-c2",
    BANK: NO_ROUTER,
    MODULES: OWN_CHUNK,
}

CONFIGS: dict[str, ModelConfig] = {
    FULL: ModelConfig(MODULES, WIDTH),
    **NARROWER,
    **SHALLOWER,
    **ROUTED,
    **ROUTER_VARIANTS,
    **{name: ModelConfig(MODULES, WIDTH, feed_forward=Fixed(chunks), attention=Fixed(chunks)) for chunks, name in FIXED_MAP.items()},
}
