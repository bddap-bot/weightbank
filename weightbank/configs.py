from __future__ import annotations

from weightbank.model import ModelConfig, Weights

MODULES = 8
WIDTH = 256
BANK = MODULES // 2
SEEDS = (0, 1, 2)

PLAIN_WIDTHS = (128, 160, 192, 224)
PLAIN_DEPTHS = (4, 6)

FULL = "full"
ROUTED = ("ff-routed", "attn-routed", "both-routed")
PINNED = "pinned"

CONFIGS: dict[str, ModelConfig] = {
    FULL: ModelConfig(MODULES, WIDTH),
    **{f"plain-w{width}": ModelConfig(MODULES, width) for width in PLAIN_WIDTHS},
    **{f"plain-l{depth}": ModelConfig(depth, WIDTH) for depth in PLAIN_DEPTHS},
    "ff-routed": ModelConfig(MODULES, WIDTH, feed_forward=Weights.ROUTED, bank=BANK),
    "attn-routed": ModelConfig(MODULES, WIDTH, attention=Weights.ROUTED, bank=BANK),
    "both-routed": ModelConfig(
        MODULES, WIDTH, feed_forward=Weights.ROUTED, attention=Weights.ROUTED, bank=BANK
    ),
    PINNED: ModelConfig(MODULES, WIDTH, feed_forward=Weights.PINNED, attention=Weights.PINNED),
}
