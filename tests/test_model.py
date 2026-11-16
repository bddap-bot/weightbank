from __future__ import annotations

import torch

from weightbank.configs import BANK, CONFIGS, FULL, MODULES, PINNED, ROUTED
from weightbank.model import ModelConfig, Transformer, Weights


def small(feed_forward: Weights, attention: Weights, bank: int = 0) -> ModelConfig:
    return ModelConfig(4, 32, feed_forward=feed_forward, attention=attention, bank=bank, context=16)


def test_pinned_model_computes_what_the_plain_model_computes() -> None:
    torch.manual_seed(0)
    plain = Transformer(small(Weights.OWN, Weights.OWN))
    pinned = Transformer(small(Weights.PINNED, Weights.PINNED))
    pinned.load_state_dict(plain.state_dict())
    tokens = torch.randint(256, (3, 16))
    expected, _ = plain(tokens)
    actual, picks = pinned(tokens)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert [int(pick.unique().item()) for pick in picks.feed_forward] == [0, 1, 2, 3]
    assert pinned.stored_weights() == plain.stored_weights()


def test_pinned_model_gets_the_plain_model_gradients() -> None:
    torch.manual_seed(0)
    plain = Transformer(small(Weights.OWN, Weights.OWN))
    pinned = Transformer(small(Weights.PINNED, Weights.PINNED))
    pinned.load_state_dict(plain.state_dict())
    tokens = torch.randint(256, (3, 16))
    for model in (plain, pinned):
        model(tokens)[0].square().mean().backward()
    for (name, expected), (_, actual) in zip(plain.named_parameters(), pinned.named_parameters()):
        assert expected.grad is not None and actual.grad is not None
        assert torch.allclose(expected.grad, actual.grad, atol=1e-6), name


def test_routed_picks_are_hard_and_every_chunk_gets_a_gradient_in_training() -> None:
    torch.manual_seed(0)
    model = Transformer(small(Weights.ROUTED, Weights.ROUTED, bank=2))
    tokens = torch.randint(256, (3, 16))
    logits, picks = model(tokens)
    assert len(picks.feed_forward) == len(picks.attention) == 4
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_routed_evaluation_is_deterministic_and_causal() -> None:
    torch.manual_seed(0)
    model = Transformer(small(Weights.ROUTED, Weights.ROUTED, bank=2)).eval()
    tokens = torch.randint(256, (2, 16))
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 256
    first, _ = model(tokens)
    second, _ = model(changed)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert torch.equal(first, model(tokens)[0])


def test_routed_banks_halve_the_routed_weights() -> None:
    def blocks(name: str) -> tuple[int, int]:
        model = Transformer(CONFIGS[name])
        return (
            sum(parameter.numel() for parameter in model.feed_forward.parameters()),
            sum(parameter.numel() for parameter in model.attention.parameters()),
        )

    assert MODULES == 2 * BANK
    full_ff, full_attention = blocks(FULL)
    assert blocks("ff-routed") == (full_ff // 2, full_attention)
    assert blocks("attn-routed") == (full_ff, full_attention // 2)
    assert blocks("both-routed") == (full_ff // 2, full_attention // 2)
    assert blocks(PINNED) == (full_ff, full_attention)
    assert set(ROUTED) < set(CONFIGS)
