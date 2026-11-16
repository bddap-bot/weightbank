from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from weightbank.configs import BANK, CONFIGS, FULL, MODULES, NO_ROUTER, OWN_CHUNK
from weightbank.model import VOCAB, Bank, Fixed, ModelConfig, Own, Routed, Sharing, Transformer, mix


def small(feed_forward: Sharing, attention: Sharing) -> Transformer:
    return Transformer(ModelConfig(4, 32, feed_forward=feed_forward, attention=attention))


def tokens(batch: int = 3, length: int = 16) -> Tensor:
    return torch.randint(VOCAB, (batch, length))


def plain_and_own_chunk() -> tuple[Transformer, Transformer]:
    torch.manual_seed(0)
    plain = small(Own(), Own())
    own_chunk = small(Fixed(4), Fixed(4))
    own_chunk.load_state_dict(plain.state_dict())
    return plain, own_chunk


def test_own_chunk_model_computes_what_the_plain_model_computes() -> None:
    plain, own_chunk = plain_and_own_chunk()
    batch = tokens()
    expected, _ = plain(batch)
    actual, trace = own_chunk(batch)
    assert torch.allclose(expected, actual, atol=1e-6)
    for kind in (trace.feed_forward, trace.attention):
        assert [picks.chunk.unique().tolist() for picks in kind] == [[0], [1], [2], [3]]
    assert own_chunk.stored_weights() == plain.stored_weights()


def test_own_chunk_model_gets_the_plain_model_gradients() -> None:
    plain, own_chunk = plain_and_own_chunk()
    batch = tokens()
    for model in (plain, own_chunk):
        model(batch)[0].square().mean().backward()
    for (name, expected), (_, actual) in zip(plain.named_parameters(), own_chunk.named_parameters()):
        assert expected.grad is not None and actual.grad is not None
        assert torch.allclose(expected.grad, actual.grad, atol=1e-6), name


def test_fixed_sharing_cycles_through_the_bank() -> None:
    torch.manual_seed(0)
    _, trace = small(Fixed(2), Fixed(2))(tokens())
    for kind in (trace.feed_forward, trace.attention):
        assert [picks.chunk.unique().tolist() for picks in kind] == [[0], [1], [0], [1]]


def test_training_picks_are_one_hot_and_every_weight_gets_a_gradient() -> None:
    torch.manual_seed(0)
    model = small(Routed(2), Routed(2))
    logits, trace = model(tokens())
    for picks in trace.feed_forward + trace.attention:
        assert picks.hard.requires_grad
        assert bool(((picks.hard == 0) | (picks.hard == 1)).all())
        assert bool((picks.hard.sum(-1) == 1).all())
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_training_picks_are_sampled_and_evaluation_picks_are_not() -> None:
    torch.manual_seed(0)
    model = small(Routed(2), Routed(2))
    batch = tokens(8, 64)

    def first_picks() -> Tensor:
        chunk: Tensor = model(batch)[1].feed_forward[0].chunk
        return chunk

    assert not torch.equal(first_picks(), first_picks())
    model.eval()
    assert torch.equal(first_picks(), first_picks())


def test_routed_output_depends_on_the_pick_at_both_attention_maps() -> None:
    torch.manual_seed(0)
    model = small(Own(), Routed(2)).eval()
    batch = tokens()
    before, _ = model(batch)
    for map_name in ("qkv", "down"):
        chunk = model.attention.chunks[1]
        saved = {name: parameter.clone() for name, parameter in chunk.named_parameters()}
        with torch.no_grad():
            getattr(chunk, map_name).weight.zero_()
        after, trace = model(batch)
        assert any(bool((picks.chunk == 1).any()) for picks in trace.attention)
        assert not torch.allclose(before, after)
        with torch.no_grad():
            for name, parameter in chunk.named_parameters():
                parameter.copy_(saved[name])


def test_router_reads_the_data_and_the_asking_module() -> None:
    torch.manual_seed(0)
    router = small(Routed(4), Own()).feed_forward.router
    assert router is not None
    router.eval()
    with torch.no_grad():
        router.asker.normal_()
        router.data.weight.normal_()
        router.score.weight.normal_()
    x = torch.randn(64, 32)
    assert len(router(x, 0).chunk.unique()) > 1
    assert not torch.equal(router(x, 0).chunk, router(x, 1).chunk)


def test_routed_model_is_causal() -> None:
    torch.manual_seed(0)
    for model in (small(Routed(2), Routed(2)).eval(), small(Own(), Own())):
        batch = tokens(2)
        changed = batch.clone()
        changed[:, -1] = (changed[:, -1] + 1) % VOCAB
        first, _ = model(batch)
        second, _ = model(changed)
        assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
        assert not torch.allclose(first[:, -1], second[:, -1])


def test_banks_store_half_the_routed_weights() -> None:
    def stored(name: str) -> tuple[int, int]:
        model = Transformer(CONFIGS[name])
        return (
            sum(parameter.numel() for chunk in model.feed_forward.chunks for parameter in chunk.parameters()),
            sum(parameter.numel() for chunk in model.attention.chunks for parameter in chunk.parameters()),
        )

    assert MODULES == 2 * BANK
    feed_forward, attention = stored(FULL)
    assert stored("ff-routed") == (feed_forward // 2, attention)
    assert stored("attn-routed") == (feed_forward, attention // 2)
    assert stored("both-routed") == (feed_forward // 2, attention // 2)
    assert stored(NO_ROUTER) == (feed_forward // 2, attention // 2)
    assert stored(OWN_CHUNK) == (feed_forward, attention)
    assert Transformer(CONFIGS[OWN_CHUNK]).stored_weights() == Transformer(CONFIGS[FULL]).stored_weights()


def test_mix_sends_a_gradient_to_the_pick_of_a_chunk_no_token_picked() -> None:
    torch.manual_seed(0)
    chunks = [nn.Linear(8, 8), nn.Linear(8, 8)]
    picks = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    torch.autograd.backward(mix(chunks, picks, torch.randn(3, 8)).square().sum())
    assert picks.grad is not None
    assert bool((picks.grad[:, 1].abs() > 0).all())


def test_routed_model_computes_the_plain_model_built_from_the_chunks_each_module_picks() -> None:
    torch.manual_seed(0)
    assignments = {"attention": [1, 0, 0, 1], "feed_forward": [0, 0, 1, 1]}
    routed = small(Routed(2), Routed(2)).eval()
    plain = small(Own(), Own()).eval()
    state = {
        name: value
        for name, value in routed.state_dict().items()
        if name.split(".")[0] not in assignments
    }
    for kind, assignment in assignments.items():
        bank = getattr(routed, kind)
        assert isinstance(bank, Bank) and bank.router is not None
        with torch.no_grad():
            for parameter in bank.router.parameters():
                parameter.zero_()
            for chunk in range(2):
                bank.router.score.weight[chunk, chunk] = 1.0
            for module, chunk in enumerate(assignment):
                bank.router.asker[module, chunk] = 5.0
        for module, chunk in enumerate(assignment):
            for name, value in bank.chunks[chunk].state_dict().items():
                state[f"{kind}.registered.{module}.{name}"] = value
    plain.load_state_dict(state)
    batch = tokens()
    expected, _ = plain(batch)
    actual, trace = routed(batch)
    assert torch.allclose(expected, actual, atol=1e-6)
    for kind, assignment in assignments.items():
        assert [picks.chunk.unique().tolist() for picks in getattr(trace, kind)] == [[chunk] for chunk in assignment]


def test_residual_output_weights_start_scaled_down_by_depth() -> None:
    torch.manual_seed(0)
    expected = 0.02 / math.sqrt(2 * 4)
    for sharing, chunks in ((Own(), 4), (Routed(2), 2), (Fixed(4), 4)):
        model = small(sharing, sharing)
        scaled = [parameter for name, parameter in model.named_parameters() if name.endswith("down.weight")]
        assert len(scaled) == 2 * chunks
        for parameter in scaled:
            assert abs(parameter.std().item() / expected - 1) < 0.1
