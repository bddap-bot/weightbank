from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler

from weightbank import train
from weightbank.configs import BANK, FIXED_MAP, FULL, KEYS, MODULES, NO_ROUTER, NOISE_FREE, OWN_CHUNK, config, named
from weightbank.model import (
    CONTEXT,
    FREED_LEAN,
    HELD_LEAN,
    MODERN,
    SETTLE_PART,
    VOCAB,
    Bank,
    Drawn,
    FeedForward,
    Fixed,
    Layout,
    KeyRouter,
    Lean,
    ModelConfig,
    Own,
    RootMeanSquareNorm,
    Routed,
    ScoreRouter,
    Settled,
    Sharing,
    Transformer,
    attend,
    mix,
    pick,
    rotate,
)


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


def test_a_drawn_map_is_the_same_whenever_it_is_drawn_and_uses_every_chunk() -> None:
    assert [Fixed(4, Drawn(draw)).map(8) for draw in (1, 2, 3, 4)] == [
        (0, 3, 3, 1, 1, 1, 2, 3),
        (3, 3, 0, 0, 3, 2, 2, 1),
        (0, 2, 1, 2, 2, 0, 0, 3),
        (3, 0, 2, 1, 0, 0, 0, 3),
    ]
    for draw in range(1, 200):
        for modules, chunks in ((8, 4), (16, 4), (4, 4), (5, 1)):
            assert set(Fixed(chunks, Drawn(draw)).map(modules)) == set(range(chunks))
    assert len({Fixed(4, Drawn(draw)).map(8) for draw in range(1, 200)}) > 150
    with pytest.raises(ValueError, match="cannot use all"):
        Fixed(8, Drawn(1)).map(4)
    with pytest.raises(ValueError, match="numbered from 1"):
        Drawn(0)


def test_own_ends_give_the_first_and_last_module_a_chunk_no_other_uses() -> None:
    assert Fixed(4, Layout.ENDS).map(8) == (0, 1, 2, 1, 2, 1, 2, 3)
    assert Fixed(3, Layout.ENDS).map(5) == (0, 1, 1, 1, 2)
    assert Fixed(4).map(8) == (0, 1, 2, 3, 0, 1, 2, 3)
    with pytest.raises(ValueError, match="at least 3"):
        Fixed(2, Layout.ENDS).map(8)


def test_a_model_computes_with_the_chunk_its_map_names_and_stores_what_the_mod_map_stores() -> None:
    torch.manual_seed(0)
    drawn = Fixed(4, Drawn(1))
    model = Transformer(ModelConfig(8, 32, feed_forward=drawn, attention=drawn))
    _, trace = model(tokens())
    for kind in (trace.feed_forward, trace.attention):
        assert [picks.chunk.unique().tolist() for picks in kind] == [[chunk] for chunk in drawn.map(8)]
    mod = Transformer(ModelConfig(8, 32, feed_forward=Fixed(4), attention=Fixed(4)))
    assert model.stored_weights() == mod.stored_weights()
    mod.load_state_dict(model.state_dict())
    assert not torch.allclose(mod(tokens())[0], model(tokens())[0])


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
    assert isinstance(router, ScoreRouter)
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
        model = Transformer(config(name))
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
    assert Transformer(config(OWN_CHUNK)).stored_weights() == Transformer(config(FULL)).stored_weights()


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
        assert isinstance(bank, Bank) and isinstance(bank.router, ScoreRouter)
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


def test_a_pick_without_noise_is_the_highest_score_and_passes_the_softmax_gradient() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 4, requires_grad=True)
    weights = torch.randn(5, 4)
    picks = pick(logits, sample=False, straight_through=True)
    assert torch.equal(picks.chunk, logits.argmax(-1))
    assert torch.allclose(picks.hard, torch.nn.functional.one_hot(logits.argmax(-1), 4).float(), atol=1e-6)
    (through,) = torch.autograd.grad((picks.hard * weights).sum(), logits)
    (soft,) = torch.autograd.grad((logits.softmax(-1) * weights).sum(), logits)
    assert torch.allclose(through, soft, atol=1e-6)
    assert bool(through.abs().sum() > 0)
    assert not pick(logits, sample=False, straight_through=False).hard.requires_grad


def test_noise_free_training_picks_are_the_evaluation_picks_and_every_weight_gets_a_gradient() -> None:
    torch.manual_seed(0)
    model = small(Routed(2, noise=False), Routed(2, noise=False))
    batch = tokens(8, 64)
    logits, trace = model(batch)
    again = model(batch)[1]
    model.eval()
    evaluated = model(batch)[1]
    model.train()
    for kind in ("feed_forward", "attention"):
        for picks, second, third in zip(getattr(trace, kind), getattr(again, kind), getattr(evaluated, kind)):
            assert picks.hard.requires_grad
            assert torch.equal(picks.chunk, second.chunk)
            assert torch.equal(picks.chunk, third.chunk)
    chunks = torch.cat([picks.chunk.flatten() for picks in trace.feed_forward])
    assert bool(chunks.min() < chunks.max())
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_key_router_scores_each_chunk_by_the_dot_product_of_a_query_with_its_key() -> None:
    torch.manual_seed(0)
    router = small(Routed(4, keys=True), Own()).feed_forward.router
    assert isinstance(router, KeyRouter)
    x = torch.randn(64, 32)
    query = (x + router.asker[1]) @ router.query.weight.T + router.query.bias
    expected = torch.stack([query @ key for key in router.keys], dim=-1)
    assert torch.allclose(router.logits(x, 1), expected, atol=1e-6)
    router.eval()
    assert torch.equal(router(x, 1).chunk, expected.argmax(-1))
    with torch.no_grad():
        router.asker.normal_()
        router.keys.normal_()
    assert len(router(x, 0).chunk.unique()) > 1
    assert not torch.equal(router(x, 0).chunk, router(x, 1).chunk)


def test_key_router_samples_in_training_and_every_weight_gets_a_gradient() -> None:
    torch.manual_seed(0)
    model = small(Routed(2, keys=True), Routed(2, keys=True))
    batch = tokens(8, 64)
    logits, trace = model(batch)
    assert not torch.equal(trace.feed_forward[0].chunk, model(batch)[1].feed_forward[0].chunk)
    for picks in trace.feed_forward + trace.attention:
        assert bool((picks.hard.sum(-1) == 1).all())
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0), name


def test_router_variants_store_the_banks_of_the_routed_model() -> None:
    routed = Transformer(config("both-routed"))
    noise_free = Transformer(config(NOISE_FREE))
    keys = Transformer(config(KEYS))
    assert noise_free.stored_weights() == routed.stored_weights()
    assert isinstance(noise_free.attention.router, ScoreRouter) and not noise_free.attention.router.noise
    assert isinstance(routed.attention.router, ScoreRouter) and routed.attention.router.noise
    assert isinstance(keys.attention.router, KeyRouter) and isinstance(keys.feed_forward.router, KeyRouter)
    assert keys.attention.router.noise

    def routers(model: Transformer) -> int:
        return sum(
            parameter.numel()
            for bank in (model.attention, model.feed_forward)
            if bank.router is not None
            for parameter in bank.router.parameters()
        )

    assert keys.stored_weights() - routers(keys) == routed.stored_weights() - routers(routed)


def test_the_fixed_map_stores_one_plain_layer_per_chunk_and_every_module_keeps_its_norms() -> None:
    norms = 2 * 2 * 256
    for chunks, name in FIXED_MAP.items():
        model = Transformer(config(name))
        assert len(model.attention.chunks) == len(model.feed_forward.chunks) == chunks
        plain = Transformer(config(FULL if chunks == MODULES else named(modules=chunks)))
        assert model.stored_weights() == plain.stored_weights() + (MODULES - chunks) * norms
        _, trace = model(tokens())
        for kind in (trace.feed_forward, trace.attention):
            assert [picks.chunk.unique().tolist() for picks in kind] == [[module % chunks] for module in range(MODULES)]


def test_both_router_forms_start_near_indifferent_at_the_same_scale() -> None:
    torch.manual_seed(0)
    x = torch.nn.functional.layer_norm(torch.randn(2048, 256), (256,))
    spreads = []
    for keys in (False, True):
        router = Transformer(ModelConfig(8, 256, feed_forward=Routed(4, keys=keys))).feed_forward.router
        assert router is not None
        logits = router.logits(x, 0).detach()
        assert float(logits.softmax(-1).amax(-1).mean()) < 0.3
        spreads.append(float(logits.std(-1).mean()))
    assert 0.5 < spreads[1] / spreads[0] < 2


def test_a_correction_starts_as_no_change_and_then_changes_only_its_module() -> None:
    torch.manual_seed(0)
    shared = small(Fixed(1), Fixed(1))
    corrected = Transformer(ModelConfig(4, 32, feed_forward=Fixed(1), attention=Fixed(1), rank=3))
    corrected.load_state_dict(shared.state_dict(), strict=False)
    batch = tokens()
    assert torch.equal(corrected(batch)[0], shared(batch)[0])
    logits, _ = corrected(batch)
    logits.square().mean().backward()
    for bank in (corrected.feed_forward, corrected.attention):
        assert bank.corrections is not None and len(bank.corrections) == 4
        for correction in bank.corrections:
            for part in (correction.first, correction.second):
                assert part.write.grad is not None and bool(part.write.grad.abs().sum() > 0)
    last = corrected.feed_forward.corrections
    assert last is not None
    with torch.no_grad():
        last[3].second.write.normal_()
    changed, _ = corrected(batch)
    assert not torch.allclose(changed, logits)
    with torch.no_grad():
        last[3].second.write.zero_()
        last[0].first.write.normal_()
    assert not torch.allclose(corrected(batch)[0], logits)


def test_a_correction_applies_under_a_router_and_needs_a_bank() -> None:
    torch.manual_seed(0)
    model = Transformer(ModelConfig(4, 32, feed_forward=Routed(2), rank=2))
    assert model.attention.corrections is None and model.feed_forward.corrections is not None
    model.eval()
    batch = tokens()
    before, _ = model(batch)
    with torch.no_grad():
        model.feed_forward.corrections[1].first.write.normal_()
    assert not torch.allclose(model(batch)[0], before)
    try:
        ModelConfig(4, 32, rank=2)
    except ValueError:
        return
    raise AssertionError("a correction with nothing shared was accepted")


def test_a_corrected_bank_stores_the_weights_of_the_bank_twice_its_size() -> None:
    def stored(**cell: int | str) -> int:
        return Transformer(config(named(**cell))).stored_weights()

    assert stored(shared="both", chunks=2) - stored(shared="both", chunks=1, rank=24) == 2304
    assert stored(shared="both", chunks=1, rank=24) - stored(shared="both", chunks=1) == 8 * 24 * (2 * 1280 + 1024 + 512)


def modern(feed_forward: Sharing = Own(), attention: Sharing = Own(), modules: int = 4, width: int = 32) -> Transformer:
    return Transformer(ModelConfig(modules, width, feed_forward=feed_forward, attention=attention, architecture=MODERN))


def test_the_modern_architecture_has_no_position_table_and_stores_about_the_classic_weights() -> None:
    model = modern(modules=8, width=256)
    classic = Transformer(ModelConfig(8, 256))
    assert model.position is None and classic.position is not None
    assert all(isinstance(layer, RootMeanSquareNorm) for layer in (*model.attention_norm, *model.feed_forward_norm, model.final_norm))
    assert model.feed_forward.chunks[0].up.out_features == 2 * model.feed_forward.chunks[0].down.in_features
    positions = CONTEXT * 256
    assert abs(model.stored_weights() - (classic.stored_weights() - positions)) < 0.005 * classic.stored_weights()
    try:
        ModelConfig(8, 256, architecture="other")
    except ValueError:
        return
    raise AssertionError("an unknown architecture was accepted")


def test_the_modern_model_is_causal_and_reads_order_only_through_rotation() -> None:
    torch.manual_seed(0)
    for model in (modern(), modern(Fixed(2), Routed(2))):
        model.eval()
        first = tokens(2, 12)
        second = first.clone()
        second[:, -1] = (second[:, -1] + 1) % VOCAB
        assert torch.allclose(model(first)[0][:, :-1], model(second)[0][:, :-1], atol=1e-5)
        swapped = first.clone()
        swapped[:, 0], swapped[:, 1] = first[:, 1], first[:, 0]
        assert not torch.allclose(model(first)[0][:, -1], model(swapped)[0][:, -1], atol=1e-5)


def test_rotation_keeps_length_and_scores_depend_on_distance_alone() -> None:
    torch.manual_seed(0)
    query, key = torch.randn(8), torch.randn(8)
    turned_query, turned_key = rotate(query.expand(10, 8)), rotate(key.expand(10, 8))
    assert torch.allclose(turned_query.norm(dim=-1), query.norm().expand(10), atol=1e-5)
    scores = turned_query @ turned_key.T
    assert torch.allclose(scores[5, 3], scores[9, 7], atol=1e-5) and torch.allclose(scores[2, 0], scores[5, 3], atol=1e-5)
    assert not torch.allclose(scores[5, 3], scores[5, 4], atol=1e-5)


def test_the_root_mean_square_norm_scales_to_unit_size_times_its_weight() -> None:
    norm = RootMeanSquareNorm(16)
    out = norm(torch.randn(5, 16) * 7)
    assert torch.allclose(out.square().mean(-1), torch.ones(5), atol=1e-4)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    assert torch.allclose(norm(torch.randn(5, 16)).square().mean(-1), torch.full((5,), 4.0), atol=1e-3)


def test_rotation_turns_keys_as_well_as_queries() -> None:
    torch.manual_seed(0)
    same = torch.randn(8).expand(1, 6, 8)
    qkv = torch.cat((same, same, torch.randn(1, 6, 8)), dim=-1)
    assert not torch.allclose(attend(qkv, True), attend(qkv, False), atol=1e-5)


def test_the_gate_multiplies_the_value() -> None:
    torch.manual_seed(0)
    step = FeedForward(ModelConfig(4, 32, architecture=MODERN))
    hidden = step.down.in_features
    assert not torch.allclose(step(torch.randn(3, 32)), step.down.bias.expand(3, 32))
    with torch.no_grad():
        step.up.weight[hidden:].zero_()
        step.up.bias[hidden:].zero_()
    assert torch.allclose(step(torch.randn(3, 32)), step.down.bias.expand(3, 32))


def text() -> Tensor:
    return torch.randint(VOCAB, (4 * CONTEXT + 1,), dtype=torch.uint8)


def test_settling_picks_fixes_each_module_to_its_most_common_chunk() -> None:
    torch.manual_seed(0)
    model = small(Routed(4, settled=Settled.PICKS), Routed(4, settled=Settled.PICKS))
    corpus = text()
    seen = train.evaluate(model, corpus)
    assert seen.feed_forward is not None and seen.attention is not None
    common = [[row.index(max(row)) for row in stats.counts] for stats in (seen.feed_forward, seen.attention)]
    train.settle(model, corpus)
    assert [model.feed_forward.settled_picks, model.attention.settled_picks] == common
    _, trace = model(tokens())
    for kind, expected in ((trace.feed_forward, common[0]), (trace.attention, common[1])):
        assert [picks.chunk.unique().tolist() for picks in kind] == [[chunk] for chunk in expected]


def test_settling_the_router_stops_its_learning_and_keeps_picks_by_input() -> None:
    torch.manual_seed(0)
    model = small(Routed(4, settled=Settled.ROUTER), Own())
    train.settle(model, text())
    assert model.feed_forward.settled_picks is None and model.feed_forward.router is not None
    logits, trace = model(tokens())
    logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.feed_forward.router.parameters())
    assert all(parameter.grad is not None for chunk in model.feed_forward.chunks for parameter in chunk.parameters())
    assert any(len(picks.chunk.unique()) > 1 for picks in trace.feed_forward)


def still(model: Transformer) -> None:
    for bank in (model.feed_forward, model.attention):
        for parameter in () if bank.router is None else bank.router.parameters():
            nn.init.zeros_(parameter)


MOD_MAP = [[0], [1], [0], [1]]


def test_a_freed_pick_is_the_mod_map_until_a_quarter_of_training_then_leans_on_it_less_each_step() -> None:
    freed = Routed(2, lean=Lean.FREED)
    steps = 8 * SETTLE_PART
    assert [freed.lean_at(step, steps) for step in (0, 7)] == [None, None]
    assert [freed.lean_at(step, steps) for step in (8, 12, 16, 31, steps)] == [FREED_LEAN, FREED_LEAN / 2, 0.0, 0.0, 0.0]
    torch.manual_seed(0)
    model = small(freed, freed)
    logits, trace = model(tokens())
    logits.square().mean().backward()
    for bank, kind in ((model.feed_forward, trace.feed_forward), (model.attention, trace.attention)):
        assert bank.router is not None and all(parameter.grad is None for parameter in bank.router.parameters())
        assert [picks.chunk.unique().tolist() for picks in kind] == MOD_MAP


def test_training_leaves_a_freed_router_alone_for_a_quarter_and_trains_it_after() -> None:
    torch.manual_seed(0)
    model = small(Routed(2, lean=Lean.FREED), Routed(2, lean=Lean.FREED))
    plan = train.Plan("unused", 0, 2 * SETTLE_PART, 1e-3)
    optimizer = train.optimizer_for(model, plan.peak_rate)
    scaler = GradScaler("cpu", enabled=False)
    routers = [bank.router for bank in (model.feed_forward, model.attention)]
    assert all(router is not None for router in routers)
    before = [parameter.clone() for router in routers if router is not None for parameter in router.parameters()]

    def moved() -> list[bool]:
        now = [parameter for router in routers if router is not None for parameter in router.parameters()]
        return [not torch.equal(was, parameter) for was, parameter in zip(before, now, strict=True)]

    leans = []
    for step in range(plan.steps):
        if step == plan.steps // SETTLE_PART:
            assert not any(moved())
        train.step_at(model, optimizer, scaler, tokens(), tokens(), step, plan)
        leans.append(model.feed_forward.lean)
        assert model.attention.lean == leans[-1]
    assert leans == [None, None, FREED_LEAN, FREED_LEAN / 2, 0.0, 0.0, 0.0, 0.0]
    assert all(moved())


def test_a_freed_pick_ends_as_the_router_alone() -> None:
    torch.manual_seed(0)
    freed, plain = small(Routed(2, lean=Lean.FREED), Own()), small(Routed(2), Own())
    plain.load_state_dict(freed.state_dict())
    freed.advance(7, 8)
    freed.eval()
    plain.eval()
    batch = tokens()
    assert torch.equal(freed(batch)[0], plain(batch)[0])


def test_a_lean_adds_to_the_score_of_the_modules_chunk_in_the_mod_map() -> None:
    torch.manual_seed(0)
    model = small(Routed(2, lean=Lean.HELD), Routed(2, lean=Lean.HELD))
    still(model)
    for step in (0, 5, 8):
        model.advance(step, 8)
        assert model.feed_forward.lean == model.attention.lean == HELD_LEAN
    _, trace = model(torch.randint(VOCAB, (64, CONTEXT)))
    expected = math.exp(HELD_LEAN) / (math.exp(HELD_LEAN) + 1)
    for kind in (trace.feed_forward, trace.attention):
        for module, picks in enumerate(kind):
            assert (picks.chunk == module % 2).float().mean().item() == pytest.approx(expected, abs=0.02)
            assert picks.confidence.mean().item() == pytest.approx(expected)
    model.eval()
    _, trace = model(tokens())
    assert [picks.chunk.unique().tolist() for picks in trace.feed_forward] == MOD_MAP


def test_a_router_that_scores_another_chunk_above_the_lean_picks_it_and_learns() -> None:
    torch.manual_seed(0)
    model = small(Routed(4, lean=Lean.HELD), Own())
    still(model)
    router = model.feed_forward.router
    assert isinstance(router, ScoreRouter)
    with torch.no_grad():
        router.score.bias[3] = HELD_LEAN + 1.0
    model.eval()
    _, trace = model(tokens())
    assert [picks.chunk.unique().tolist() for picks in trace.feed_forward] == [[3]] * 4
    model.train()
    logits, _ = model(tokens())
    logits.square().mean().backward()
    assert router.score.bias.grad is not None and bool(router.score.bias.grad.any())


def test_a_pick_with_no_lean_never_leans() -> None:
    model = small(Routed(4), Routed(4, settled=Settled.ROUTER))
    for step in (0, 3, 8):
        model.advance(step, 8)
        assert model.feed_forward.lean == model.attention.lean == 0.0


def test_the_classic_tables_start_as_they_always_have() -> None:
    torch.manual_seed(0)
    model = small(Own(), Own())
    torch.manual_seed(0)
    token, position = nn.Embedding(VOCAB, 32), nn.Embedding(CONTEXT, 32)
    for embedding in (token, position):
        nn.init.normal_(embedding.weight, std=0.02)
    assert model.position is not None
    assert torch.equal(model.token.weight, token.weight) and torch.equal(model.position.weight, position.weight)


def test_a_model_that_settles_nothing_is_left_alone() -> None:
    torch.manual_seed(0)
    model = small(Routed(4), Fixed(2))
    train.settle(model, torch.empty(0, dtype=torch.uint8))
    assert model.feed_forward.settled_picks is None and model.feed_forward.router is not None
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_a_training_step_unscales_then_clips_at_one_then_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = small(Own(), Own())
    optimizer = train.optimizer_for(model, 1e-3)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [train.WEIGHT_DECAY, 0.0]
    matrices, vectors = (group["params"] for group in optimizer.param_groups)
    assert all(parameter.dim() >= 2 for parameter in matrices) and all(parameter.dim() < 2 for parameter in vectors)
    assert len(matrices) + len(vectors) == len(list(model.parameters()))
    scaler = GradScaler("cpu", enabled=False)
    calls: list[object] = []
    monkeypatch.setattr(scaler, "unscale_", lambda optimizer: calls.append("unscale"))
    monkeypatch.setattr(scaler, "step", lambda optimizer: calls.append("step"))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda parameters, norm: calls.append(("clip", norm)))
    loss = train.step_once(model, optimizer, scaler, tokens(), tokens())
    assert calls == ["unscale", ("clip", 1.0), "step"]
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.1)
