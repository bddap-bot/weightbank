from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from weightbank.compress import bank, heal, lm, run
from weightbank.compress.run import Arm, Full, Trial
from weightbank.data import windows

WIDTH = 18
HIDDEN = 64
VOCAB = 40
torch.set_num_threads(2)


def tiny(seed: int = 0) -> lm.Lm:
    generator = torch.Generator().manual_seed(seed)

    def matrix(rows: int, columns: int) -> Tensor:
        return torch.randn(rows, columns, generator=generator) / float(columns**0.5)

    key_width = WIDTH * lm.KEY_VALUE_HEADS // lm.HEADS
    weights = {"model.embed_tokens.weight": matrix(VOCAB, WIDTH), "model.norm.weight": torch.ones(WIDTH)}
    for layer in range(lm.LAYERS):
        prefix = f"model.layers.{layer}."
        weights |= {
            prefix + "input_layernorm.weight": torch.ones(WIDTH),
            prefix + "post_attention_layernorm.weight": torch.ones(WIDTH),
            prefix + "self_attn.q_proj.weight": matrix(WIDTH, WIDTH),
            prefix + "self_attn.k_proj.weight": matrix(key_width, WIDTH),
            prefix + "self_attn.v_proj.weight": matrix(key_width, WIDTH),
            prefix + "self_attn.o_proj.weight": matrix(WIDTH, WIDTH),
            prefix + "mlp.gate_proj.weight": matrix(HIDDEN, WIDTH),
            prefix + "mlp.up_proj.weight": matrix(HIDDEN, WIDTH),
            prefix + "mlp.down_proj.weight": matrix(WIDTH, HIDDEN),
        }
    model = lm.Lm(weights)
    model.requires_grad_(False)
    return model.eval()


def layers_of(model: lm.Lm) -> list[Tensor]:
    return [bank.stack(step) for step in model.feed_forwards()]


def tokens(seed: int = 0, length: int = 24) -> Tensor:
    return torch.randint(VOCAB, (2, length), generator=torch.Generator().manual_seed(seed))


def pairs(count: int) -> list[int]:
    return [layer * count // lm.LAYERS for layer in range(lm.LAYERS)]


def test_reordering_hidden_units_leaves_the_step_unchanged() -> None:
    step = tiny().feed_forwards()[0]
    order = torch.randperm(HIDDEN, generator=torch.Generator().manual_seed(1))
    gate, up, down = bank.unstack(bank.stack(step)[order])
    reordered = lm.FeedForward(lm.Dense(gate), lm.Dense(up), lm.Dense(down))
    x = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(step(x), reordered(x), atol=1e-6)


def test_match_undoes_a_reordering() -> None:
    reference = layers_of(tiny())[0]
    order = torch.randperm(HIDDEN, generator=torch.Generator().manual_seed(1))
    shuffled = reference[order]
    assert torch.equal(shuffled[bank.match(shuffled, reference)], reference)


def test_pairwise_sees_through_a_reordering_only_when_matched() -> None:
    reference = layers_of(tiny())[0]
    shuffled = reference[torch.randperm(HIDDEN, generator=torch.Generator().manual_seed(1))]
    assert bank.pairwise([reference, shuffled], matched=True)[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert bank.pairwise([reference, shuffled], matched=False)[0, 1] < 0.5


def test_scrambled_keeps_each_columns_values() -> None:
    layer = layers_of(tiny())[0]
    (mixed,) = bank.scrambled([layer], torch.Generator().manual_seed(0))
    assert not torch.equal(mixed, layer)
    assert torch.equal(mixed.sort(0).values, layer.sort(0).values)


def test_shared_energy_of_identical_and_of_unrelated_layers() -> None:
    layer = torch.randn(4, 6, generator=torch.Generator().manual_seed(0))
    assert bank.shared_energy([layer] * 3) == pytest.approx([1.0, 1.0, 1.0])
    apart = [torch.eye(3)[index : index + 1] for index in range(3)]
    assert bank.shared_energy(apart) == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_groups_puts_similar_layers_together() -> None:
    similarity = np.full((4, 4), 0.1)
    similarity[0, 2] = similarity[2, 0] = similarity[1, 3] = similarity[3, 1] = 0.9
    np.fill_diagonal(similarity, 1.0)
    assignment = bank.groups(similarity, 2)
    assert assignment[0] == assignment[2] != assignment[1] == assignment[3]
    assert bank.groups(similarity, 4) == [0, 1, 2, 3]


def shared_layers(rank: int) -> list[Tensor]:
    """Layers that are one matrix under different hidden-unit orders, plus a rank-`rank` term."""
    generator = torch.Generator().manual_seed(0)
    centre = torch.randn(HIDDEN, 3 * WIDTH, generator=generator)
    layers = []
    for _ in range(4):
        parts = [
            0.1 * torch.randn(HIDDEN, rank, generator=generator) @ torch.randn(rank, WIDTH, generator=generator)
            for _ in range(3)
        ]
        layers.append((centre + torch.cat(parts, dim=1))[torch.randperm(HIDDEN, generator=generator)])
    return layers


def test_fit_recovers_a_shared_matrix_only_with_matching() -> None:
    layers = shared_layers(rank=2)
    assert bank.fit(layers, [0, 0, 0, 0], 2, matched=True).error() < 0.02
    assert bank.fit(layers, [0, 0, 0, 0], 2, matched=False).error() > 0.3


def test_fit_error_falls_with_rank_and_is_zero_for_own_matrices() -> None:
    layers = layers_of(tiny())
    errors = [bank.fit(layers, pairs(15), rank, matched=True).error() for rank in (0, 4, 16)]
    assert errors[0] > errors[1] > errors[2] > 0
    assert bank.fit(layers, pairs(lm.LAYERS), 4, matched=True).error() == 0


def test_banked_model_computes_the_fit() -> None:
    model = tiny()
    fitted = bank.fit(layers_of(model), pairs(15), 4, matched=True)
    student = bank.banked(model, fitted)
    x = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(2))
    for layer in range(lm.LAYERS):
        block = student.block(layer)
        gate, up, down = bank.unstack(fitted.rebuilt(layer))
        dense = lm.FeedForward(lm.Dense(gate), lm.Dense(up), lm.Dense(down))
        assert torch.allclose(block.feed_forward(x), dense(x), atol=1e-5)
    assert len(model.feed_forwards()) == lm.LAYERS


def test_bank_with_a_matrix_per_layer_is_the_original() -> None:
    model = tiny()
    student = bank.banked(model, bank.fit(layers_of(model), pairs(lm.LAYERS), 4, matched=True))
    assert torch.allclose(student(tokens()), model(tokens()), atol=1e-5)
    assert bank.stored_bytes(student) == bank.stored_bytes(model)


def test_bank_stores_each_shared_matrix_once() -> None:
    model = tiny()
    rank = 4
    student = bank.banked(model, bank.fit(layers_of(model), pairs(15), rank, matched=True))
    step = 3 * HIDDEN * WIDTH
    correction = 3 * rank * (HIDDEN + WIDTH)
    saved = (lm.LAYERS - 15) * step - lm.LAYERS * correction
    assert bank.stored_bytes(model) - bank.stored_bytes(student) == 2 * saved


def test_layers_alone_on_a_bank_matrix_get_no_correction() -> None:
    model = tiny()
    assignment = [0, 0] + list(range(1, lm.LAYERS - 1))
    fitted = bank.fit(layers_of(model), assignment, 4, matched=True)
    assert fitted.corrections(0) is not None and fitted.corrections(5) is None
    student = bank.banked(model, fitted)
    x = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(student.block(5).feed_forward(x), model.block(5).feed_forward(x), atol=1e-6)


def test_low_rank_control_fits_the_bank_bytes() -> None:
    model = tiny()
    student = bank.banked(model, bank.fit(layers_of(model), pairs(15), 4, matched=True))
    target = bank.stored_bytes(student)
    rank = bank.equal_rank(target, model)
    assert bank.stored_bytes(bank.low_rank(model, rank)) <= target < bank.stored_bytes(bank.low_rank(model, rank + 1))


def test_kept_control_keeps_whole_layers_and_fits_the_bank_bytes() -> None:
    model = tiny()
    assignment = [0, 0, 0] + list(range(1, lm.LAYERS - 2))
    fitted = bank.fit(layers_of(model), assignment, 4, matched=True)
    kept = bank.kept_low_rank(model, fitted)
    banked = bank.stored_bytes(bank.banked(model, fitted))
    per_rank = 2 * 3 * 3 * (HIDDEN + WIDTH)
    assert banked - per_rank < bank.stored_bytes(kept) <= banked
    x = torch.randn(5, WIDTH, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(kept.block(5).feed_forward(x), model.block(5).feed_forward(x), atol=1e-6)
    low = kept.block(0).feed_forward
    assert isinstance(low, lm.FeedForward) and isinstance(low.gate, bank.LowRank)
    assert 0 < bank.weight_error(model, kept) < 1


def test_truncate_is_the_best_low_rank_matrix() -> None:
    matrix = torch.randn(12, 8, generator=torch.Generator().manual_seed(0))
    spectrum = torch.linalg.svdvals(matrix)
    lost = (matrix - bank.truncate(matrix, 3).dense()).pow(2).sum()
    assert float(lost.detach()) == pytest.approx(float(spectrum[3:].pow(2).sum()), rel=1e-4)


def test_routed_model_starts_as_the_fixed_bank() -> None:
    model = tiny()
    fitted = bank.fit(layers_of(model), pairs(5), 4, matched=True)
    routed = bank.routed(model, fitted).eval()
    assert torch.allclose(routed(tokens()), bank.banked(model, fitted)(tokens()), atol=1e-4)
    step = routed.block(7).feed_forward
    assert isinstance(step, bank.RoutedFeedForward) and step.picked is not None
    assert bool((step.picked == fitted.assignment[7]).all())


def test_routed_pick_changes_the_output() -> None:
    model = tiny()
    fitted = bank.fit(layers_of(model), pairs(5), 4, matched=True)
    routed = bank.routed(model, fitted).eval()
    before = routed(tokens())
    step = routed.block(7).feed_forward
    assert isinstance(step, bank.RoutedFeedForward)
    with torch.no_grad():
        step.router.prior[7] = step.router.prior[7].roll(1)
    assert not torch.allclose(routed(tokens()), before, atol=1e-3)


def test_routed_training_sends_gradient_to_router_and_every_bank_matrix() -> None:
    model = tiny()
    routed = bank.routed(model, bank.fit(layers_of(model), pairs(5), 4, matched=True)).train()
    routed(tokens()).sum().backward()
    step = routed.block(7).feed_forward
    assert isinstance(step, bank.RoutedFeedForward)
    assert step.router.prior.grad is not None and bool(step.router.prior.grad[7].abs().sum() > 0)
    for gate in step.gates:
        assert isinstance(gate, lm.Dense) and gate.weight.grad is not None and bool(gate.weight.grad.abs().sum() > 0)


@pytest.mark.parametrize("bits", [4, 8])
def test_rounded_stays_within_half_a_step(bits: int) -> None:
    weight = torch.randn(6, bank.GROUP + 5, generator=torch.Generator().manual_seed(0))
    out = bank.rounded(weight, bits)
    assert out.shape == weight.shape
    levels = len(set(out.flatten()[: bank.GROUP].tolist()))
    assert 2 < levels <= 2**bits
    spread = float(weight.max() - weight.min())
    assert float((out - weight).abs().max()) <= spread / (2**bits - 1) / 2 + 1e-3


def test_quantized_leaves_the_model_alone_and_counts_bytes() -> None:
    model = tiny()
    before = model.embedding.clone()
    squeezed = bank.quantized(model, 4)
    assert torch.equal(model.embedding, before)
    assert isinstance(squeezed, lm.Lm) and not torch.equal(squeezed.embedding, before)
    matrices = [parameter for parameter in model.parameters() if parameter.dim() == 2]
    vectors = sum(parameter.numel() for parameter in model.parameters() if parameter.dim() == 1)
    grouped = sum(-(-matrix.numel() // bank.GROUP) for matrix in matrices)
    assert bank.stored_bytes(model, 4) == grouped * (bank.GROUP * 4 + 32) // 8 + 2 * vectors
    assert bank.stored_bytes(model) == 2 * sum(parameter.numel() for parameter in model.parameters())


def test_perplexity_of_a_uniform_model_is_the_vocabulary() -> None:
    model = tiny()
    with torch.no_grad():
        model.embedding.zero_()
    text = torch.randint(VOCAB, (2 * lm.CONTEXT + 1,), generator=torch.Generator().manual_seed(0))
    assert lm.perplexity(model, text) == pytest.approx(VOCAB, rel=1e-4)


def test_windows_predict_the_next_token() -> None:
    inputs, targets = windows(torch.arange(11), 5)
    assert inputs.tolist() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert targets.tolist() == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


def test_heal_lowers_the_loss_and_trains_only_the_new_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heal, "CONTEXT", 16)
    model = tiny()
    student = bank.banked(model, bank.fit(layers_of(model), pairs(8), 2, matched=True))
    frozen = model.block(0).query.weight.clone()
    text = torch.randint(VOCAB, (20_000,), generator=torch.Generator().manual_seed(0))
    curve = heal.heal(student, model, text, steps=100, rate=3e-3, seed=0)
    assert len(curve) == 2 and curve[1] < curve[0]
    assert torch.equal(model.block(0).query.weight, frozen)
    assert student.block(0).query.weight is model.block(0).query.weight


def test_heal_in_single_rows_matches_whole_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(heal, "CONTEXT", 16)
    model = tiny()
    text = torch.randint(VOCAB, (20_000,), generator=torch.Generator().manual_seed(0))
    curves = []
    for micro in (4, 1):
        student = bank.banked(model, bank.fit(layers_of(model), pairs(8), 2, matched=True))
        curves.append(heal.heal(student, model, text, steps=50, rate=1e-3, seed=0, micro=micro))
    assert curves[0] == pytest.approx(curves[1], rel=1e-3)


def record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, losses: dict[float, float]) -> None:
    monkeypatch.setattr(run, "RUNS", tmp_path)
    chunks, rank = run.SWEEP_BANK
    for rate, loss in losses.items():
        held = {"kind": "bank", "chunks": chunks, "rank": rank, "seed": 0, "steps": run.SWEEP_STEPS, "rate": rate}
        run.write(Arm("bank", chunks, rank, Trial(rate)).path, held | {"validation_perplexity": loss})


def test_sweep_starts_with_the_first_rates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record(monkeypatch, tmp_path, {})
    assert [arm.rate for arm in run.sweep("bank")] == list(run.FIRST_RATES)


def test_sweep_stops_when_the_best_rate_is_inside(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record(monkeypatch, tmp_path, dict(zip(run.FIRST_RATES, (9.0, 5.0, 7.0, 8.0))))
    assert run.sweep("bank") == []
    assert run.chosen_rate("bank") == run.FIRST_RATES[1]
    assert run.chosen_rate("routed") == run.FIRST_RATES[1]


def test_sweep_extends_past_a_best_rate_at_the_edge(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record(monkeypatch, tmp_path, dict(zip(run.FIRST_RATES, (9.0, 8.0, 7.0, 6.0))))
    assert [arm.rate for arm in run.sweep("bank")] == [run.LADDER[5]]
    record(monkeypatch, tmp_path, {run.LADDER[5]: 6.5})
    assert run.sweep("bank") == []
    record(monkeypatch, tmp_path, dict(zip(run.FIRST_RATES, (5.0, 8.0, 9.0, 9.5))))
    assert [arm.rate for arm in run.sweep("bank")] == [run.LADDER[0]]


def test_plan_pairs_every_bank_with_both_controls() -> None:
    arms = run.plan()
    assert len({arm.path for arm in arms}) == len(arms)
    for chunks, rank in run.BANKS:
        assert {arm.kind for arm in arms if (arm.chunks, arm.rank, arm.seed) == (chunks, rank, 0)} >= set(run.FIXED)
    assert sum(arm.kind == "routed" for arm in arms) == 1


def test_a_diverged_rate_is_never_chosen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record(monkeypatch, tmp_path, dict(zip(run.FIRST_RATES, (float("nan"), 5.0, 7.0, float("inf")))))
    assert run.chosen_rate("bank") == run.FIRST_RATES[1]


def test_a_record_of_another_plan_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    record(monkeypatch, tmp_path, {run.FIRST_RATES[0]: 5.0})
    arm = Arm("bank", *run.SWEEP_BANK, Trial(run.FIRST_RATES[0]))
    assert arm.read()["validation_perplexity"] == 5.0
    run.write(arm.path, arm.read() | {"steps": run.SWEEP_STEPS + 1})
    with pytest.raises(ValueError):
        arm.read()


def test_paths_tell_trials_and_seeds_apart() -> None:
    paths = {Arm("bank", 8, 16, budget).path for budget in (Full(), Full(1), Trial(1e-3), Trial(3e-3))}
    assert len(paths) == 4


def test_fit_centre_is_the_mean_of_its_members() -> None:
    generator = torch.Generator().manual_seed(0)
    first, second = (torch.randn(HIDDEN, 3 * WIDTH, generator=generator) for _ in range(2))
    fitted = bank.fit([first, second], [0, 0], 0, matched=False)
    assert torch.allclose(fitted.centres[0], (first + second) / 2, atol=1e-6)


def test_routed_training_pass_is_the_fixed_bank_when_the_prior_decides() -> None:
    model = tiny()
    fitted = bank.fit(layers_of(model), pairs(5), 4, matched=True)
    routed = bank.routed(model, fitted).train()
    step = routed.block(0).feed_forward
    assert isinstance(step, bank.RoutedFeedForward)
    with torch.no_grad():
        step.router.prior.mul_(1e4)
    assert torch.allclose(routed(tokens()), bank.banked(model, fitted)(tokens()), atol=1e-4)


def test_schedule_warms_up_then_decays_to_zero() -> None:
    assert heal.schedule(0, 100) == pytest.approx(0.2)
    assert heal.schedule(4, 100) == pytest.approx(1.0)
    assert heal.schedule(52, 100) == pytest.approx(0.5, abs=0.02)
    assert heal.schedule(99, 100) < 0.01


def test_rotate_turns_each_pair_by_its_position() -> None:
    x = torch.tensor([1.0, 0.0]).expand(3, 2)
    angle = torch.arange(3.0)
    assert torch.allclose(lm.rotate(x), torch.stack([angle.cos(), angle.sin()], dim=1), atol=1e-6)


def test_query_heads_read_their_own_key_value_head() -> None:
    block = tiny().block(0)
    head = WIDTH // lm.HEADS
    group = lm.HEADS // lm.KEY_VALUE_HEADS * head
    with torch.no_grad():
        block.value.weight[head:].zero_()
        block.out.weight.copy_(torch.eye(WIDTH))
    out = block.attend(torch.randn(1, 6, WIDTH, generator=torch.Generator().manual_seed(0)))
    assert bool(out[..., :group].abs().sum() > 0) and bool((out[..., group:] == 0).all())


def test_a_later_token_changes_no_earlier_prediction() -> None:
    model = tiny()
    first = tokens()
    second = first.clone()
    second[:, -1] = (second[:, -1] + 1) % VOCAB
    assert torch.allclose(model(first)[:, :-1], model(second)[:, :-1], atol=1e-6)
    assert not torch.allclose(model(first)[:, -1], model(second)[:, -1], atol=1e-6)


def test_normalise_scales_to_unit_size_times_its_weight() -> None:
    x = torch.tensor([[3.0, 4.0]])
    out = lm.normalise(x, torch.tensor([2.0, 0.5]))
    assert torch.allclose(out, torch.tensor([[2 * 3.0, 0.5 * 4.0]]) / 12.5**0.5, atol=1e-4)
