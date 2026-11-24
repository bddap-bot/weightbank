from __future__ import annotations

from weightbank import memory
from weightbank.configs import config
from weightbank.model import Transformer

FLOAT_BYTES = 4


def test_every_measured_cell_is_one_the_model_builds_and_its_weights_take_four_bytes_each() -> None:
    measured = memory.read()
    assert measured
    for name, found in measured.items():
        stored = Transformer(config(name)).stored_weights()
        assert FLOAT_BYTES * stored <= found.weights_bytes <= 1.01 * FLOAT_BYTES * stored


def test_a_peak_holds_the_weights_and_training_holds_more_than_one_window_of_inference() -> None:
    for found in memory.read().values():
        assert found.weights_bytes < found.inference_window_bytes < found.inference_batch_bytes
        assert found.inference_window_bytes < found.training_bytes
