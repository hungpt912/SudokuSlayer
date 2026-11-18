"""Kiểm thử mô hình CNN, hàm giải mã và CNNEngine.

Chạy: python -m pytest tests/test_cnn.py
"""

import numpy as np
import pytest

from sudoku_ai.recognition.cnn_engine import CNNEngine
from sudoku_ai.recognition.cnn_model import build_cnn, decode_cells


@pytest.fixture(scope="module")
def engine():
    return CNNEngine()


def test_model_has_presence_and_nine_digit_outputs():
    occupied, digits = build_cnn()(np.zeros((2, 28, 28, 1), dtype=np.float32), training=False)
    assert occupied.shape == (2, 1)
    assert digits.shape == (2, 9)
    np.testing.assert_allclose(np.sum(digits, axis=1), 1, atol=1e-6)


def test_decoder_uses_presence_and_digits_one_to_nine():
    scores = np.eye(9)
    np.testing.assert_array_equal(decode_cells(np.ones(9), scores), np.arange(1, 10))
    np.testing.assert_array_equal(decode_cells(np.zeros(9), scores), np.zeros(9))


def test_trained_model_loads(engine):
    assert engine.model.output_names == ["occupied", "digit"]


@pytest.mark.parametrize("shape", [(28, 28), (28, 28, 3)])
def test_empty_grayscale_and_rgb_cells_return_zero(engine, shape):
    predicted = engine.predict(np.zeros(shape, dtype=np.uint8))
    assert isinstance(predicted, int)
    assert predicted == 0


def test_predict_board_empty_cells_are_9x9_zeros(engine):
    board = np.asarray(engine.predict_board(np.zeros((81, 28, 28, 1), dtype=np.float32)))
    assert board.shape == (9, 9)
    np.testing.assert_array_equal(board, np.zeros((9, 9)))


def test_predict_board_rejects_incomplete_batch(engine):
    with pytest.raises(ValueError, match="Expected 81 cells"):
        engine.predict_board(np.zeros((80, 28, 28, 1), dtype=np.float32))
