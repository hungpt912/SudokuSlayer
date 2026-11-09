import numpy as np

from sudoku_ai.recognition.cnn_model import build_cnn, decode_cells


def test_model_predicts_presence_and_nine_digits():
    model = build_cnn()
    occupied, digits = model(np.zeros((2, 28, 28, 1), dtype="float32"))
    assert occupied.shape == (2, 1)
    assert digits.shape == (2, 9)
    np.testing.assert_allclose(np.sum(digits, axis=1), 1, atol=1e-6)


def test_only_presence_can_mark_a_cell_empty():
    digit_scores = np.eye(9)
    np.testing.assert_array_equal(decode_cells(np.ones(9), digit_scores), np.arange(1, 10))
    np.testing.assert_array_equal(decode_cells(np.zeros(9), digit_scores), np.zeros(9))
