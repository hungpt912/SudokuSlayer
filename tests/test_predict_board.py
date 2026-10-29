
import numpy as np
import pytest

from sudoku_ai.recognition.cnn_engine import CNNEngine


def test_predict_board_returns_9x9_matrix():
    engine = CNNEngine()

    # Create a batch of 81 empty cells.
    cells = np.zeros((81, 28, 28, 1), dtype=np.float32)

    board = engine.predict_board(cells)

    assert len(board) == 9
    assert all(len(row) == 9 for row in board)


def test_predict_board_empty_cells_are_zero():
    engine = CNNEngine()

    cells = np.zeros((81, 28, 28, 1), dtype=np.float32)

    board = engine.predict_board(cells)

    assert all(
        digit == 0
        for row in board
        for digit in row
    )


def test_predict_board_rejects_invalid_input():
    engine = CNNEngine()

    invalid_cells = np.zeros((80, 28, 28, 1), dtype=np.float32)

    with pytest.raises(ValueError):
        engine.predict_board(invalid_cells)
