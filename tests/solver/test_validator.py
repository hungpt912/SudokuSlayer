import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sudoku_ai.solver import is_valid_board, is_valid_placement  # noqa: E402

PUZZLE = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])


# ---------- test is_valid_placement ----------

def test_placement_ok():
    assert is_valid_placement(PUZZLE, 0, 2, 4)


def test_placement_conflict_in_row():
    assert not is_valid_placement(PUZZLE, 0, 2, 7)   # Trùng 7 ở hàng 0 (cột 4)


def test_placement_conflict_in_column():
    assert not is_valid_placement(PUZZLE, 2, 0, 4)   # Trùng 4 ở cột 0 (hàng 4)


def test_placement_conflict_in_box():
    assert not is_valid_placement(PUZZLE, 2, 0, 3)   # Trùng 3 ở khối 3x3 (ô (0, 1))


# ---------- test is_valid_board ----------

def test_valid_board_true_for_puzzle():
    assert is_valid_board(PUZZLE.copy())


def test_valid_board_true_for_empty_board():
    assert is_valid_board(np.zeros((9, 9), dtype=int))


def test_valid_board_detects_row_duplicate():
    bad = PUZZLE.copy()
    bad[0, 2] = 7             # Trùng hàng 0
    assert not is_valid_board(bad)


def test_valid_board_detects_column_duplicate():
    bad = PUZZLE.copy()
    bad[2, 0] = 4             # Trùng cột 0
    assert not is_valid_board(bad)


def test_valid_board_detects_box_duplicate():
    bad = PUZZLE.copy()
    bad[2, 0] = 3             # Trùng khối 3x3
    assert not is_valid_board(bad)


def test_valid_board_does_not_change_board():
    board = PUZZLE.copy()
    is_valid_board(board)
    assert (board == PUZZLE).all()