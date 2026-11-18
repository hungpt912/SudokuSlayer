"""Kiểm thử bộ kiểm tra hợp lệ và bộ giải.

Chạy: python -m pytest tests/test_backtracking.py
"""

import numpy as np
import pytest

from _support import correct_solution
from sudoku_ai.solver import is_valid_board, solve

PUZZLE = np.array(list("530070000600195000098000060800060003400803001700020006060000280000419005000080079"),
                  dtype=int).reshape(9, 9)


def test_solver_completes_board_and_preserves_input():
    before = PUZZLE.copy()
    assert correct_solution(solve(PUZZLE), PUZZLE)
    np.testing.assert_array_equal(PUZZLE, before)


def test_solver_accepts_a_completed_board():
    board = solve(PUZZLE)
    np.testing.assert_array_equal(solve(board), board)


@pytest.mark.parametrize("change", ["row", "column", "box", "range", "shape"])
def test_invalid_boards_are_rejected(change):
    board = PUZZLE.copy()
    if change == "row":
        board[0, 2] = 7      # Trùng số 7 trong hàng 1
    elif change == "column":
        board[2, 0] = 4      # Trùng số 4 trong cột 1
    elif change == "box":
        board[2, 0] = 3      # Trùng số 3 trong khối trên trái
    elif change == "range":
        board[0, 0] = 10     # Giá trị ngoài 0..9
    else:
        board = board[:8]    # Sai kích thước
    assert not is_valid_board(board)
    assert solve(board) is None


def test_locally_valid_but_unsatisfiable_board_is_rejected():
    # Không trùng số nhưng vô nghiệm (đề trong ảnh Lexski/hnp3xzbl6k0d1.webp)
    board = np.array(list("479000685530000209182090000001206470706049001305000000000910040900407068200683107"),
                     dtype=int).reshape(9, 9)
    assert is_valid_board(board)
    assert solve(board) is None
