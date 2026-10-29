"""Kiểm tra luồng: ảnh -> lưới -> 81 ô -> solver (không cần Streamlit/TensorFlow)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from sudoku_ai.solver import solve  # noqa: E402

PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


def test_solver_solves_puzzle():
    board = np.array([int(c) for c in PUZZLE]).reshape(9, 9)
    solved = solve(board)
    assert solved is not None and (solved > 0).all()
