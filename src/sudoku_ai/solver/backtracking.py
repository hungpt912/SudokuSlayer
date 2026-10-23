"""Giải Sudoku bằng quay lui (backtracking)."""

from typing import Optional

import numpy as np

from sudoku_ai.solver.validator import is_valid_board, is_valid_placement


def _solve_in_place(grid) -> bool:
    """Điền trực tiếp vào grid. Trả về True nếu giải được."""
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid_placement(grid, row, col, num):
                        grid[row][col] = num
                        if _solve_in_place(grid):
                            return True
                        grid[row][col] = 0   # quay lui
                return False
    return True


def solve(board) -> Optional[np.ndarray]:
    """Nhận bàn cờ (9, 9), 0 là ô trống. Trả về bàn cờ đã giải (bản sao),
    hoặc None nếu bàn cờ sai (trùng số) hoặc vô nghiệm. Không sửa bàn cờ đầu vào."""
    grid = np.array(board, dtype=int)  # np.array tạo bản sao
    if grid.shape != (9, 9) or not is_valid_board(grid):
        return None
    return grid if _solve_in_place(grid) else None