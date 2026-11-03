"""Giải Sudoku bằng quay lui (backtracking) ket hop bitmask"""

from typing import Optional
import numpy as np 
from sudoku_ai.solver.validator import is_valid_board
 
ALL_DIGITS = 0b1111111110  # bit 1..9 được bật
 
 
def _box_index(row: int, col: int) -> int:
    return 3 * (row // 3) + col // 3
 
 
def _solve_in_place(grid: list, rows: list, cols: list, boxes: list) -> bool:
    """Điền trực tiếp vào grid (list 9x9). Trả về True nếu giải được.
 
    rows/cols/boxes: bitmask các số đã dùng trong từng hàng/cột/khối.
    Mỗi bước chọn ô trống có ít ứng viên nhất (MRV) để cắt nhánh sớm.
    """
    best = None          # (row, col, mask ứng viên)
    best_count = 10
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                continue
            candidates = ALL_DIGITS & ~(rows[r] | cols[c] | boxes[_box_index(r, c)])
            count = bin(candidates).count("1")
            if count == 0:
                return False          # ô không còn số nào hợp lệ -> nhánh chết
            if count < best_count:
                best, best_count = (r, c, candidates), count
                if count == 1:
                    break
        if best_count == 1:
            break
 
    if best is None:
        return True                   # không còn ô trống
 
    r, c, candidates = best
    b = _box_index(r, c)
    for num in range(1, 10):
        bit = 1 << num
        if candidates & bit:
            grid[r][c] = num
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            if _solve_in_place(grid, rows, cols, boxes):
                return True
            grid[r][c] = 0            # quay lui
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[b] &= ~bit
    return False
 
 
def solve(board) -> Optional[np.ndarray]:
    """Nhận bàn cờ (9, 9), 0 là ô trống. Trả về bàn cờ đã giải (bản sao),
    hoặc None nếu bàn cờ sai (trùng số, số ngoài 0..9) hoặc vô nghiệm.
    Không sửa bàn cờ đầu vào."""
    if not is_valid_board(board):
        return None
 
    grid = np.array(board, dtype=int).tolist()  # list thuần Python nhanh hơn numpy khi truy cập từng ô
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                bit = 1 << grid[r][c]
                rows[r] |= bit
                cols[c] |= bit
                boxes[_box_index(r, c)] |= bit
 
    if not _solve_in_place(grid, rows, cols, boxes):
        return None
    return np.array(grid, dtype=int)
 