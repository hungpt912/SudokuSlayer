"""Kiểm tra tính hợp lệ của bàn cờ Sudoku."""

import numpy as np
  
def is_valid_placement(grid, row, col, num):
    """Đặt `num` vào grid[row][col] có hợp lệ không (không trùng hàng, cột, khối 3x3)."""
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
 
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num:
                return False
 
    return True
 
def _has_duplicates(units: np.ndarray) -> bool:
    """`units` có shape (9, 9), mỗi hàng là một nhóm 9 ô (hàng, cột hoặc khối).
    True nếu có nhóm nào trùng số (bỏ qua ô trống = 0)."""
    for unit in units:
        values = unit[unit != 0]
        if values.size != np.unique(values).size:
            return True
    return False
 
def is_valid_board(grid) -> bool:
    """Bàn cờ (9, 9) có hợp lệ không: số trong 0..9 và không mâu thuẫn nhau
    (dùng để phát hiện nhận diện sai). Không sửa `grid`."""
    try:
        arr = np.asarray(grid, dtype=int)
    except (TypeError, ValueError):
        return False
    if arr.shape != (9, 9) or arr.min() < 0 or arr.max() > 9:
        return False
 
    # Mỗi khối 3x3 được duỗi thành một hàng để kiểm tra giống hàng/cột
    boxes = arr.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
    return not (_has_duplicates(arr) or _has_duplicates(arr.T) or _has_duplicates(boxes))
 