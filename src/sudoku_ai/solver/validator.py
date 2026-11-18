"""Kiểm tra bảng Sudoku có hợp lệ trước khi giải."""

import numpy as np


def _has_duplicates(units: np.ndarray) -> bool:
    """units có shape (9, 9), mỗi hàng là một nhóm (hàng, cột hoặc khối). Bỏ qua ô trống."""
    for unit in units:
        values = unit[unit != 0]
        if values.size != np.unique(values).size:
            return True
    return False


def is_valid_board(grid) -> bool:
    """Bảng hợp lệ khi có shape (9, 9), giá trị trong 0..9 và không trùng số. Không sửa grid."""
    try:
        arr = np.asarray(grid, dtype=int)
    except (TypeError, ValueError):
        return False
    if arr.shape != (9, 9) or arr.min() < 0 or arr.max() > 9:
        return False

    # Duỗi mỗi khối 3x3 thành một hàng để kiểm tra giống hàng và cột
    boxes = arr.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
    return not (_has_duplicates(arr) or _has_duplicates(arr.T) or _has_duplicates(boxes))
