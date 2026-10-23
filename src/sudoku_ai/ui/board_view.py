"""Hiển thị bàn cờ Sudoku dạng bảng."""

import numpy as np
import pandas as pd
 
COLUMNS = [str(i) for i in range(1, 10)]
 
# Style CSS: Streamlit chỉ hỗ trợ màu chữ, màu nền và độ đậm của chữ
BLOCK_STYLE = "background-color: #f0f2f6"                                       # khối 3x3 xen kẽ
NEW_CELL_STYLE = "background-color: #c8f0d2; color: #0a5c24; font-weight: bold"  # số chương trình điền
 
 
def to_frame(board: np.ndarray) -> pd.DataFrame:
    """Bàn cờ (9, 9) -> DataFrame 9x9 với cột đánh số 1..9."""
    return pd.DataFrame(np.array(board, dtype=int), columns=COLUMNS)
 
 
def _style_matrix(original: np.ndarray) -> np.ndarray:
    """Ma trận (9, 9) chứa chuỗi CSS cho từng ô. Ô nào do chương trình điền thì tô xanh."""
    styles = np.full((9, 9), "", dtype=object)
    for row in range(9):
        for col in range(9):
            if original[row, col] == 0:
                styles[row, col] = NEW_CELL_STYLE
            elif (row // 3 + col // 3) % 2 == 1:
                styles[row, col] = BLOCK_STYLE
    return styles
 
 
def highlight_new_cells(solved: np.ndarray, original: np.ndarray):
    """Bảng lời giải: ô do chương trình điền tô xanh, ô đề cho giữ nền khối 3x3 xen kẽ.
 
    solved   -- bàn cờ đã giải (9, 9)
    original -- bàn cờ trước khi giải (9, 9), 0 là ô trống
    """
    original = np.array(original, dtype=int)
    styles = pd.DataFrame(_style_matrix(original), columns=COLUMNS)
    return to_frame(solved).style.apply(lambda _: styles, axis=None)
 