"""Định dạng bảng lời giải: tô màu khối 3x3 và các ô do chương trình điền."""

import numpy as np
import pandas as pd

COLUMNS = [str(i) for i in range(1, 10)]

# Streamlit chỉ hỗ trợ màu chữ, màu nền và độ đậm của chữ
BLOCK_STYLE = "background-color: #f0f2f6"                                        # Khối 3x3 xen kẽ
NEW_CELL_STYLE = "background-color: #c8f0d2; color: #0a5c24; font-weight: bold"  # Số chương trình điền

# Mặt nạ khối 3x3 xen kẽ, tính một lần
_ROWS, _COLS = np.indices((9, 9))
_ALT_BLOCK = (_ROWS // 3 + _COLS // 3) % 2 == 1


def to_frame(board: np.ndarray) -> pd.DataFrame:
    """Bảng (9, 9) -> DataFrame với cột đánh số 1..9."""
    return pd.DataFrame(np.array(board, dtype=int), columns=COLUMNS)


def _style_matrix(original: np.ndarray) -> np.ndarray:
    """Ma trận CSS (9, 9): ô do chương trình điền tô xanh, ô còn lại theo khối xen kẽ."""
    styles = np.where(_ALT_BLOCK, BLOCK_STYLE, "")
    return np.where(original == 0, NEW_CELL_STYLE, styles).astype(object)


def highlight_new_cells(solved: np.ndarray, original: np.ndarray):
    """Trả về Styler của bảng lời giải.

    solved   -- bảng đã giải (9, 9)
    original -- bảng trước khi giải (9, 9), 0 là ô trống
    """
    original = np.array(original, dtype=int)
    styles = pd.DataFrame(_style_matrix(original), columns=COLUMNS)
    return to_frame(solved).style.apply(lambda _: styles, axis=None)
