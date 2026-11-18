"""Giao diện Streamlit: tải ảnh, sửa kết quả nhận diện và xem lời giải.

Chạy: streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sudoku_ai.pipeline_image.cells import split_and_normalize  # noqa: E402
from sudoku_ai.pipeline_image.grid import detect_and_warp  # noqa: E402
from sudoku_ai.pipeline_image.preprocess import preprocess  # noqa: E402
from sudoku_ai.recognition import get_engine  # noqa: E402
from sudoku_ai.solver import solve  # noqa: E402
from sudoku_ai.ui.board_view import COLUMNS  # noqa: E402

# Chỉ cho nhập một chữ số 1..9 khi sửa kết quả nhận diện; xóa số để làm ô trống
CELL_CONFIG = {
    col: st.column_config.TextColumn(width=34, alignment="center", validate=r"^[1-9]?$", max_chars=1)
    for col in COLUMNS
}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.step {display: flex; align-items: center; gap: .5rem; font-weight: 600; margin-bottom: .75rem;}
.step b {display: inline-flex; align-items: center; justify-content: center; width: 1.6rem; height: 1.6rem;
         border-radius: 50%; background: #1f2430; color: #fff; font-size: .85rem;}
.sdk {display: grid; grid-template-columns: repeat(9, 1fr); aspect-ratio: 1; width: 100%; max-width: 420px;
      margin: 0 auto; border: 2px solid #1f2430; background: #fff; font-family: ui-monospace, monospace;}
.sdk div {display: flex; align-items: center; justify-content: center; color: #1f2430;
          font-size: clamp(.9rem, 2vw, 1.35rem); border-right: 1px solid #c9ccd3; border-bottom: 1px solid #c9ccd3;}
.sdk div.r {border-right: 2px solid #1f2430;}
.sdk div.b {border-bottom: 2px solid #1f2430;}
.sdk div.n {color: #0a5c24; font-weight: 700; background: #c8f0d2;}
.sdk div.x {color: #b42318; font-weight: 700; background: #fde8e7;}
</style>
"""


@st.cache_resource(show_spinner="Đang tải mô hình CNN...")
def load_engine():
    """Nạp mô hình một lần, không nạp lại mỗi khi Streamlit chạy lại script."""
    return get_engine()


def decode_image(data: bytes) -> np.ndarray:
    """Bytes ảnh tải lên -> ảnh BGR."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được ảnh. Hãy thử file PNG/JPG khác.")
    return image


@st.cache_data(show_spinner="Đang nhận diện...")
def read_sudoku(data: bytes):
    """Ảnh -> (bảng đã nắn, ma trận 9x9). Trả về (None, None) nếu không thấy lưới."""
    image = decode_image(data)
    warped = None
    for connect_lines in (False, True):  # Lần 2 nối các đường kẻ bị đứt
        try:
            warped = detect_and_warp(preprocess(image, connect_lines))["board_gray"]
            break
        except ValueError:
            continue
    if warped is None:
        return None, None
    cells = split_and_normalize(warped)  # (81, 28, 28, 1), float trong [0, 1]
    board = np.array(load_engine().predict_board(cells), dtype=int)
    return warped, board


@st.cache_data(show_spinner="Đang giải...")
def solve_cached(board: np.ndarray):
    """Streamlit chạy lại cả script mỗi lần tương tác; bảng không đổi thì không giải lại."""
    return solve(board)


def conflicts(board: np.ndarray) -> np.ndarray:
    """Mặt nạ (9, 9): ô có số trùng với ô khác cùng hàng, cột hoặc khối 3x3."""
    mask = np.zeros((9, 9), dtype=bool)
    for row, col in zip(*np.nonzero(board)):
        top, left = row // 3 * 3, col // 3 * 3
        peers = np.concatenate([board[row], board[:, col], board[top:top + 3, left:left + 3].ravel()])
        mask[row, col] = np.count_nonzero(peers == board[row, col]) > 3   # Chính ô đó được đếm 3 lần
    return mask


def board_html(board: np.ndarray, new: np.ndarray, wrong: np.ndarray) -> str:
    """Bảng 9x9 dạng lưới HTML: ô trong `new` tô xanh, ô trong `wrong` tô đỏ, số 0 để trống."""
    cells = []
    for row in range(9):
        for col in range(9):
            classes = []
            if col in (2, 5):
                classes.append("r")
            if row in (2, 5):
                classes.append("b")
            if wrong[row, col]:
                classes.append("x")
            elif new[row, col]:
                classes.append("n")
            cells.append(f'<div class="{" ".join(classes)}">{board[row, col] or ""}</div>')
    return f'<div class="sdk">{"".join(cells)}</div>'


def step(number: int, title: str) -> None:
    """Tiêu đề một bước: số thứ tự trong vòng tròn và tên bước."""
    st.markdown(f'<div class="step"><b>{number}</b>{title}</div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Sudoku Vision Solver", page_icon="🧩", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Nhận diện và giải Sudoku từ ảnh")

    file = st.file_uploader("Tải ảnh Sudoku", type=["png", "jpg", "jpeg"])
    if file is None:
        st.info("Hãy tải lên một ảnh Sudoku để bắt đầu.")
        return

    try:
        warped, board = read_sudoku(file.getvalue())
    except (ValueError, RuntimeError) as err:
        st.error(str(err))
        return
    if board is None:
        st.error("Không tìm thấy lưới Sudoku. Hãy thử ảnh rõ nét, chụp thẳng hơn.")
        return

    col1, col2, col3 = st.columns(3)
    with col1.container(border=True):
        step(1, "Lưới tìm được")
        st.image(warped, width="stretch")

    with col2.container(border=True):
        step(2, "Nhận diện")
        st.caption("Bấm vào ô để sửa nếu đọc sai.")
        edited = st.data_editor(
            pd.DataFrame(np.where(board > 0, board.astype(str), ""), columns=COLUMNS),
            hide_index=True,
            column_config=CELL_CONFIG,
            key=file.file_id,
            width="stretch",
        )
        numbers = edited.apply(pd.to_numeric, errors="coerce").fillna(0)  # Ô trống -> 0
        board = np.clip(numbers.to_numpy(dtype=int), 0, 9)

    with col3.container(border=True):
        step(3, "Lời giải")
        solved = solve_cached(board)
        if solved is None:
            wrong = conflicts(board)
            st.markdown(board_html(board, np.zeros_like(wrong), wrong), unsafe_allow_html=True)
            st.warning("Bảng mâu thuẫn hoặc vô nghiệm, hãy sửa lại các số nhận diện sai.")
        else:
            st.markdown(board_html(solved, board == 0, np.zeros((9, 9), dtype=bool)), unsafe_allow_html=True)
            st.caption("Ô xanh là số do chương trình điền.")


if __name__ == "__main__":
    main()
