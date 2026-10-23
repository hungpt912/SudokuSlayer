"""Giao diện người dùng cho Sudoku AI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# from sudoku_ai.recognition import recognize_board  # noqa: E402
from sudoku_ai.solver import solve  # noqa: E402
# from sudoku_ai.pipeline_image import decode_image, extract_cells, find_grid, warp_grid  # noqa: E402
from sudoku_ai.ui.board_view import COLUMNS, highlight_new_cells  # noqa: E402

# Chỉ cho nhập số nguyên 0..9 khi sửa kết quả nhận diện
CELL_CONFIG = {
    col: st.column_config.NumberColumn(min_value=0, max_value=9, step=1, default=0, width="small")
    for col in COLUMNS
}


# @st.cache_data(show_spinner="Đang nhận diện...")
# def read_sudoku(data: bytes):
#     """Ảnh -> (ảnh lưới đã làm phẳng, bàn cờ 9x9). Trả về (None, None) nếu không thấy lưới."""
#     image = decode_image(data)
#     corners = find_grid(image)
#     if corners is None:
#         return None, None
#     warped = warp_grid(image, corners)
#     return warped, recognize_board(extract_cells(warped))


@st.cache_data(show_spinner="Đang giải...")
def solve_cached(board: np.ndarray):
    """Giải có nhớ kết quả: Streamlit chạy lại cả script mỗi lần tương tác,
    nên bàn cờ không đổi thì không cần giải lại."""
    return solve(board)


def main() -> None:
    st.set_page_config(page_title="Sudoku Vision Solver", layout="wide")
    st.title("Nhận diện và giải Sudoku từ ảnh")

    file = st.file_uploader("Tải ảnh Sudoku", type=["png", "jpg", "jpeg"])
    if file is None:
        st.info("Hãy tải lên một ảnh Sudoku để bắt đầu.")
        return

    # try:
    #     warped, board = read_sudoku(file.getvalue())
    # except (ValueError, RuntimeError) as err:
    #     st.error(str(err))
    #     return
    # if board is None:
    #     st.error("Không tìm thấy lưới Sudoku. Hãy thử ảnh rõ nét, chụp thẳng hơn.")
    #     return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Lưới tìm được")
        # st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    with col2:
        st.subheader("2. Nhận diện (sửa nếu sai)")
        st.caption("0 là ô trống. Bấm vào ô để sửa số.")
        edited = st.data_editor(
            pd.DataFrame(board, columns=COLUMNS),
            hide_index=True,
            column_config=CELL_CONFIG,
            key=file.file_id,
        )
        board = np.clip(edited.fillna(0).to_numpy(dtype=int), 0, 9)
    with col3:
        st.subheader("3. Lời giải")
        solved = solve_cached(board)
        if solved is None:
            st.warning("Bàn cờ mâu thuẫn hoặc vô nghiệm, hãy sửa lại các số nhận diện sai.")
        else:
            st.dataframe(highlight_new_cells(solved, board), hide_index=True)
            st.caption("Ô màu xanh là số do chương trình điền.")


if __name__ == "__main__":
    main()
