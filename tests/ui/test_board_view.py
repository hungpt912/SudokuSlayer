
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sudoku_ai.solver import solve  # noqa: E402
from sudoku_ai.ui import board_view  # noqa: E402
from sudoku_ai.ui.board_view import COLUMNS, highlight_new_cells, to_frame  # noqa: E402

PUZZLE = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])


# ---------- to_frame ----------

def test_to_frame_shape_and_columns():
    frame = to_frame(PUZZLE)
    assert frame.shape == (9, 9)
    assert list(frame.columns) == COLUMNS == [str(i) for i in range(1, 10)]


def test_to_frame_keeps_values():
    assert (to_frame(PUZZLE).to_numpy() == PUZZLE).all()


def test_to_frame_accepts_list_of_lists():
    assert (to_frame(PUZZLE.tolist()).to_numpy() == PUZZLE).all()


def test_to_frame_does_not_modify_board():
    original = PUZZLE.copy()
    to_frame(PUZZLE)
    assert (PUZZLE == original).all()


# ---------- bảng style ----------

def test_new_cells_match_empty_cells():
    styles = board_view._style_matrix(PUZZLE)
    green = styles == board_view.NEW_CELL_STYLE
    assert (green == (PUZZLE == 0)).all()          # ô xanh đúng bằng các ô trống của đề


def test_given_cells_never_green():
    styles = board_view._style_matrix(PUZZLE)
    assert not (styles[PUZZLE != 0] == board_view.NEW_CELL_STYLE).any()


def test_block_shading_alternates_on_full_board():
    full = solve(PUZZLE)                           # không còn ô trống -> chỉ còn nền khối
    styles = board_view._style_matrix(full)
    assert not (styles == board_view.NEW_CELL_STYLE).any()
    shaded = styles == board_view.BLOCK_STYLE
    assert shaded.sum() == 36                      # 4 khối 3x3 được tô nền, mỗi khối 9 ô
    assert not shaded[0, 0] and shaded[0, 3] and not shaded[0, 6]
    assert shaded[3, 0] and not shaded[3, 3] and shaded[3, 6]


# ---------- highlight_new_cells ----------

def test_highlight_returns_styler_with_solved_values():
    solved = solve(PUZZLE)
    styler = highlight_new_cells(solved, PUZZLE)
    assert (styler.data.to_numpy() == solved).all()
    assert list(styler.data.columns) == COLUMNS


def test_highlight_colors_exactly_the_new_cells():
    solved = solve(PUZZLE)
    html = highlight_new_cells(solved, PUZZLE).to_html()
    # pandas gộp các ô cùng style vào một quy tắc CSS: "#T_x_row0_col2, #T_x_row0_col3 {... màu xanh ...}"
    rule = re.search(r"([^{}]+)\{[^}]*c8f0d2[^}]*\}", html).group(1)
    colored = {(int(r), int(c)) for r, c in re.findall(r"row(\d+)_col(\d+)", rule)}
    empty = {(int(r), int(c)) for r, c in np.argwhere(PUZZLE == 0)}
    assert colored == empty


def test_highlight_does_not_modify_inputs():
    solved = solve(PUZZLE)
    solved_copy, puzzle_copy = solved.copy(), PUZZLE.copy()
    highlight_new_cells(solved, PUZZLE).to_html()
    assert (solved == solved_copy).all() and (PUZZLE == puzzle_copy).all()