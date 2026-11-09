import json

import cv2
import numpy as np
import pytest

from preview_pipeline_image import preview_image
from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DATA_DIR, DIGIT_SIZE, GRID_SIZE, MAX_IMAGE_SIDE, WARP_SIZE, WICHT_DIR
from sudoku_ai.dataset import read_corners
from sudoku_ai.pipeline_image.cells import crop_cell, find_cell_boundaries, normalize_cell, split_and_normalize, split_cells
from sudoku_ai.pipeline_image.grid import detect_and_warp, find_grid_hull
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image


def create_test_image():
    board_size = 450
    last_pixel = board_size - 1
    board = np.full((board_size, board_size), 255, dtype=np.uint8)
    cell_size = board_size // GRID_SIZE
    digit_positions = {(0, 0): "5", (4, 4): "3", (8, 8): "7"}

    for index in range(GRID_SIZE + 1):
        position = min(index * cell_size, last_pixel)
        thickness = 3 if index % 3 == 0 else 1
        cv2.line(board, (position, 0), (position, last_pixel), 0, thickness)
        cv2.line(board, (0, position), (last_pixel, position), 0, thickness)

    for position, digit in digit_positions.items():
        row, column = position
        text_size, baseline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_width, text_height = text_size
        x = column * cell_size + (cell_size - text_width) // 2
        y = row * cell_size + (cell_size + text_height) // 2
        cv2.putText(board, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    source_corners = np.array([[0, 0], [last_pixel, 0], [last_pixel, last_pixel], [0, last_pixel]], dtype=np.float32)
    expected_corners = np.array([[100, 70], [560, 100], [590, 540], [70, 510]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_corners, expected_corners)
    image_gray = cv2.warpPerspective(board, transform, (700, 600), borderValue=255)
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    return image, expected_corners, digit_positions


def test_preprocess_resize():
    original_height = MAX_IMAGE_SIDE + 200
    original_width = MAX_IMAGE_SIDE + 400
    image = np.full((original_height, original_width, 3), 255, dtype=np.uint8)
    image_before = image.copy()

    result = preprocess(image)
    mask = result["detection_mask"]
    mask_height, mask_width = mask.shape

    np.testing.assert_array_equal(image, image_before)
    assert result["original_gray"].shape == (original_height, original_width)
    assert max(mask_height, mask_width) <= MAX_IMAGE_SIDE
    assert mask.dtype == np.uint8
    assert np.all((mask == 0) | (mask == 255))
    assert result["scale_x"] == pytest.approx(mask_width / original_width)
    assert result["scale_y"] == pytest.approx(mask_height / original_height)


def test_complete_pipeline():
    image, expected_corners, digit_positions = create_test_image()

    preprocessed = preprocess(image)
    grid_result = detect_and_warp(preprocessed)
    batch = split_and_normalize(grid_result["board_gray"])

    assert grid_result["board_gray"].shape == (WARP_SIZE, WARP_SIZE)
    np.testing.assert_allclose(grid_result["corners_original"], expected_corners, atol=8)

    assert batch.shape == (81, DIGIT_SIZE, DIGIT_SIZE, 1)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all()
    assert float(batch.min()) >= 0.0
    assert float(batch.max()) <= 1.0

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            index = row * GRID_SIZE + column
            has_digit = bool(np.any(batch[index] > 0))
            expected_digit = (row, column) in digit_positions
            assert has_digit == expected_digit, f"Unexpected content at row {row}, column {column}"


def test_preview_exports_cnn_input(tmp_path):
    image, _, _ = create_test_image()
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), image)
    directory = preview_image(path, tmp_path / "preview")
    board = detect_and_warp(preprocess(image))["board_gray"]
    expected = split_and_normalize(board)

    for index, cell in enumerate(expected, start=1):
        cell_path = directory / "cell" / f"cell_{index:02d}.png"
        actual = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(actual.astype(np.float32) / 255, cell[..., 0])


def test_split_cells_preserves_grayscale_and_order():
    board_gray = np.zeros((WARP_SIZE, WARP_SIZE), dtype=np.uint8)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            value = 30 + 2 * (row * GRID_SIZE + column)
            top = row * CELL_SIZE
            left = column * CELL_SIZE
            board_gray[top:top + CELL_SIZE, left:left + CELL_SIZE] = value
            cv2.rectangle(board_gray, (left, top), (left + CELL_SIZE - 1, top + CELL_SIZE - 1), 0, 2)

    cells = split_cells(board_gray)
    inner_size = CELL_SIZE - 2 * int(CELL_SIZE * CELL_MARGIN)

    assert len(cells) == GRID_SIZE * GRID_SIZE

    for index, cell in enumerate(cells):
        assert inner_size <= cell.shape[0] <= CELL_SIZE - 4
        assert inner_size <= cell.shape[1] <= CELL_SIZE - 4
        assert cell.dtype == np.uint8
        assert np.all(cell == 30 + 2 * index)


def test_crop_keeps_narrow_stroke_near_grid():
    cell = np.full((50, 50), 230, dtype=np.uint8)
    cv2.rectangle(cell, (0, 0), (49, 49), 0, 1)
    cv2.rectangle(cell, (4, 8), (5, 28), 40, -1)
    before = cell.copy()
    cropped = crop_cell(cell)
    assert np.count_nonzero(cropped == 40) == np.count_nonzero(cell == 40)
    assert not np.any(cropped == 0)
    np.testing.assert_array_equal(cell, before)


def test_empty_cells():
    board_gray = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)

    batch = split_and_normalize(board_gray)

    assert batch.shape == (81, DIGIT_SIZE, DIGIT_SIZE, 1)
    assert np.count_nonzero(batch) == 0


def test_image_without_grid():
    image = np.full((600, 700, 3), 255, dtype=np.uint8)
    preprocessed = preprocess(image)

    with pytest.raises(ValueError):
        detect_and_warp(preprocessed)


@pytest.mark.parametrize("name", ["image25.jpg", "image34.jpg"])
def test_grid_reaches_photo_edge(name):
    path = WICHT_DIR / "images" / name
    if not path.is_file():
        pytest.skip(f"Missing Wicht image: {path}")
    actual = cv2.convexHull(detect_and_warp(preprocess(read_image(path)))["corners_original"])
    expected = cv2.convexHull(read_corners(path))
    overlap, _ = cv2.intersectConvexConvex(actual, expected)
    union = cv2.contourArea(actual) + cv2.contourArea(expected) - overlap
    assert overlap / union >= 0.90


def test_hull_rejects_frame_without_internal_grid():
    paper = np.full((270, 270), 200, dtype=np.uint8)
    outline = np.int32([[5, 5], [260, 5], [260, 260], [5, 260]])
    cv2.polylines(paper, [outline], True, 20, 3)
    cv2.putText(paper, "123", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 50, 3)
    assert find_grid_hull([outline], paper, paper.size * 0.10) is None


def test_dotted_grid_uses_whole_board():
    board = np.full((640, 640), 240, dtype=np.uint8)
    positions = [5 + index * 70 for index in range(10)]
    for position in positions:
        for offset in range(5, 637, 9):
            cv2.circle(board, (offset, position), 2, 20, -1)
            cv2.circle(board, (position, offset), 2, 20, -1)
    for position in (positions[3], positions[6]):
        cv2.line(board, (5, position), (635, position), 20, 3)
        cv2.line(board, (position, 5), (position, 635), 20, 3)
    image = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    corners = detect_and_warp(preprocess(image))["corners_original"]
    expected = np.float32([[5, 5], [635, 5], [635, 635], [5, 635]])
    overlap, _ = cv2.intersectConvexConvex(corners, expected)
    union = cv2.contourArea(corners) + cv2.contourArea(expected) - overlap
    assert overlap / union >= 0.90


@pytest.mark.parametrize("name", ["3bor2sty6q0d1.webp", "gqx094yuu3ec1.webp", "hwthrt5zv5cc1.webp"])
def test_lexski_grid_excludes_subgrids_and_menus(name):
    directory = DATA_DIR / "Lexski"
    path = directory / name
    if not path.is_file():
        pytest.skip(f"Missing Lexski image: {path}")
    samples = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sample = next(row for row in samples if row["file"] == name)
    reference = cv2.convexHull(np.float32(sample["corners"]))
    actual = cv2.convexHull(detect_and_warp(preprocess(read_image(path)))["corners_original"])
    overlap, _ = cv2.intersectConvexConvex(actual, reference)
    union = cv2.contourArea(actual) + cv2.contourArea(reference) - overlap
    assert overlap / union >= 0.90


def test_uncertain_cell_keeps_grayscale_for_the_model():
    random = np.random.default_rng(42)
    background = np.tile(np.linspace(110, 170, 38), (38, 1))
    cell = np.clip(background + random.normal(0, 1, background.shape), 0, 255).astype(np.uint8)
    cv2.line(cell, (0, 0), (37, 0), 50, 2)
    cv2.line(cell, (0, 0), (0, 37), 50, 2)
    cell = cv2.GaussianBlur(cell, (5, 5), 1)
    before = cell.copy()

    normalized = normalize_cell(cell)

    assert normalized.shape == (DIGIT_SIZE, DIGIT_SIZE)
    assert len(np.unique(normalized)) > 10
    np.testing.assert_array_equal(cell, before)


@pytest.mark.parametrize("digit", ["1", "6", "8"])
def test_blurred_digit_preserves_shape(digit):
    cell = np.full((38, 38), 180, dtype=np.uint8)
    cv2.putText(cell, digit, (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 40, 2, cv2.LINE_AA)
    cell = cv2.GaussianBlur(cell, (7, 7), 1.3)

    normalized = normalize_cell(cell)

    assert normalized.shape == (DIGIT_SIZE, DIGIT_SIZE)
    assert normalized.dtype == np.uint8
    assert np.count_nonzero(normalized) > 0
    assert np.unique(normalized).size > 10

    # Closed loops in 6 and 8 must not become solid blobs after normalization
    if digit in ("6", "8"):
        mask = (normalized > 127).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = np.count_nonzero(hierarchy[0, :, 3] >= 0)
        assert hole_count == (1 if digit == "6" else 2)


def test_digit_is_kept_beside_grid_remnant():
    cell = np.full((38, 38), 200, dtype=np.uint8)
    cv2.putText(cell, "1", (9, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 30, 2, cv2.LINE_AA)
    expected = normalize_cell(cell)
    cv2.line(cell, (0, 0), (37, 0), 0, 2)
    cv2.line(cell, (0, 0), (0, 37), 0, 2)

    # A dark border must preserve the shape of the digit in the interior
    actual_mask = normalize_cell(cell)[4:-4, 4:-4] > 127
    expected_mask = expected[4:-4, 4:-4] > 127
    overlap = np.count_nonzero(actual_mask & expected_mask)
    combined = np.count_nonzero(actual_mask | expected_mask)
    assert overlap / combined > 0.90


def test_resizing_preserves_single_pixel_stroke():
    cell = np.full((30, 24), 230, dtype=np.uint8)
    cv2.line(cell, (12, 5), (12, 24), 10, 1)

    normalized = normalize_cell(cell)

    assert np.count_nonzero(normalized) > 0


def test_normalization_keeps_a_small_mark_small():
    cell = np.full((50, 50), 230, dtype=np.uint8)
    cv2.rectangle(cell, (25, 25), (27, 27), 10, -1)
    normalized = normalize_cell(cell)
    assert 0 < np.count_nonzero(normalized > 127) <= 9


def test_split_cells_follows_shifted_grid_lines():
    board = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)
    scale = WARP_SIZE / 450
    rows = [round(position * scale) for position in [0, 58, 94, 158, 194, 255, 295, 358, 394, 450]]
    columns = [round(position * scale) for position in [0, 44, 108, 145, 208, 244, 306, 344, 408, 450]]

    for position in rows[1:-1]:
        cv2.line(board, (0, position), (WARP_SIZE - 1, position), 0, 3)
    for position in columns[1:-1]:
        cv2.line(board, (position, 0), (position, WARP_SIZE - 1), 0, 3)

    before = board.copy()
    detected_rows, detected_columns = find_cell_boundaries(board)
    np.testing.assert_allclose(detected_rows, rows, atol=1)
    np.testing.assert_allclose(detected_columns, columns, atol=1)
    cells = split_cells(board)
    assert len(cells) == GRID_SIZE * GRID_SIZE
    assert all(np.all(cell == 255) for cell in cells)
    np.testing.assert_array_equal(board, before)


def test_missing_grid_lines_use_equal_cells():
    board = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)
    expected = list(range(0, WARP_SIZE + 1, CELL_SIZE))

    assert find_cell_boundaries(board) == [expected, expected]


def test_cell_boundaries_follow_grid_pitch_with_extra_footer():
    board = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)
    rows = [10 + 46 * i for i in range(10)]
    for position in rows:
        cv2.line(board, (0, position), (WARP_SIZE - 1, position), 0, 2)
    detected_rows, _ = find_cell_boundaries(board)
    np.testing.assert_allclose(detected_rows, rows, atol=1)


def test_sloping_bottom_edge_keeps_digit_and_excludes_footer():
    board = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)
    scale = WARP_SIZE / 450
    last = WARP_SIZE - 1
    for index in range(GRID_SIZE + 1):
        left_y = round((2 + 48 * index) * scale)
        right_y = round((2 + 49.5 * index) * scale)
        cv2.line(board, (0, left_y), (last, right_y), 0, 3)
        x = min(last, index * CELL_SIZE)
        cv2.line(board, (x, 0), (x, last), 0, 2)
    cv2.rectangle(board, (round(20 * scale), round(426 * scale)),
                  (round(23 * scale), round(429 * scale)), 110, -1)
    cv2.rectangle(board, (round(20 * scale), round(442 * scale)),
                  (round(23 * scale), round(445 * scale)), 80, -1)

    bottom_left = split_cells(board)[(GRID_SIZE - 1) * GRID_SIZE]
    assert np.any(bottom_left == 110)
    assert not np.any(bottom_left == 80)


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3)])
def test_tiny_empty_cell(shape):
    normalized = normalize_cell(np.full(shape, 255, dtype=np.uint8))

    assert normalized.shape == (DIGIT_SIZE, DIGIT_SIZE)
    assert np.count_nonzero(normalized) == 0
