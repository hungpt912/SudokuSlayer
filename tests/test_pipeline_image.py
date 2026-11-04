import cv2
import numpy as np
import pytest

from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DIGIT_SIZE, GRID_SIZE, MAX_IMAGE_SIDE, WARP_SIZE
from sudoku_ai.pipeline_image.cells import find_cell_boundaries, normalize_cell, split_and_normalize, split_cells
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess


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
        assert cell.shape == (inner_size, inner_size)
        assert cell.dtype == np.uint8
        assert np.all(cell == 30 + 2 * index)


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


def test_empty_cell_with_shadow_and_grid_remnants():
    random = np.random.default_rng(42)
    background = np.tile(np.linspace(110, 170, 38), (38, 1))
    cell = np.clip(background + random.normal(0, 1, background.shape), 0, 255).astype(np.uint8)
    cv2.line(cell, (0, 0), (37, 0), 50, 2)
    cv2.line(cell, (0, 0), (0, 37), 50, 2)
    cell = cv2.GaussianBlur(cell, (5, 5), 1)
    before = cell.copy()

    normalized = normalize_cell(cell)

    assert np.count_nonzero(normalized) == 0
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
    moments = cv2.moments(normalized)
    assert moments["m10"] / moments["m00"] == pytest.approx(13.5, abs=0.6)
    assert moments["m01"] / moments["m00"] == pytest.approx(13.5, abs=0.6)

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

    # Local thresholding can change edge intensities, but must preserve the digit
    actual_mask = normalize_cell(cell) > 127
    expected_mask = expected > 127
    overlap = np.count_nonzero(actual_mask & expected_mask)
    combined = np.count_nonzero(actual_mask | expected_mask)
    assert overlap / combined > 0.90


def test_thinning_preserves_single_pixel_stroke():
    cell = np.full((30, 24), 230, dtype=np.uint8)
    cv2.line(cell, (12, 5), (12, 24), 10, 1)

    normalized = normalize_cell(cell, resize_scale=1.0)

    assert np.count_nonzero(normalized) > 0


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


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3)])
def test_tiny_empty_cell(shape):
    normalized = normalize_cell(np.full(shape, 255, dtype=np.uint8))

    assert normalized.shape == (DIGIT_SIZE, DIGIT_SIZE)
    assert np.count_nonzero(normalized) == 0
