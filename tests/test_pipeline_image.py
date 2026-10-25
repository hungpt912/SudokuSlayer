import cv2
import numpy as np
import pytest

from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DIGIT_SIZE, GRID_SIZE, MAX_IMAGE_SIDE, WARP_SIZE
from sudoku_ai.pipeline_image.cells import split_and_normalize, split_cells
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess


def create_test_image():
    board = np.full((WARP_SIZE, WARP_SIZE), 255, dtype=np.uint8)
    cell_size = WARP_SIZE // GRID_SIZE
    digit_positions = {(0, 0): "5", (4, 4): "3", (8, 8): "7"}

    for index in range(GRID_SIZE + 1):
        position = min(index * cell_size, WARP_SIZE - 1)
        thickness = 3 if index % 3 == 0 else 1
        cv2.line(board, (position, 0), (position, WARP_SIZE - 1), 0, thickness)
        cv2.line(board, (0, position), (WARP_SIZE - 1, position), 0, thickness)

    for position, digit in digit_positions.items():
        row, column = position
        text_size, baseline = cv2.getTextSize(digit, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_width, text_height = text_size
        x = column * cell_size + (cell_size - text_width) // 2
        y = row * cell_size + (cell_size + text_height) // 2
        cv2.putText(board, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

    source_corners = np.array([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]], dtype=np.float32)
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
