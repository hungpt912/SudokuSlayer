import cv2
import numpy as np

from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DIGIT_CONTENT_SIZE, DIGIT_SIZE, GRID_SIZE, WARP_SIZE


# Extract and normalize the digit from cell image
def normalize_cell(cell_gray, resize_scale=2.0):
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)

    if cell_gray.ndim != 2 or cell_gray.size == 0 or cell_gray.dtype != np.uint8:
        raise ValueError("Cell must be a non-empty uint8 grayscale image.")

    if min(cell_gray.shape) < 3:
        return canvas

    # Use a consistent working scale before thresholding
    if resize_scale != 1.0:
        interpolation = cv2.INTER_CUBIC if resize_scale > 1.0 else cv2.INTER_AREA
        cell_gray = cv2.resize(cell_gray, None, fx=resize_scale, fy=resize_scale, interpolation=interpolation)
    cell_height, cell_width = cell_gray.shape
    smallest_side = min(cell_height, cell_width)

    # Compare each pixel with its local background to avoid turning shadows into digits
    block_size = min(61, smallest_side if smallest_side % 2 else smallest_side - 1)
    binary = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)

    # Find connected regions and avoid noise
    count, labels, statistics, centroids = (cv2.connectedComponentsWithStats(binary, connectivity=8))

    minimum_area = cell_gray.size * 0.02
    minimum_height = cell_gray.shape[0] * 0.25

    best_label = 0
    best_area = 0

    for label in range(1, count):
        area = statistics[label, cv2.CC_STAT_AREA]
        width = statistics[label, cv2.CC_STAT_WIDTH]
        height = statistics[label, cv2.CC_STAT_HEIGHT]
        center_x, center_y = centroids[label]

        if area < minimum_area or height < minimum_height:
            continue

        # Reject broad shadows and grid lines that span most of the cropped cell
        if area > cell_gray.size * 0.65 or width > cell_width * 0.80 or height > cell_height * 0.95:
            continue

        # Keep digits near the cell center, including narrow digits such as 1
        if not (cell_width * 0.20 <= center_x <= cell_width * 0.80 and cell_height * 0.15 <= center_y <= cell_height * 0.85):
            continue

        if area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        return canvas

    x = statistics[best_label, cv2.CC_STAT_LEFT]
    y = statistics[best_label, cv2.CC_STAT_TOP]
    width = statistics[best_label, cv2.CC_STAT_WIDTH]
    height = statistics[best_label, cv2.CC_STAT_HEIGHT]

    digit_mask = np.zeros(binary.shape, dtype=np.uint8)
    digit_mask[labels == best_label] = 255

    digit = digit_mask[y:y + height, x:x + width]

    resize_ratio = min(DIGIT_CONTENT_SIZE / width, DIGIT_CONTENT_SIZE / height)

    new_width = max(1, round(width * resize_ratio))
    new_height = max(1, round(height * resize_ratio))

    interpolation = cv2.INTER_AREA

    if resize_ratio > 1:
        interpolation = cv2.INTER_LINEAR

    resized_digit = cv2.resize(digit, (new_width, new_height), interpolation=interpolation)

    left = (DIGIT_SIZE - new_width) // 2
    top = (DIGIT_SIZE - new_height) // 2

    canvas[top:top + new_height, left:left + new_width] = resized_digit

    # Center by stroke mass, as in MNIST, without stretching or clipping the digit
    moments = cv2.moments(canvas)

    if moments["m00"] > 0:
        center = (DIGIT_SIZE - 1) / 2
        shift_x = center - moments["m10"] / moments["m00"]
        shift_y = center - moments["m01"] / moments["m00"]
        shift_x = np.clip(shift_x, -left, DIGIT_SIZE - left - new_width)
        shift_y = np.clip(shift_y, -top, DIGIT_SIZE - top - new_height)
        transform = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
        canvas = cv2.warpAffine(canvas, transform, (DIGIT_SIZE, DIGIT_SIZE))

    return canvas


# Locate internal grid lines near their expected positions
def find_cell_boundaries(board_gray):
    block_size = 2 * round(0.30 * CELL_SIZE) + 1
    mask = cv2.adaptiveThreshold(board_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)
    boundaries = []
    search_radius = int(CELL_SIZE * 0.20)

    for horizontal in (True, False):
        kernel_shape = (1, CELL_SIZE * 2) if horizontal else (CELL_SIZE * 2, 1)
        kernel = np.ones(kernel_shape, dtype=np.uint8)
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        profile = np.mean(lines, axis=1 if horizontal else 0) / 255.0
        positions = [0]

        for index in range(1, GRID_SIZE):
            center = index * CELL_SIZE
            start = center - search_radius
            window = profile[start:center + search_radius + 1]
            peak = window.max()

            # Fall back to equal cells when the line is too faint or incomplete
            if peak < 0.30:
                positions.append(center)
                continue

            offsets = np.flatnonzero(window >= peak * 0.85)
            position = start + round(np.average(offsets, weights=window[offsets]))
            positions.append(position)

        positions.append(WARP_SIZE)
        boundaries.append(positions)

    return boundaries


# Split the board and crop cell borders before digit normalization
def split_cells(board_gray):
    if board_gray.shape != (WARP_SIZE, WARP_SIZE):
        raise ValueError("Square image not be of the correct size (WARP_SIZE).")

    cell_images = []
    rows, columns = find_cell_boundaries(board_gray)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            top, bottom = rows[row], rows[row + 1]
            left, right = columns[column], columns[column + 1]
            margin_y = max(2, int((bottom - top) * CELL_MARGIN))
            margin_x = max(2, int((right - left) * CELL_MARGIN))

            inner_cell = board_gray[top + margin_y:bottom - margin_y, left + margin_x:right - margin_x]

            cell_images.append(inner_cell.copy())

    return cell_images


# Normalize the 81 cropped cells for digit recognition
def split_and_normalize(board_gray):
    cell_images = split_cells(board_gray)
    normalized_cells = []
    # Keep the thresholding scale stable when the rectified board size changes
    resize_scale = 100 / CELL_SIZE

    for cell in cell_images:
        normalized_cells.append(normalize_cell(cell, resize_scale=resize_scale))

    batch = np.array(normalized_cells, dtype=np.float32)
    batch = batch / 255.0
    batch = np.expand_dims(batch, axis=-1)

    return batch
