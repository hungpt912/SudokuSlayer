import cv2
import numpy as np

from sudoku_ai.config import (GRID_SIZE, WARP_SIZE, CELL_SIZE, CELL_MARGIN, DIGIT_SIZE)


# Extract and normalize the digit from cell image
def normalize_cell(cell_gray):
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)

    # Change cell to binary
    threshold_value, binary = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected regions and avoid noise
    count, labels, statistics, centroids = (cv2.connectedComponentsWithStats(binary, connectivity=8))

    minimum_area = cell_gray.size * 0.02
    minimum_height = cell_gray.shape[0] * 0.25

    best_label = 0
    best_area = 0

    for label in range(1, count):
        area = statistics[label, cv2.CC_STAT_AREA]
        height = statistics[label, cv2.CC_STAT_HEIGHT]

        if area < minimum_area or height < minimum_height:
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

    padding = 4
    content_size = DIGIT_SIZE - 2 * padding

    resize_ratio = min(content_size / width, content_size / height)

    new_width = max(1, round(width * resize_ratio))
    new_height = max(1, round(height * resize_ratio))

    interpolation = cv2.INTER_AREA

    if resize_ratio > 1:
        interpolation = cv2.INTER_LINEAR

    resized_digit = cv2.resize(digit, (new_width, new_height), interpolation=interpolation)

    left = (DIGIT_SIZE - new_width) // 2
    top = (DIGIT_SIZE - new_height) // 2

    canvas[top:top + new_height, left:left + new_width] = resized_digit

    return canvas


# Divide the image into 9x9 cells
def split_and_normalize(board_gray):
    if board_gray.shape != (WARP_SIZE, WARP_SIZE):
        raise ValueError("Square image not be of the correct size (WARP_SIZE).")

    cell_images = []
    margin = int(CELL_SIZE * CELL_MARGIN)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            top = row * CELL_SIZE
            left = column * CELL_SIZE

            cell = board_gray[top:top + CELL_SIZE, left:left + CELL_SIZE]

            inner_cell = cell[margin:CELL_SIZE - margin, margin:CELL_SIZE - margin]

            normalized_cell = normalize_cell(inner_cell)
            cell_images.append(normalized_cell)

    batch = np.array(cell_images, dtype=np.float32)
    batch = batch / 255.0
    batch = np.expand_dims(batch, axis=-1)

    return batch