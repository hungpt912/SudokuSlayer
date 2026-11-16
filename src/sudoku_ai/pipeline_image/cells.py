import cv2
import numpy as np

from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DIGIT_SIZE, GRID_SIZE, WARP_SIZE


# Keep the grayscale ink and correct for changes in paper brightness
def normalize_cell(cell_gray):
    if cell_gray.ndim != 2 or cell_gray.size == 0 or cell_gray.dtype != np.uint8:
        raise ValueError("Cell must be a non-empty uint8 grayscale image.")

    height, width = cell_gray.shape
    ratio = DIGIT_SIZE / max(height, width)
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    image = cv2.resize(cell_gray, size, interpolation=cv2.INTER_AREA)
    # Estimate paper and ink inside the cell so a dark grid edge cannot set the contrast.
    border_y, border_x = size[1] // 8, size[0] // 8
    interior = image[border_y:size[1] - border_y, border_x:size[0] - border_x]
    ink, paper = np.percentile(interior, (5, 90))
    contrast = max(32.0, paper - ink)
    normalized = np.clip((paper - image) * (255.0 / contrast), 0, 255)

    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)
    left = (DIGIT_SIZE - size[0]) // 2
    top = (DIGIT_SIZE - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = normalized
    return canvas


# Recover the cell spacing when a detected frame includes a header or footer.
def refine_cell_boundaries(profile, positions):
    anchors = np.array([i for i in range(1, GRID_SIZE) if profile[positions[i]] >= 0.30])
    if len(anchors) < 4:
        return positions
    observed = np.asarray(positions)[anchors]
    step, offset = np.polyfit(anchors, observed, 1)
    residual = np.sqrt(np.mean((observed - (step * anchors + offset)) ** 2))
    if not (abs(step - CELL_SIZE) > CELL_SIZE * 0.05
            and CELL_SIZE * 0.85 < step < CELL_SIZE * 1.05
            and residual < 1.5 and abs(offset) < CELL_SIZE * 0.30):
        return positions

    refined = []
    radius = int(CELL_SIZE * 0.20)
    for index in range(GRID_SIZE + 1):
        center = round(offset + index * step)
        start = max(0, center - radius)
        window = profile[start:min(len(profile), center + radius + 1)]
        peak = window.max()
        if peak >= 0.30:
            offsets = np.flatnonzero(window >= peak * 0.85)
            refined.append(start + round(np.average(offsets, weights=window[offsets])))
        else:
            refined.append(int(np.clip(center, 0, len(profile))))
    return refined


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
        boundaries.append(refine_cell_boundaries(profile, positions))

    return boundaries


# Trim each grid edge by its measured thickness while keeping nearby digit strokes.
def crop_cell(cell_gray):
    height, width = cell_gray.shape
    margin_y = max(2, int(height * CELL_MARGIN))
    margin_x = max(2, int(width * CELL_MARGIN))
    block_size = min(31, min(height, width) // 2 * 2 - 1)
    mask = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block_size, 5)
    margins = []
    # Exclude the perpendicular borders when measuring horizontal or vertical ink.
    for strip, limit in ((mask[:, margin_x:width - margin_x], margin_y),
                         (mask[margin_y:height - margin_y, :].T, margin_x)):
        coverage = np.mean(strip > 0, axis=1)
        for edge in (coverage, coverage[::-1]):
            line = np.flatnonzero(edge[:limit] >= 0.65)
            # Include the interpolation fringe around the measured dark stroke.
            margins.append(min(limit, max(2, int(line[-1]) + 3)) if line.size else 2)
    top, bottom, left, right = margins
    return cell_gray[top:height - bottom, left:width - right].copy()


def nearby_grid_line(profile, position):
    radius = round(CELL_SIZE * 0.40)
    start = max(0, position - radius)
    window = profile[start:min(len(profile), position + radius + 1)]
    peak = window.max()
    if peak < 0.30:
        return position
    offsets = np.flatnonzero(window >= peak * 0.85)
    return start + round(np.average(offsets, weights=window[offsets]))


# Split the board and crop cell borders before digit normalization
def split_cells(board_gray):
    if board_gray.shape != (WARP_SIZE, WARP_SIZE):
        raise ValueError("Square image not be of the correct size (WARP_SIZE).")

    cell_images = []
    rows, columns = find_cell_boundaries(board_gray)
    mask = cv2.adaptiveThreshold(board_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 5)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, CELL_SIZE), dtype=np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((CELL_SIZE, 1), dtype=np.uint8))

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            top, bottom = rows[row], rows[row + 1]
            left, right = columns[column], columns[column + 1]
            # Measure each cell locally because a photographed grid can slope or curve.
            profile = np.mean(horizontal[:, left:right] > 0, axis=1)
            local_top = nearby_grid_line(profile, top)
            local_bottom = nearby_grid_line(profile, bottom)
            # Keep enough room for the two-pixel margin on each edge.
            if local_bottom - local_top > 4:
                top, bottom = local_top, local_bottom
            profile = np.mean(vertical[top:bottom, :] > 0, axis=0)
            local_left = nearby_grid_line(profile, left)
            local_right = nearby_grid_line(profile, right)
            if local_right - local_left > 4:
                left, right = local_left, local_right
            cell_images.append(crop_cell(board_gray[top:bottom, left:right]))

    return cell_images


# Normalize the 81 cropped cells for digit recognition
def split_and_normalize(board_gray):
    cells = [normalize_cell(cell) for cell in split_cells(board_gray)]
    return np.asarray(cells, dtype=np.float32)[..., None] / 255.0
