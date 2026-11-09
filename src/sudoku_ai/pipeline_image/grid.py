import cv2
import numpy as np

from sudoku_ai.config import GRID_SIZE, WARP_SIZE


# Order corners as top-left, top-right, bottom-right, bottom-left
def order_corners(polygon):
    points = polygon.reshape(4, 2).astype(np.float32)

    start_index = 0

    for index in range(1, 4):
        current_sum = points[index][0] + points[index][1]
        smallest_sum = points[start_index][0] + points[start_index][1]

        if current_sum < smallest_sum:
            start_index = index

    ordered = np.zeros((4, 2), dtype=np.float32)

    for index in range(4):
        ordered[index] = points[(start_index + index) % 4]

    first_edge = ordered[1] - ordered[0]
    second_edge = ordered[2] - ordered[1]

    direction = (first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])

    if direction < 0:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)

    return ordered


# Recover the outer corners when gaps in the border leave a jagged contour.
def find_grid_hull(contours, gray, minimum_area):
    cell_size = 30
    side = GRID_SIZE * cell_size
    last = side - 1
    square = np.float32([[0, 0], [last, 0], [last, last], [0, last]])
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < minimum_area:
            continue
        polygon = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(polygon) != 4:
            continue

        # A hull can also enclose text. Require internal grid lines in both directions.
        matrix = cv2.getPerspectiveTransform(order_corners(polygon), square)
        board = cv2.warpPerspective(gray, matrix, (side, side))
        ink = cv2.adaptiveThreshold(board, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 21, 2)
        for axis in (0, 1):
            coverage = (ink > 0).mean(axis=axis)
            radius = cell_size // 5
            lines = sum(coverage[pos - radius:pos + radius + 1].max() >= 0.40
                        for pos in range(cell_size, side, cell_size))
            if lines < GRID_SIZE - 3:
                break
        else:
            return polygon
    return None


# Compare ink at the expected eight grid lines with ink inside the cells.
def score_grid(polygon, gray):
    cell_size = 30
    side = GRID_SIZE * cell_size
    last = side - 1
    square = np.float32([[0, 0], [last, 0], [last, last], [0, last]])
    corners = order_corners(polygon)
    matrix = cv2.getPerspectiveTransform(corners, square)
    board = cv2.warpPerspective(gray, matrix, (side, side))
    ink = cv2.adaptiveThreshold(board, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 2)
    scores = []
    for axis in (0, 1):
        coverage = (ink > 0).mean(axis=axis)
        peaks = [coverage[pos - 6:pos + 7].max() for pos in range(cell_size, side, cell_size)]
        interiors = [coverage[pos + 10:pos + 21].mean() for pos in range(0, side, cell_size)]
        scores.append(max(0.0, np.mean(peaks) - np.mean(interiors)))

    # A single 3x3 box must compete with the full board.
    area_ratio = cv2.contourArea(corners) / gray.size
    return min(scores) * np.sqrt(area_ratio)


def grid_polygons(contours, minimum_area, image_shape, include_rectangle=False):
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < minimum_area:
            continue
        polygons = [cv2.approxPolyDP(points, 0.02 * cv2.arcLength(points, True), True)
                    for points in (contour, hull)]
        x, y, width, height = cv2.boundingRect(contour)
        clipped_edges = sum((x <= 1, y <= 1, x + width >= image_shape[1] - 1,
                             y + height >= image_shape[0] - 1))
        # Infer missing corners only when the grid runs out of the photograph.
        if include_rectangle and clipped_edges >= 3:
            rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
            rectangle = np.clip(rectangle, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])
            polygons.append(rectangle.astype(np.float32))
        for polygon in polygons:
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                yield polygon


def select_grid_candidate(preprocessed, preferred, contour_groups, minimum_area):
    gray = preprocessed["blurred"]
    preferred_score = score_grid(preferred, gray) if preferred is not None else 0.0
    best, best_score = preferred, preferred_score
    candidates = []
    for contours in contour_groups:
        candidates.extend(grid_polygons(contours, minimum_area, gray.shape))

    # Isolate long strokes and connect dotted borders independently in each direction.
    mask = cv2.adaptiveThreshold(preprocessed["resized_gray"], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 1)
    lines = []
    length = max(9, min(mask.shape) // 6)
    for close_shape, open_shape in (((1, 9), (1, length)), ((9, 1), (length, 1))):
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(close_shape, dtype=np.uint8))
        lines.append(cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones(open_shape, dtype=np.uint8)))
    contours, _ = cv2.findContours(lines[0] | lines[1], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates.extend(grid_polygons(contours, minimum_area, gray.shape, include_rectangle=True))

    seen = set()
    for polygon in candidates:
        key = order_corners(polygon).tobytes()
        if key in seen:
            continue
        seen.add(key)
        score = score_grid(polygon, gray)
        if score > best_score:
            best, best_score = polygon, score

    if best_score < 0.12:
        return None
    # Require a clear improvement to avoid moving an already well-aligned crop.
    if preferred_score < 0.12 or best_score > preferred_score * 1.25:
        return best
    return preferred


# Find board corners and warp board to square image
def detect_and_warp(preprocessed):
    mask = preprocessed["detection_mask"]
    original_gray = preprocessed["original_gray"]

    minimum_area = mask.shape[0] * mask.shape[1] * 0.10
    selected_polygon = None
    contour_groups = []
    # Keep the first contour as a starting point and compare all three masks.
    for kernel_size in (1, 3, 5):
        candidate_mask = mask
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            candidate_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour_groups.append(contours)
        if selected_polygon is not None:
            continue
        for contour in contours:
            if cv2.contourArea(contour) < minimum_area:
                break
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                selected_polygon = polygon
                break

    if selected_polygon is None:
        selected_polygon = find_grid_hull(contours, preprocessed["blurred"], minimum_area)
    selected_polygon = select_grid_candidate(preprocessed, selected_polygon, contour_groups, minimum_area)
    if selected_polygon is None:
        raise ValueError("Cannot find suitable quadrilateral for the Sudoku board.")

    # Convert detected corners back to original image coordinates
    corners_original = order_corners(selected_polygon)

    for index in range(4):
        corners_original[index][0] /= preprocessed["scale_x"]
        corners_original[index][1] /= preprocessed["scale_y"]

    last_pixel = WARP_SIZE - 1

    destination = np.array([[0, 0], [last_pixel, 0], [last_pixel, last_pixel], [0, last_pixel]], dtype=np.float32)

    forward_matrix = cv2.getPerspectiveTransform(corners_original, destination)

    board_gray = cv2.warpPerspective(original_gray, forward_matrix, (WARP_SIZE, WARP_SIZE))

    inverse_matrix = cv2.getPerspectiveTransform(destination, corners_original) # Convert back to original image coordinates if needed

    return {"board_gray": board_gray, "corners_original": corners_original, "h_original_to_board": forward_matrix, "h_board_to_original": inverse_matrix}
