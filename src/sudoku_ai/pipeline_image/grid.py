import cv2
import numpy as np

from sudoku_ai.config import WARP_SIZE


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

# Find board corners and warp board to square image
def detect_and_warp(preprocessed):
    mask = preprocessed["detection_mask"]
    original_gray = preprocessed["original_gray"]

    # Find contours and check large, convex shapes with four corners
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    minimum_area = mask.shape[0] * mask.shape[1] * 0.10
    selected_polygon = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            break

        perimeter = cv2.arcLength(contour, True)

        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            selected_polygon = polygon
            break

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