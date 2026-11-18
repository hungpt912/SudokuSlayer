"""Tìm khung bảng Sudoku và nắn phối cảnh về ảnh vuông."""

import cv2
import numpy as np

from sudoku_ai.config import GRID_SIZE, WARP_SIZE

MIN_AREA_RATIO = 0.10      # Khung phải chiếm ít nhất 10% diện tích ảnh
POLY_EPSILON = 0.02        # Sai số xấp xỉ đa giác = 2% chu vi
PREVIEW_CELL = 30          # Cạnh ô khi nắn thử ứng viên (bảng 270x270)
PREVIEW_SIDE = GRID_SIZE * PREVIEW_CELL
LINE_RADIUS = 6            # Bán kính tìm đường kẻ quanh vị trí lý thuyết
MIN_GRID_SCORE = 0.12      # Điểm thấp hơn ngưỡng này coi như không có lưới
SCORE_GAIN = 1.25          # Ứng viên mới phải tốt hơn 25% mới thay ứng viên ban đầu


def order_corners(polygon):
    """Sắp 4 góc theo thứ tự: trên trái, trên phải, dưới phải, dưới trái."""
    points = polygon.reshape(4, 2).astype(np.float32)

    # Bắt đầu từ góc có x + y nhỏ nhất (trên trái)
    start = int(np.argmin(points.sum(axis=1)))
    ordered = np.roll(points, -start, axis=0)

    # Đảo chiều nếu các góc đang đi ngược chiều kim đồng hồ
    first_edge = ordered[1] - ordered[0]
    second_edge = ordered[2] - ordered[1]
    if first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0] < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _preview_ink(corners, gray):
    """Nắn thử vùng tứ giác về 270x270 và trả về mặt nạ nét mực."""
    last = PREVIEW_SIDE - 1
    square = np.float32([[0, 0], [last, 0], [last, last], [0, last]])
    matrix = cv2.getPerspectiveTransform(corners, square)
    board = cv2.warpPerspective(gray, matrix, (PREVIEW_SIDE, PREVIEW_SIDE))
    return cv2.adaptiveThreshold(board, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 21, 2)


def find_grid_hull(contours, gray, minimum_area):
    """Dùng bao lồi khi viền ngoài bị đứt; chỉ nhận nếu bên trong có đủ đường kẻ."""
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < minimum_area:
            continue
        polygon = cv2.approxPolyDP(hull, POLY_EPSILON * cv2.arcLength(hull, True), True)
        if len(polygon) != 4:
            continue

        # Bao lồi có thể ôm cả chữ bên ngoài: yêu cầu ít nhất 6/8 đường kẻ mỗi chiều
        ink = _preview_ink(order_corners(polygon), gray)
        for axis in (0, 1):
            coverage = (ink > 0).mean(axis=axis)
            lines = sum(coverage[pos - LINE_RADIUS:pos + LINE_RADIUS + 1].max() >= 0.40
                        for pos in range(PREVIEW_CELL, PREVIEW_SIDE, PREVIEW_CELL))
            if lines < GRID_SIZE - 3:
                break
        else:
            return polygon
    return None


def score_grid(polygon, gray):
    """Điểm lưới: mật độ mực ở 8 đường kẻ trừ mật độ trong ô, nhân căn tỉ lệ diện tích."""
    corners = order_corners(polygon)
    ink = _preview_ink(corners, gray)
    scores = []
    for axis in (0, 1):
        coverage = (ink > 0).mean(axis=axis)
        peaks = [coverage[pos - LINE_RADIUS:pos + LINE_RADIUS + 1].max()
                 for pos in range(PREVIEW_CELL, PREVIEW_SIDE, PREVIEW_CELL)]
        interiors = [coverage[pos + 10:pos + 21].mean() for pos in range(0, PREVIEW_SIDE, PREVIEW_CELL)]
        scores.append(max(0.0, np.mean(peaks) - np.mean(interiors)))

    # Nhân căn tỉ lệ diện tích để một khối 3x3 không thắng cả bảng
    area_ratio = cv2.contourArea(corners) / gray.size
    return min(scores) * np.sqrt(area_ratio)


def grid_polygons(contours, minimum_area, image_shape, include_rectangle=False):
    """Sinh các tứ giác lồi ứng viên từ đường bao và bao lồi."""
    for contour in contours:
        hull = cv2.convexHull(contour)
        if cv2.contourArea(hull) < minimum_area:
            continue
        polygons = [cv2.approxPolyDP(points, POLY_EPSILON * cv2.arcLength(points, True), True)
                    for points in (contour, hull)]

        # Lưới bị cắt ra ngoài ảnh (chạm >= 3 cạnh): suy góc bằng hình chữ nhật bao
        x, y, width, height = cv2.boundingRect(contour)
        clipped_edges = sum((x <= 1, y <= 1, x + width >= image_shape[1] - 1,
                             y + height >= image_shape[0] - 1))
        if include_rectangle and clipped_edges >= 3:
            rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
            rectangle = np.clip(rectangle, [0, 0], [image_shape[1] - 1, image_shape[0] - 1])
            polygons.append(rectangle.astype(np.float32))

        for polygon in polygons:
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                yield polygon


def select_grid_candidate(preprocessed, preferred, contour_groups, minimum_area):
    """Chọn tứ giác có điểm lưới cao nhất trong mọi ứng viên."""
    gray = preprocessed["blurred"]
    preferred_score = score_grid(preferred, gray) if preferred is not None else 0.0
    best, best_score = preferred, preferred_score

    candidates = []
    for contours in contour_groups:
        candidates.extend(grid_polygons(contours, minimum_area, gray.shape))

    # Thêm ứng viên từ ảnh chỉ giữ nét thẳng dài (nối viền chấm chấm theo từng chiều)
    mask = cv2.adaptiveThreshold(preprocessed["resized_gray"], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 1)
    length = max(9, min(mask.shape) // 6)
    lines = []
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

    if best_score < MIN_GRID_SCORE:
        return None
    # Chỉ thay khi tốt hơn rõ rệt, tránh dịch một khung đã khớp
    if preferred_score < MIN_GRID_SCORE or best_score > preferred_score * SCORE_GAIN:
        return best
    return preferred


def detect_and_warp(preprocessed):
    """Tìm khung bảng trên ảnh thu nhỏ, nắn ảnh xám gốc về WARP_SIZE x WARP_SIZE."""
    mask = preprocessed["detection_mask"]
    minimum_area = mask.shape[0] * mask.shape[1] * MIN_AREA_RATIO

    # Thử 3 mức đóng hình thái; lấy tứ giác lồi lớn đầu tiên làm ứng viên ban đầu
    selected_polygon = None
    contour_groups = []
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
            polygon = cv2.approxPolyDP(contour, POLY_EPSILON * cv2.arcLength(contour, True), True)
            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                selected_polygon = polygon
                break

    if selected_polygon is None:
        selected_polygon = find_grid_hull(contour_groups[-1], preprocessed["blurred"], minimum_area)
    selected_polygon = select_grid_candidate(preprocessed, selected_polygon, contour_groups, minimum_area)
    if selected_polygon is None:
        raise ValueError("Cannot find suitable quadrilateral for the Sudoku board.")

    # Đổi tọa độ góc từ ảnh thu nhỏ về ảnh gốc
    corners_original = order_corners(selected_polygon)
    corners_original[:, 0] /= preprocessed["scale_x"]
    corners_original[:, 1] /= preprocessed["scale_y"]

    last = WARP_SIZE - 1
    destination = np.float32([[0, 0], [last, 0], [last, last], [0, last]])
    forward_matrix = cv2.getPerspectiveTransform(corners_original, destination)
    board_gray = cv2.warpPerspective(preprocessed["original_gray"], forward_matrix, (WARP_SIZE, WARP_SIZE))
    # Ma trận ngược dùng khi cần vẽ lời giải lên ảnh gốc
    inverse_matrix = cv2.getPerspectiveTransform(destination, corners_original)

    return {
        "board_gray": board_gray,
        "corners_original": corners_original,
        "h_original_to_board": forward_matrix,
        "h_board_to_original": inverse_matrix,
    }
