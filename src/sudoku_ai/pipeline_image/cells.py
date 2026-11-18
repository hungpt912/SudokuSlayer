"""Tách bảng đã nắn thành 81 ô và chuẩn hóa từng ô cho CNN."""

import cv2
import numpy as np

from sudoku_ai.config import CELL_MARGIN, CELL_SIZE, DIGIT_SIZE, GRID_SIZE, WARP_SIZE

MIN_LINE_PEAK = 0.30   # Đỉnh hồ sơ đường kẻ thấp hơn mức này coi như không thấy đường kẻ
PEAK_RATIO = 0.85      # Lấy trung bình có trọng số các điểm đạt từ 85% đỉnh


def normalize_cell(cell_gray):
    """Ô xám uint8 -> ảnh 28x28 uint8: nét sáng trên nền đen, đã cân bằng độ tương phản."""
    if cell_gray.ndim != 2 or cell_gray.size == 0 or cell_gray.dtype != np.uint8:
        raise ValueError("Cell must be a non-empty uint8 grayscale image.")

    # Thu nhỏ giữ tỉ lệ để cạnh dài bằng 28
    height, width = cell_gray.shape
    ratio = DIGIT_SIZE / max(height, width)
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    image = cv2.resize(cell_gray, size, interpolation=cv2.INTER_AREA)

    # Ước lượng mực (p5) và giấy (p90) ở vùng trong, tránh để viền lưới quyết định độ tương phản
    border_y, border_x = size[1] // 8, size[0] // 8
    interior = image[border_y:size[1] - border_y, border_x:size[0] - border_x]
    ink, paper = np.percentile(interior, (5, 90))
    contrast = max(32.0, paper - ink)   # Chặn dưới 32 để không khuếch đại nhiễu ở ô trống
    normalized = np.clip((paper - image) * (255.0 / contrast), 0, 255)

    # Đặt vào giữa khung 28x28
    canvas = np.zeros((DIGIT_SIZE, DIGIT_SIZE), dtype=np.uint8)
    left = (DIGIT_SIZE - size[0]) // 2
    top = (DIGIT_SIZE - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = normalized
    return canvas


def _line_position(profile, center, radius):
    """Vị trí đường kẻ gần center; trả về None nếu đường kẻ quá mờ."""
    start = max(0, center - radius)
    window = profile[start:min(len(profile), center + radius + 1)]
    peak = window.max()
    if peak < MIN_LINE_PEAK:
        return None
    offsets = np.flatnonzero(window >= peak * PEAK_RATIO)
    return start + round(np.average(offsets, weights=window[offsets]))


def refine_cell_boundaries(profile, positions):
    """Khi khung tìm được có lẫn tiêu đề/chú thích, ước lượng lại bước lưới bằng hồi quy tuyến tính."""
    anchors = np.array([i for i in range(1, GRID_SIZE) if profile[positions[i]] >= MIN_LINE_PEAK])
    if len(anchors) < 4:
        return positions
    observed = np.asarray(positions)[anchors]
    step, offset = np.polyfit(anchors, observed, 1)
    residual = np.sqrt(np.mean((observed - (step * anchors + offset)) ** 2))

    # Chỉ áp dụng khi bước lưới lệch rõ (>5%) nhưng vẫn hợp lý và đường hồi quy khớp tốt
    if not (abs(step - CELL_SIZE) > CELL_SIZE * 0.05
            and CELL_SIZE * 0.85 < step < CELL_SIZE * 1.05
            and residual < 1.5 and abs(offset) < CELL_SIZE * 0.30):
        return positions

    refined = []
    radius = int(CELL_SIZE * 0.20)
    for index in range(GRID_SIZE + 1):
        center = round(offset + index * step)
        position = _line_position(profile, center, radius)
        refined.append(int(np.clip(center, 0, len(profile))) if position is None else position)
    return refined


def find_cell_boundaries(board_gray):
    """Tìm 10 vị trí đường kẻ ngang và 10 vị trí đường kẻ dọc trên bảng 450x450."""
    block_size = 2 * round(0.30 * CELL_SIZE) + 1
    mask = cv2.adaptiveThreshold(board_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, 5)
    search_radius = int(CELL_SIZE * 0.20)
    boundaries = []

    for horizontal in (True, False):
        # Mở hình thái với nhân thẳng 1x100 (ngang) hoặc 100x1 (dọc) để chỉ giữ đường kẻ dài
        kernel_shape = (1, CELL_SIZE * 2) if horizontal else (CELL_SIZE * 2, 1)
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(kernel_shape, dtype=np.uint8))
        profile = np.mean(lines, axis=1 if horizontal else 0) / 255.0

        positions = [0]
        for index in range(1, GRID_SIZE):
            center = index * CELL_SIZE
            position = _line_position(profile, center, search_radius)
            # Đường kẻ quá mờ: quay về chia đều
            positions.append(center if position is None else position)
        positions.append(WARP_SIZE)
        boundaries.append(refine_cell_boundaries(profile, positions))

    return boundaries


def crop_cell(cell_gray):
    """Cắt viền lưới theo độ dày đo được (2 pixel đến 12% cạnh ô), giữ nét chữ gần viền."""
    height, width = cell_gray.shape
    margin_y = max(2, int(height * CELL_MARGIN))
    margin_x = max(2, int(width * CELL_MARGIN))
    block_size = min(31, min(height, width) // 2 * 2 - 1)
    mask = cv2.adaptiveThreshold(cell_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block_size, 5)

    # Đo viền trên/dưới (bỏ viền trái/phải) rồi đo viền trái/phải (bỏ viền trên/dưới)
    margins = []
    for strip, limit in ((mask[:, margin_x:width - margin_x], margin_y),
                         (mask[margin_y:height - margin_y, :].T, margin_x)):
        coverage = np.mean(strip > 0, axis=1)
        for edge in (coverage, coverage[::-1]):
            line = np.flatnonzero(edge[:limit] >= 0.65)
            # Cộng 3 pixel để bỏ cả phần viền mờ do nội suy
            margins.append(min(limit, max(2, int(line[-1]) + 3)) if line.size else 2)
    top, bottom, left, right = margins
    return cell_gray[top:height - bottom, left:width - right].copy()


def nearby_grid_line(profile, position):
    """Tinh chỉnh một biên ô theo đường kẻ gần nhất (bán kính 40% cạnh ô)."""
    found = _line_position(profile, position, round(CELL_SIZE * 0.40))
    return position if found is None else found


def split_cells(board_gray):
    """Tách bảng 450x450 thành 81 ảnh ô xám đã cắt viền, theo thứ tự hàng rồi cột."""
    if board_gray.shape != (WARP_SIZE, WARP_SIZE):
        raise ValueError(f"Board image must be {WARP_SIZE}x{WARP_SIZE}.")

    rows, columns = find_cell_boundaries(board_gray)
    mask = cv2.adaptiveThreshold(board_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 5)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, CELL_SIZE), dtype=np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((CELL_SIZE, 1), dtype=np.uint8))

    cell_images = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            top, bottom = rows[row], rows[row + 1]
            left, right = columns[column], columns[column + 1]

            # Đo lại biên cho từng ô vì giấy chụp có thể cong hoặc nghiêng
            profile = np.mean(horizontal[:, left:right] > 0, axis=1)
            local_top, local_bottom = nearby_grid_line(profile, top), nearby_grid_line(profile, bottom)
            if local_bottom - local_top > 4:   # Đủ chỗ cho viền 2 pixel mỗi bên
                top, bottom = local_top, local_bottom

            profile = np.mean(vertical[top:bottom, :] > 0, axis=0)
            local_left, local_right = nearby_grid_line(profile, left), nearby_grid_line(profile, right)
            if local_right - local_left > 4:
                left, right = local_left, local_right

            cell_images.append(crop_cell(board_gray[top:bottom, left:right]))
    return cell_images


def split_and_normalize(board_gray):
    """Bảng 450x450 -> mảng float32 (81, 28, 28, 1) trong [0, 1] cho CNN."""
    cells = [normalize_cell(cell) for cell in split_cells(board_gray)]
    return np.asarray(cells, dtype=np.float32)[..., None] / 255.0
