"""Đọc ảnh và tạo mặt nạ nhị phân để tìm lưới."""

import cv2
import numpy as np

from sudoku_ai.config import MAX_IMAGE_SIDE


def read_image(image_path):
    """Đọc ảnh màu BGR từ đường dẫn."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read image from the provided path.")
    return image


def preprocess(image, connect_lines=False):
    """Ảnh BGR -> ảnh xám gốc, ảnh thu nhỏ, ảnh làm mờ, mặt nạ nhị phân và tỉ lệ thu nhỏ.

    connect_lines=True: đóng hình thái 3x3 để nối các nét kẻ bị đứt.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot process empty image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input not valid BGR image.")

    # Giữ ảnh xám gốc để nắn bảng cho nét
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = original_gray.shape

    # Thu nhỏ ảnh lớn để tìm lưới nhanh hơn
    resize_ratio = min(1.0, MAX_IMAGE_SIDE / max(original_height, original_width))
    resized_width = max(1, round(original_width * resize_ratio))
    resized_height = max(1, round(original_height * resize_ratio))
    small_gray = cv2.resize(original_gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Làm mờ để giảm nhiễu, rồi phân ngưỡng thích nghi (nét trắng trên nền đen)
    blurred = cv2.GaussianBlur(small_gray, (5, 5), 0)
    detection_mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 31, 5)

    if connect_lines:
        kernel = np.ones((3, 3), dtype=np.uint8)
        detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_CLOSE, kernel)

    return {
        "original_gray": original_gray,
        "resized_gray": small_gray,
        "blurred": blurred,
        "detection_mask": detection_mask,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
    }
