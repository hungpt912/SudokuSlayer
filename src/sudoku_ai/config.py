"""Hằng số dùng chung cho toàn dự án."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WICHT_DIR = DATA_DIR / "wicht"

GRID_SIZE = 9
WARP_SIZE = 450                      # Cạnh bảng sau khi nắn (pixel), chia hết cho 9
CELL_SIZE = WARP_SIZE // GRID_SIZE   # Cạnh một ô (50 pixel)
CELL_MARGIN = 0.12                   # Tỉ lệ viền tối đa được cắt ở mỗi cạnh ô
DIGIT_SIZE = 28                      # Kích thước ảnh ô đưa vào CNN
DIGIT_RECHECK_CONFIDENCE = 0.65      # Ô có số nhưng xác suất chữ số thấp hơn ngưỡng này sẽ được đọc lại
MAX_IMAGE_SIDE = 1200                # Ảnh có cạnh dài hơn sẽ được thu nhỏ khi tìm lưới

DEFAULT_ENGINE = "cnn"               # Bộ nhận diện mặc định
