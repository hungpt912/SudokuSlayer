"""Hằng số dùng chung toàn dự án."""

GRID_SIZE = 9
WARP_SIZE = 450                      # cạnh ảnh lưới sau khi làm phẳng (px)
CELL_SIZE = WARP_SIZE // GRID_SIZE   # cạnh mỗi ô (px)
CELL_MARGIN = 0.12                   # tỉ lệ viền cắt bỏ trong mỗi ô để loại đường kẻ
DIGIT_SIZE = 28                      # kích thước ảnh chữ số đưa vào engine nhận diện
MAX_IMAGE_SIDE = 1200                # thu nhỏ ảnh đầu vào nếu cạnh dài hơn giá trị này

DEFAULT_ENGINE = "tesseract"
AVAILABLE_ENGINES = ("tesseract",)   # thêm "easyocr", "cnn" khi cài đặt xong
