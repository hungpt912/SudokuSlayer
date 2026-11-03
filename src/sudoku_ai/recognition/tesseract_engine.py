"""Recognize digits with Tesseract; requires a local Tesseract OCR installation."""

import cv2
import numpy as np

from sudoku_ai.recognition.base import DigitEngine


class TesseractEngine(DigitEngine):
    """Nhận diện chữ số bằng Tesseract (cần cài Tesseract OCR trên máy)."""

    def __init__(self, tesseract_cmd=None):
        try:
            import pytesseract
        except ImportError as err:
            raise RuntimeError("Thiếu thư viện pytesseract: pip install pytesseract") from err

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError as err:
            raise RuntimeError(
                "Chưa cài Tesseract OCR (hoặc chưa có trong PATH)."
            ) from err
        self._ocr = pytesseract

    def predict(self, cell_image: np.ndarray) -> int:
        """Trả về 0 nếu ô trống, ngược lại là chữ số 1..9."""
        image = np.asarray(cell_image)
        if image.size == 0:
            raise ValueError("cell_image must not be empty")
        if image.ndim == 3:
            image = image[..., 0]
        if image.ndim != 2:
            raise ValueError("Expected a grayscale cell image")

        # Ảnh từ normalize_cell: chữ trắng trên nền đen, ô trống đen hoàn toàn.
        if np.max(image) <= 0:
            return 0

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 \
                else np.clip(image, 0, 255).astype(np.uint8)

        # Tesseract đọc tốt nhất chữ đen trên nền trắng, có viền và đủ lớn.
        image = cv2.bitwise_not(image)
        image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

        text = self._ocr.image_to_string(
            image, config="--psm 13 --oem 3 -c tessedit_char_whitelist=123456789"
        ).strip()
        return int(text[0]) if text and text[0] in "123456789" else 0
