from sudoku_ai.recognition.base import DigitEngine


def get_engine(name: str) -> DigitEngine:
    """Tạo engine theo tên (xem config.AVAILABLE_ENGINES)."""
    if name == "tesseract":
        from sudoku_ai.recognition.tesseract_engine import TesseractEngine

        return TesseractEngine()
    raise ValueError(f"Engine '{name}' chưa được hỗ trợ.")


__all__ = ["DigitEngine", "get_engine"]
