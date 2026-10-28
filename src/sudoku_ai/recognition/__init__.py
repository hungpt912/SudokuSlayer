from sudoku_ai.recognition.base import DigitEngine


def get_engine(name: str) -> DigitEngine:
    """Create an engine by name; see config.AVAILABLE_ENGINES."""
    if name == "tesseract":
        from sudoku_ai.recognition.tesseract_engine import TesseractEngine

        return TesseractEngine()
    raise ValueError(f"Engine '{name}' is not supported.")


__all__ = ["DigitEngine", "get_engine"]