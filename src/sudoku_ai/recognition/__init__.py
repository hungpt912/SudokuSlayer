
from sudoku_ai.recognition.base import DigitEngine


def get_engine(name: str) -> DigitEngine:
    """Create a digit recognition engine by name."""

    if name == "cnn":
        from sudoku_ai.recognition.cnn_engine import CNNEngine
        return CNNEngine()

    if name == "tesseract":
        from sudoku_ai.recognition.tesseract_engine import TesseractEngine
        return TesseractEngine()

    raise ValueError(f"Engine '{name}' is not supported.")


__all__ = ["DigitEngine", "get_engine"]
