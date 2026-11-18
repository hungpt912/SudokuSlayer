"""Nhận diện chữ số trong 81 ô Sudoku."""

from sudoku_ai.config import DEFAULT_ENGINE
from sudoku_ai.recognition.base import DigitEngine


def get_engine(name: str = DEFAULT_ENGINE) -> DigitEngine:
    """Tạo bộ nhận diện theo tên. Bộ nhận diện mới cần được đăng ký tại đây."""
    if name == "cnn":
        # Import trễ để không nạp TensorFlow khi chưa cần
        from sudoku_ai.recognition.cnn_engine import CNNEngine
        return CNNEngine()
    raise ValueError(f"Engine '{name}' is not supported.")


__all__ = ["DigitEngine", "get_engine"]
