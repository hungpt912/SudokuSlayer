"""Giao diện chung cho các bộ nhận diện chữ số."""

from abc import ABC, abstractmethod

import numpy as np


class DigitEngine(ABC):
    """Mọi bộ nhận diện phải đọc được một ô và cả bảng 81 ô."""

    @abstractmethod
    def predict(self, cell_image: np.ndarray) -> int:
        """Đọc một ô: trả về 0 nếu ô trống, ngược lại là chữ số 1..9."""

    @abstractmethod
    def predict_board(self, normalized_cells: np.ndarray) -> list[list[int]]:
        """Đọc 81 ô (81, H, W, 1) và trả về bảng 9x9, 0 là ô trống."""
