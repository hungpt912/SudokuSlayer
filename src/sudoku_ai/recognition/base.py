"""Common interface for digit recognition engines."""


from abc import ABC, abstractmethod
import numpy as np


class DigitEngine(ABC):
    """Base interface for Sudoku digit recognition engines."""

    @abstractmethod
    def predict(self, cell_image: np.ndarray) -> int:
        """
        Recognize one Sudoku cell.

        Returns:
            0 if the cell is empty, otherwise a digit from 1 to 9.
        """
        pass
