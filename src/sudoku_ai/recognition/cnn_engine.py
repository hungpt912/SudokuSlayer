
from pathlib import Path

import numpy as np
import tensorflow as tf

from sudoku_ai.recognition.base import DigitEngine


class CNNEngine(DigitEngine):
    """Recognize digits using a trained CNN model."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent / "mnist_cnn.keras"
            )

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, cell_image: np.ndarray) -> int:
        """
        Predict one Sudoku cell.

        Returns 0 for an empty cell, otherwise a digit from 1 to 9.
        """
        image = np.asarray(cell_image)

        if image.size == 0:
            raise ValueError("cell_image must not be empty")

        # Convert a single-channel image from (H, W, 1) to (H, W).
        if image.ndim == 3:
            if image.shape[-1] == 1:
                image = image[..., 0]
            elif image.shape[-1] == 3:
                image = tf.image.rgb_to_grayscale(image).numpy()[..., 0]
            else:
                raise ValueError("Expected 1 or 3 image channels")

        if image.ndim != 2:
            raise ValueError("Expected a grayscale or color cell image")

        # Empty cells produced by normalize_cell are all black.
        if np.max(image) <= 0:
            return 0

        # Resize to the model's expected input size.
        image = tf.image.resize(
            image[..., np.newaxis], (28, 28)
        ).numpy()

        image = image.astype(np.float32)

        # Accept either uint8 images [0, 255] or normalized images [0, 1].
        if image.max() > 1.0:
            image /= 255.0

        probabilities = self.model.predict(
            image[np.newaxis, ...], verbose=0
        )[0]

        predicted_digit = int(np.argmax(probabilities))

        # Sudoku givens must be in the range 1–9.
        # A model prediction of class 0 is treated as uncertain/empty.
        if predicted_digit == 0:
            return 0

        return predicted_digit



    def predict_board(self, normalized_cells: np.ndarray) -> list[list[int]]:
        """
        Recognize 81 normalized Sudoku cells.

        Args:
            normalized_cells: Array with shape (81, H, W, 1),
                              produced by split_and_normalize().

        Returns:
            A 9x9 matrix. Zero represents an empty cell.
        """
        cells = np.asarray(normalized_cells)

        if cells.ndim != 4 or cells.shape[0] != 81:
            raise ValueError(
                "Expected 81 cells with shape (81, H, W, 1)"
            )

        # Predict each cell and arrange the results into 9 rows.
        digits = [self.predict(cell) for cell in cells]

        board = [
            digits[row * 9:(row + 1) * 9]
            for row in range(9)
        ]

        return board
