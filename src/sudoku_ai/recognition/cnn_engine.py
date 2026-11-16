import cv2
import numpy as np
import tensorflow as tf

from sudoku_ai.config import DIGIT_RECHECK_CONFIDENCE, DIGIT_SIZE, GRID_SIZE
from sudoku_ai.recognition.base import DigitEngine
from sudoku_ai.recognition.cnn_model import MODEL_PATH, decode_cells


class CNNEngine(DigitEngine):
    """Read digits from normalized grayscale Sudoku cells."""

    def __init__(self, model_path=None):
        self.model = tf.keras.models.load_model(model_path or MODEL_PATH, compile=False)
        self.occupied_threshold = 0.5

    def _prepare_cells(self, cells):
        cells = np.asarray(cells)
        if cells.ndim != 4 or cells.size == 0 or cells.shape[-1] not in (1, 3):
            raise ValueError("Expected non-empty grayscale or RGB cells.")

        cells = cells.astype(np.float32)
        if cells.max() > 1:
            cells /= 255.0
        if cells.shape[-1] == 3:
            cells = tf.image.rgb_to_grayscale(cells)
        if cells.shape[1:3] != (DIGIT_SIZE, DIGIT_SIZE):
            cells = tf.image.resize(cells, (DIGIT_SIZE, DIGIT_SIZE))

        return np.asarray(cells)

    def _read_digits(self, cells, occupied, digits):
        digits = np.asarray(digits)
        uncertain = (np.asarray(occupied).ravel() >= self.occupied_threshold) & (digits.max(axis=1) < DIGIT_RECHECK_CONFIDENCE)
        if uncertain.any():
            # Boost soft edges while leaving headroom for bright strokes.
            sharp = []
            for cell in cells[uncertain, :, :, 0]:
                detail = cell - cv2.GaussianBlur(cell, (0, 0), 2)
                sharp.append(np.clip(0.5 * (cell + 3 * detail), 0, 1))

            # Average five nearby alignments so a single crop offset cannot decide the digit.
            padded = np.pad(sharp, ((0, 0), (2, 2), (2, 2)))
            offsets = [(2, 2), (0, 2), (4, 2), (2, 0), (2, 4)]
            views = [padded[:, y:y + DIGIT_SIZE, x:x + DIGIT_SIZE] for y, x in offsets]
            _, scores = self.model(np.concatenate(views)[..., None], training=False)
            digits = digits.copy()
            digits[uncertain] = np.asarray(scores).reshape(len(views), len(sharp), 9).mean(axis=0)
        return decode_cells(occupied, digits, self.occupied_threshold)

    def predict(self, cell_image):
        image = np.asarray(cell_image)
        if image.ndim == 2:
            image = image[..., None]
        cells = self._prepare_cells(image[None])
        occupied, digits = self.model(cells, training=False)
        return int(self._read_digits(cells, occupied, digits)[0])

    def _board_scores(self, cells):
        # Read the board in the orientation with the strongest digit evidence.
        rotations = [np.rot90(cells, turns, axes=(1, 2)) for turns in range(4)]
        occupied, digits = self.model(np.concatenate(rotations), training=False)
        visible = np.asarray(occupied).reshape(4, -1) >= self.occupied_threshold
        confidence = np.max(digits, axis=1).reshape(4, -1)
        mean_score = (confidence * visible).sum(axis=1) / np.maximum(1, visible.sum(axis=1))
        turns = int(np.argmax(mean_score))
        occupied = np.asarray(occupied).reshape(4, -1)[turns]
        digits = np.asarray(digits).reshape(4, -1, 9)[turns]
        return rotations[turns], occupied, digits, turns

    def predict_board(self, normalized_cells):
        cells = np.asarray(normalized_cells)
        if cells.ndim != 4 or len(cells) != GRID_SIZE ** 2:
            raise ValueError("Expected 81 cells with shape (81, H, W, 1).")
        cells = self._prepare_cells(cells)
        cells, occupied, digits, turns = self._board_scores(cells)
        board = self._read_digits(cells, occupied, digits).reshape(GRID_SIZE, GRID_SIZE)
        return np.rot90(board, turns).tolist()
