
import numpy as np

from sudoku_ai.recognition.cnn_engine import CNNEngine


def test_cnn_engine_loads_model():
    """The engine should load the trained CNN model."""
    engine = CNNEngine()

    assert engine.model is not None


def test_predict_returns_valid_digit():
    """Prediction should return an integer from 0 to 9."""
    engine = CNNEngine()

    # Create a blank 28x28 grayscale image for testing.
    cell_image = np.zeros((28, 28), dtype=np.uint8)

    result = engine.predict(cell_image)

    assert isinstance(result, int)
    assert 0 <= result <= 9


def test_predict_accepts_rgb_image():
    """The engine should also accept a 3-channel color image."""
    engine = CNNEngine()

    cell_image = np.zeros((28, 28, 3), dtype=np.uint8)

    result = engine.predict(cell_image)

    assert isinstance(result, int)
    assert 0 <= result <= 9



def test_empty_cell_returns_zero():
    """An empty normalized Sudoku cell should return 0."""
    engine = CNNEngine()

    empty_cell = np.zeros((28, 28), dtype=np.uint8)

    assert engine.predict(empty_cell) == 0
