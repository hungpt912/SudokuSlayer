from sudoku_ai.recognition.cnn_model import build_cnn

def test_cnn_output_shape():
    model = build_cnn()
    predictions = model.predict(
        __import__("numpy").zeros((1, 28, 28, 1), dtype="float32"),
        verbose = 0
    )

    assert predictions.shape == (1, 10)