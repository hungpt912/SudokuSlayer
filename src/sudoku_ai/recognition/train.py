
from pathlib import Path

import tensorflow as tf

from sudoku_ai.recognition.cnn_model import build_cnn


def main():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (28, 28) -> (28, 28, 1)
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # Build and train the model
    model = build_cnn()

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
    )

    # Evaluate on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")

    # Save the trained model
    output_path = (
        Path(__file__).resolve().parent / "mnist_cnn.keras"
    )
    model.save(output_path)
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
