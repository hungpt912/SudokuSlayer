
import numpy as np
import tensorflow as tf

from sudoku_ai.recognition.cnn_engine import CNNEngine


def test_recognizes_real_mnist_digits():
    # Load real images from the MNIST test dataset
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Select 100 real images
    x_sample = x_test[:100]
    y_sample = y_test[:100]

    engine = CNNEngine()

    correct = 0

    for image, expected in zip(x_sample, y_sample):
        predicted = engine.predict(image)

        if predicted == int(expected):
            correct += 1

    accuracy = correct / len(y_sample)

    print(f"\nAccuracy on 100 MNIST images: {accuracy:.2%}")

    # The model should recognize most images correctly
    assert accuracy >= 0.90
