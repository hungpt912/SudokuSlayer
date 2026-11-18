"""Kiến trúc CNN hai đầu ra và hàm giải mã kết quả."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from sudoku_ai.config import DIGIT_SIZE

MODEL_PATH = Path(__file__).with_name("witch_cnn.keras")
LEARNING_RATE = 0.0007


def build_cnn(input_shape=(DIGIT_SIZE, DIGIT_SIZE, 1)):
    """Khối trích đặc trưng dùng chung + 2 đầu ra: occupied (ô có số?) và digit (số 1..9)."""
    inputs = layers.Input(shape=input_shape)
    features = layers.Conv2D(32, 3, activation="relu")(inputs)
    features = layers.MaxPooling2D()(features)
    features = layers.Conv2D(64, 3, activation="relu")(features)
    features = layers.MaxPooling2D()(features)
    features = layers.Flatten()(features)
    features = layers.Dense(128, activation="relu")(features)
    features = layers.Dropout(0.25)(features)

    occupied = layers.Dense(1, activation="sigmoid", name="occupied")(features)
    digit = layers.Dense(9, activation="softmax", name="digit")(features)
    model = tf.keras.Model(inputs, [occupied, digit])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=["binary_crossentropy", "sparse_categorical_crossentropy"])
    return model


def decode_cells(occupied, digit, occupied_threshold=0.5):
    """Ô có occupied >= ngưỡng thì lấy chữ số có xác suất cao nhất, ngược lại là 0."""
    return np.where(np.asarray(occupied).ravel() >= occupied_threshold, np.argmax(digit, axis=1) + 1, 0)
