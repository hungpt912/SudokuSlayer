"""Train the Sudoku CNN on Wicht V2 and MNIST training images."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import tensorflow as tf

from sudoku_ai.config import CELL_SIZE, WARP_SIZE
from sudoku_ai.dataset import get_image_paths, read_board, read_corners
from sudoku_ai.pipeline_image.cells import crop_cell, normalize_cell, split_and_normalize
from sudoku_ai.pipeline_image.grid import detect_and_warp, order_corners
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image
from sudoku_ai.recognition.cnn_model import MODEL_PATH, build_cnn

SEED = 42
MNIST_PER_DIGIT = 1500
MNIST_WEIGHT = 0.2


def load_cells(paths):
    batches, targets = [], []
    for path in paths:
        image = preprocess(read_image(path))
        try:
            board = detect_and_warp(image)["board_gray"]
        except ValueError as error:
            # A labelled outline can provide training cells even when detection fails.
            corners = read_corners(path)
            if corners is None:
                raise ValueError(f"Cannot crop training board: {path}") from error
            gray = image["original_gray"]
            if (corners < 0).any() or (corners >= [gray.shape[1], gray.shape[0]]).any():
                raise ValueError(f"Cannot crop training board: {path}") from error
            last = WARP_SIZE - 1
            square = np.float32([[0, 0], [last, 0], [last, last], [0, last]])
            matrix = cv2.getPerspectiveTransform(order_corners(corners), square)
            board = cv2.warpPerspective(gray, matrix, (WARP_SIZE, WARP_SIZE))

        labels = read_board(path)
        cells = split_and_normalize(board)
        # Align sideways training photos to the annotated rows using the ink in each cell.
        ink = cells[:, 5:-5, 5:-5].mean(axis=(1, 2, 3)).reshape(labels.shape)
        scores = []
        for turns in range(4):
            oriented = np.rot90(ink, turns)
            scores.append(oriented[labels > 0].mean() - oriented[labels == 0].mean())
        turns = int(np.argmax(scores))
        if turns:
            board = np.ascontiguousarray(np.rot90(board, turns))
            cells = split_and_normalize(board)
        batches.append(cells)
        targets.append(labels.ravel())

        # Keep the labels and show the model a slightly defocused photograph.
        blurred = cv2.GaussianBlur(board, (5, 5), 1.0)
        batches.append(split_and_normalize(blurred))
        targets.append(labels.ravel())
    print(f"Loaded {len(paths)} Wicht training photos, {len(batches)} board versions", flush=True)
    return np.concatenate(batches), np.concatenate(targets)


def augment(cells):
    rng = np.random.default_rng(SEED)
    result = np.empty_like(cells)
    height, width = cells.shape[1:3]
    for i, cell in enumerate(cells):
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2),
                                         rng.uniform(-7, 7), rng.uniform(0.88, 1.12))
        matrix[:, 2] += rng.uniform(-2, 2, 2)
        result[i, :, :, 0] = cv2.warpAffine(cell, matrix, (width, height))
    return result


def load_mnist():
    (images, labels), _ = tf.keras.datasets.mnist.load_data()
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([rng.choice(np.flatnonzero(labels == digit), MNIST_PER_DIGIT, replace=False)
                               for digit in range(1, 10)])
    cells = []
    for image in images[selected]:
        # Place the handwritten digit on a light background at Sudoku cell scale.
        y, x = np.nonzero(image > 20)
        glyph = image[y.min():y.max() + 1, x.min():x.max() + 1]
        scale = rng.uniform(0.50, 0.78) * CELL_SIZE / max(glyph.shape)
        width = max(1, round(glyph.shape[1] * scale))
        height = max(1, round(glyph.shape[0] * scale))
        glyph = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA) / 255.0
        paper = rng.uniform(170, 250)
        cell = np.full((CELL_SIZE, CELL_SIZE), paper, dtype=np.float32)
        top = (CELL_SIZE - height) // 2 + int(rng.integers(-2, 3))
        left = (CELL_SIZE - width) // 2 + int(rng.integers(-2, 3))
        cell[top:top + height, left:left + width] -= glyph * rng.uniform(90, 160)
        cell = np.clip(cell, 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            cell = cv2.GaussianBlur(cell, (3, 3), 0.6)
        cells.append(normalize_cell(crop_cell(cell)))
    return np.asarray(cells, dtype=np.float32)[..., None] / 255.0, labels[selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, help="Fine-tune an existing Sudoku CNN")
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, help="Default: 5 for fine-tuning, 32 from scratch")
    parser.add_argument("--mnist-weight", type=float, default=MNIST_WEIGHT)
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (5 if args.initial_model else 32)
    if epochs < 1:
        parser.error("epochs must be positive.")
    if not np.isfinite(args.mnist_weight) or args.mnist_weight < 0:
        parser.error("mnist-weight must be finite and non-negative.")

    tf.keras.utils.set_random_seed(SEED)
    train, labels = load_cells(get_image_paths("train"))
    train = np.concatenate([train, augment(train)])
    labels = np.tile(labels, 2)
    handwritten, hand_labels = load_mnist()
    print(f"Training on {len(train)} Wicht cells and {len(handwritten)} MNIST digits", flush=True)
    weights = np.concatenate([np.ones(len(train), dtype=np.float32),
                              np.full(len(handwritten), args.mnist_weight, dtype=np.float32)])
    train = np.concatenate([train, handwritten])
    labels = np.concatenate([labels, hand_labels])
    occupied = (labels > 0).astype(np.float32)
    digits = np.maximum(labels.astype(int) - 1, 0)

    # Train the occupancy head on every cell and the digit head only on filled cells.
    model = build_cnn()
    if args.initial_model:
        initial = tf.keras.models.load_model(args.initial_model, compile=False)
        model.set_weights(initial.get_weights())
        model.optimizer.learning_rate.assign(0.0002)
    model.fit(train, (occupied, digits), sample_weight=(weights, weights * occupied),
              batch_size=128, epochs=epochs, verbose=2)

    args.output.parent.mkdir(exist_ok=True, parents=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".keras", delete=False) as handle:
        candidate = Path(handle.name)
    try:
        model.save(candidate)
        tf.keras.models.load_model(candidate, compile=False)
        candidate.replace(args.output)
    finally:
        candidate.unlink(missing_ok=True)
    print(f"Model saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
