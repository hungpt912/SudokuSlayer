"""Load Wicht V2 images and reference annotations for pipeline development."""

import csv
from pathlib import Path

import cv2
import numpy as np

from sudoku_ai.config import GRID_SIZE, WICHT_DIR


def get_image_paths(split="train"):
    """Return image paths from the original train/test split without resplitting."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")

    split_path = WICHT_DIR / f"v2_{split}.desc"
    lines = split_path.read_text(encoding="utf-8").splitlines()
    image_paths = []

    for line in lines:
        relative_path = line.strip()
        if not relative_path:
            continue

        image_path = WICHT_DIR / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing dataset image: {image_path}")

        image_paths.append(image_path)

    return image_paths


def read_board(image_path):
    """Read the 9x9 puzzle from .dat, not its solution; 0 marks an empty cell."""
    label_path = Path(image_path).with_suffix(".dat")
    lines = label_path.read_text(encoding="utf-8").splitlines()

    # The first two lines contain camera information and image metadata.
    board_rows = []
    for line in lines[2:]:
        row = []
        for value in line.split():
            row.append(int(value))

        if len(row) != GRID_SIZE:
            raise ValueError(f"Each row must contain {GRID_SIZE} cells: {label_path}")
        board_rows.append(row)

    if len(board_rows) != GRID_SIZE:
        raise ValueError(f"The board must contain {GRID_SIZE} rows: {label_path}")

    board = np.array(board_rows, dtype=np.int32)
    if np.any(board < 0) or np.any(board > 9):
        raise ValueError(f"Cell values must be in 0..9: {label_path}")

    return board.astype(np.uint8)


def read_corners(image_path):
    """Read four reference corners; return None when no annotation is available."""
    image_name = Path(image_path).name
    outline_path = WICHT_DIR / "outlines_sorted.csv"

    with outline_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if Path(row["filepath"]).name != image_name:
                continue

            corners = []
            for number in range(1, 5):
                x = float(row[f"p{number}_x"])
                y = float(row[f"p{number}_y"])
                corners.append([x, y])

            return np.array(corners, dtype=np.float32)

    return None


def load_sample(image_path):
    """Load a BGR image and evaluation labels without running the pipeline."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Use OpenCV's default image loading, including EXIF orientation handling.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return {
        "image_path": image_path,
        "image": image,
        "expected_board": read_board(image_path),
        "expected_corners": read_corners(image_path),
    }
