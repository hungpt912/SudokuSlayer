"""Đọc danh sách ảnh, nhãn và tọa độ góc của bộ dữ liệu Wicht V2."""

import csv
from pathlib import Path

import numpy as np

from sudoku_ai.config import GRID_SIZE, WICHT_DIR


def get_image_paths(split="train"):
    """Trả về danh sách ảnh theo đúng chia train/test gốc của bộ dữ liệu."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'.")

    split_path = WICHT_DIR / f"v2_{split}.desc"
    image_paths = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        relative_path = line.strip()
        if not relative_path:
            continue
        image_path = WICHT_DIR / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing dataset image: {image_path}")
        image_paths.append(image_path)
    return image_paths


def read_board(image_path):
    """Đọc đề 9x9 từ file .dat cùng tên (0 là ô trống)."""
    label_path = Path(image_path).with_suffix(".dat")
    lines = label_path.read_text(encoding="utf-8").splitlines()

    # Hai dòng đầu là thông tin máy chụp, 9 dòng sau là bảng số
    board_rows = []
    for line in lines[2:]:
        row = [int(value) for value in line.split()]
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
    """Đọc 4 góc bảng đã gán nhãn; trả về None nếu ảnh không có nhãn góc."""
    image_name = Path(image_path).name
    outline_path = WICHT_DIR / "outlines_sorted.csv"

    with outline_path.open(encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if Path(row["filepath"]).name != image_name:
                continue
            corners = [[float(row[f"p{i}_x"]), float(row[f"p{i}_y"])] for i in range(1, 5)]
            return np.array(corners, dtype=np.float32)
    return None
