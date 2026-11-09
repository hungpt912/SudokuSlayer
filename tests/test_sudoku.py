"""Run real Sudoku images through preprocessing, CNN and backtracking.

Run directly for the official Wicht test split, or use --images for a custom directory.
Only the input image and the backtracking result image are saved per sample.
"""

import argparse
import sys
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_ai.config import DEFAULT_ENGINE, PROJECT_ROOT, WICHT_DIR
from sudoku_ai.dataset import get_image_paths, read_board
from sudoku_ai.pipeline_image.cells import split_and_normalize
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image
from sudoku_ai.recognition import get_engine
from sudoku_ai.solver import is_valid_board, solve

IMAGE_DIRECTORY = WICHT_DIR / "images"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "sudoku"


def render_result(recognized, solved):
    """Display the solution, or explicitly report None with the CNN board."""
    canvas = np.full((550, 490, 3), 255, dtype=np.uint8)
    title = "Backtracking: solved" if solved is not None else "Backtracking: None"
    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    if solved is None:
        cv2.putText(canvas, "CNN board below; invalid or unsolvable", (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1)
    board = recognized if solved is None else solved
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(canvas, (20 + i * 50, 80), (20 + i * 50, 530), (0, 0, 0), thickness)
        cv2.line(canvas, (20, 80 + i * 50), (470, 80 + i * 50), (0, 0, 0), thickness)
    for row in range(9):
        for col in range(9):
            value = int(board[row, col])
            if value:
                color = (0, 130, 0) if solved is not None and recognized[row, col] == 0 else (0, 0, 0)
                cv2.putText(canvas, str(value), (35 + col * 50, 114 + row * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return canvas


def run_sudoku(image_path, output_directory, engine):
    image = read_image(image_path)
    destination = output_directory / image_path.name
    destination.mkdir(parents=True, exist_ok=True)
    copy2(image_path, destination / f"input{image_path.suffix}")

    warped = detect_and_warp(preprocess(image))["board_gray"]
    cells = split_and_normalize(warped)
    recognized = np.asarray(engine.predict_board(cells), dtype=int)
    solved = solve(recognized)
    if solved is not None:
        assert solved.shape == (9, 9) and (solved > 0).all()
        assert is_valid_board(solved)
        givens = recognized != 0
        np.testing.assert_array_equal(solved[givens], recognized[givens])

    result_path = destination / "backtracking.png"
    if not cv2.imwrite(str(result_path), render_result(recognized, solved)):
        raise OSError(f"Cannot save image: {result_path}")
    return recognized, solved


def test_sudoku(tmp_path):
    """Exercise the real pipeline on the first official test image, when installed."""
    import pytest

    if not IMAGE_DIRECTORY.is_dir():
        pytest.skip(f"No JPG images in {IMAGE_DIRECTORY}")
    path = get_image_paths("test")[0]
    recognized, result = run_sudoku(path, tmp_path, get_engine(DEFAULT_ENGINE))
    np.testing.assert_array_equal(recognized, read_board(path))
    assert result is not None, f"Backtracking returned None for {path.name}; see {tmp_path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("train", "test"), default="test", help="Official Wicht split (default: test)")
    source.add_argument("--images", type=Path, help="Read JPG files from a custom directory")
    parser.add_argument("--limit", type=int, default=0, help="0 processes the entire selection")
    parser.add_argument("--output", type=Path, help="Output directory (default: outputs/sudoku/<split> or outputs/sudoku/custom)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be 0 or greater.")
    if args.images is None:
        paths = get_image_paths(args.split)
        name = args.split
    else:
        if not args.images.is_dir():
            parser.error(f"Image directory does not exist: {args.images}")
        paths = sorted(args.images.glob("*.jpg"))
        name = "custom"
    if not paths:
        parser.error("No JPG images found.")
    if args.limit:
        paths = paths[:args.limit]
    output = args.output or OUTPUT_DIRECTORY / name
    print(f"Selected {len(paths)} images ({name})")

    engine = get_engine(DEFAULT_ENGINE)
    solved_count = 0
    for path in paths:
        try:
            _, result = run_sudoku(path, output, engine)
            solved_count += result is not None
            print(f"{path.name}: backtracking = {'solved' if result is not None else 'None'}")
        except (ValueError, OSError, RuntimeError, AssertionError) as error:
            print(f"{path.name}: {type(error).__name__}: {error}")
    print(f"Solved: {solved_count}/{len(paths)}; output: {output.resolve()}")
    return 0 if solved_count == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
