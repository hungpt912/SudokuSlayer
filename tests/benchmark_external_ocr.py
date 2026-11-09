"""Score the Sudoku images in the Lexski and Murphy datasets."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_ai.config import DATA_DIR, PROJECT_ROOT
from sudoku_ai.pipeline_image.cells import split_and_normalize
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image
from sudoku_ai.recognition.cnn_engine import CNNEngine


def benchmark(engine, directory):
    samples = []
    for name in ("Lexski", "Murphy"):
        dataset = directory / name
        labels = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        samples.extend((dataset / label["file"], label) for label in labels)

    counts = defaultdict(Counter)
    results = []
    for path, sample in samples:
        expected = np.array(list(sample["board"]), dtype=int).reshape(9, 9)
        count = counts[path.parent.name.lower()]
        count["images"] += 1
        result = {"file": path.relative_to(directory).as_posix()}
        results.append(result)
        image = preprocess(read_image(path))
        try:
            grid = detect_and_warp(image)
        except ValueError as error:
            result["grid_error"] = str(error)
            continue
        if "corners" in sample:
            actual_corners = cv2.convexHull(grid["corners_original"])
            reference = cv2.convexHull(np.float32(sample["corners"]))
            overlap, _ = cv2.intersectConvexConvex(actual_corners, reference)
            union = cv2.contourArea(actual_corners) + cv2.contourArea(reference) - overlap
            result["grid_iou"] = float(overlap / union)
            count["annotated_grids"] += 1
            count["correct_grids"] += int(result["grid_iou"] >= 0.90)
        board = grid["board_gray"]
        actual = np.array(engine.predict_board(split_and_normalize(board)))
        wrong = np.argwhere(actual != expected)
        count.update(detected=1, exact=int(len(wrong) == 0), wrong_cells=len(wrong))
        result["errors"] = [{"row": int(r + 1), "column": int(c + 1),
                             "expected": int(expected[r, c]), "actual": int(actual[r, c])}
                            for r, c in wrong]
    return {"counts": dict(counts), "images": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DATA_DIR,
                        help="Directory containing Lexski and Murphy (default: data)")
    args = parser.parse_args()
    try:
        report = benchmark(CNNEngine(), args.directory)
    except FileNotFoundError as error:
        parser.error(str(error))
    path = PROJECT_ROOT / "outputs" / "external_ocr_test.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]))
    print(path)
