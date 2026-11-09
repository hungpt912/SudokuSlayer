"""Measure OCR against all 81 original givens before invoking any solver."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sudoku_ai.config import PROJECT_ROOT
from sudoku_ai.dataset import get_image_paths, read_board
from sudoku_ai.pipeline_image.cells import split_and_normalize
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image
from sudoku_ai.recognition.cnn_engine import CNNEngine


def benchmark(engine=None):
    engine = engine or CNNEngine()
    known = {read_board(path).tobytes() for path in get_image_paths("train")}
    counts = Counter()
    rows = []
    for path in get_image_paths("test"):
        expected = read_board(path)
        unseen = expected.tobytes() not in known
        counts["boards"] += 1
        counts["unseen_boards"] += int(unseen)
        row = {"image": path.name, "unseen_layout": unseen}
        rows.append(row)
        image = preprocess(read_image(path))
        try:
            board = detect_and_warp(image)["board_gray"]
        except ValueError as error:
            row["grid_error"] = str(error)
            continue
        actual = np.asarray(engine.predict_board(split_and_normalize(board)))
        exact = np.array_equal(actual, expected)
        given = expected > 0
        counts.update(detected=1, exact=int(exact), unseen_exact=int(unseen and exact),
                      given=int(given.sum()), given_correct=int((given & (actual == expected)).sum()),
                      missed=int((given & (actual == 0)).sum()),
                      extra=int((~given & (actual > 0)).sum()),
                      nine_to_eight=int(((expected == 9) & (actual == 8)).sum()),
                      six_to_five=int(((expected == 6) & (actual == 5)).sum()))
        row["errors"] = [{"row": int(r + 1), "col": int(c + 1),
                          "expected": int(expected[r, c]), "predicted": int(actual[r, c])}
                         for r, c in np.argwhere(actual != expected)]
    return {"counts": dict(counts), "images": rows}


def test_wicht_train_and_test_use_separate_photos():
    train = set(get_image_paths("train"))
    test = set(get_image_paths("test"))
    assert len(train) == 160
    assert len(test) == 40
    assert train.isdisjoint(test)


def test_wicht_sudoku_accuracy():
    report = benchmark()
    counts = report["counts"]
    assert counts["boards"] == 40
    assert counts["exact"] >= 35, counts
    boards = {row["image"]: row for row in report["images"]}
    for name in ("image25.jpg", "image85.jpg"):
        assert boards[name].get("errors") == [], boards[name]


if __name__ == "__main__":
    results = benchmark()
    path = PROJECT_ROOT / "outputs" / "witch_ocr_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(results["counts"])
    print(path)
