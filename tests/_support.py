"""Hàm hỗ trợ cho run_tests.py: đọc nhãn, chạy và đo thời gian từng bước, lưu kết quả."""

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
from time import perf_counter_ns

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sudoku_ai.dataset import get_image_paths, read_board, read_corners  # noqa: E402
from sudoku_ai.pipeline_image.cells import split_and_normalize  # noqa: E402
from sudoku_ai.pipeline_image.grid import detect_and_warp  # noqa: E402
from sudoku_ai.pipeline_image.preprocess import preprocess  # noqa: E402
from sudoku_ai.solver import solve  # noqa: E402

RESULTS = ROOT / "results"
STAGES = ("decode_ms", "preprocess_ms", "grid_ms", "cells_ms", "ocr_ms", "backtracking_ms", "total_ms")
EXPECTED_COUNTS = {"all": 76, "wicht": 40, "lexski": 30, "murphy": 6}
IOU_THRESHOLD = 0.90


# ---------- Ghi file ----------

def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_value, allow_nan=False),
                    encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save {path}")


def create_run(partial=False):
    """Xóa kết quả cũ rồi tạo lại thư mục kết quả.

    Chạy đủ 76 ảnh: ghi vào results/. Chạy một phần (--dataset, --limit): ghi vào results/partial/
    để không ghi đè kết quả đầy đủ đã commit.
    """
    directory = RESULTS / "partial" if partial else RESULTS
    # Không xóa qua link/junction để tránh xóa nhầm thư mục khác
    if directory.is_symlink() or getattr(directory, "is_junction", lambda: False)():
        raise ValueError(f"Results directory must not be a link: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    return directory


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_directory(run, sample):
    """Thư mục ảnh từng bước của một ảnh: <run>/images/<bộ dữ liệu>/<tên ảnh>/."""
    return run / "images" / sample.dataset / sample.path.name


# ---------- Dữ liệu ----------

@dataclass
class Sample:
    dataset: str
    path: Path
    expected: object       # Đề 9x9 (số gợi ý), 0 là ô trống
    corners: object        # 4 góc tham chiếu, None nếu không có
    label_source: object   # File chứa nhãn

    @property
    def name(self):
        return f"{self.dataset}/{self.path.name}"


def read_murphy_labels(path):
    """Đọc TestData.txt của Murphy: mỗi dòng 'tên_file | 81 chữ số'."""
    labels = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, board = (part.strip() for part in line.split("|", 1))
        if name in labels or len(board) != 81 or not set(board) <= set("0123456789"):
            raise ValueError(f"Invalid label at {path}:{number}")
        labels[name] = board
    return labels


def load_samples(dataset="all"):
    """Nạp 40 ảnh test Wicht, 30 ảnh Lexski và 6 ảnh Murphy cùng nhãn đề gốc."""
    samples = []
    if dataset in ("all", "wicht"):
        for path in get_image_paths("test"):
            samples.append(Sample("Wicht", path, read_board(path), read_corners(path), path.with_suffix(".dat")))

    for name in ("Lexski", "Murphy"):
        if dataset not in ("all", name.lower()):
            continue
        directory = ROOT / "data" / name
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Murphy: nhãn đọc từ TestData.txt và phải khớp manifest
        labels = read_murphy_labels(directory / "TestData.txt") if name == "Murphy" else None
        if labels is not None and set(labels) != {row["file"] for row in manifest}:
            raise ValueError("Murphy TestData.txt and manifest list different images")
        for row in manifest:
            text = labels[row["file"]] if labels is not None else row["board"]
            if text != row["board"]:
                raise ValueError(f"Label disagreement: {name}/{row['file']}")
            if len(text) != 81 or not set(text) <= set("0123456789"):
                raise ValueError(f"Invalid board: {name}/{row['file']}")
            samples.append(Sample(name, directory / row["file"],
                                  np.array(list(text), dtype=np.uint8).reshape(9, 9),
                                  np.float32(row["corners"]) if "corners" in row else None,
                                  directory / "TestData.txt" if labels is not None else manifest_path))

    for sample in samples:
        if not sample.path.is_file():
            raise FileNotFoundError(f"{sample.path} (chạy scripts/download_data.py để tải dữ liệu)")
    if len(samples) != EXPECTED_COUNTS[dataset] or len({s.name for s in samples}) != len(samples):
        raise ValueError(f"Expected {EXPECTED_COUNTS[dataset]} unique image names; found {len(samples)}")
    return samples


def audit_samples(samples):
    """Thống kê nguồn nhãn và mức trùng giữa ảnh test với ảnh train."""
    train = get_image_paths("train")
    hashes = {sha256(p) for p in train}
    layouts = {read_board(p).tobytes() for p in train}
    return {
        "images": len(samples),
        "dataset_counts": {d: sum(s.dataset == d for s in samples) for d in sorted({s.dataset for s in samples})},
        "unique_image_hashes": len({sha256(s.path) for s in samples}),
        "unique_boards": len({s.expected.astype(np.uint8).tobytes() for s in samples}),
        "wicht_train_images": len(train),
        "wicht_test_images_also_in_train": [s.name for s in samples if s.dataset == "Wicht" and sha256(s.path) in hashes],
        "wicht_test_unseen_layouts": sum(s.expected.tobytes() not in layouts for s in samples if s.dataset == "Wicht"),
        "sources": [{"image": s.name, "image_sha256": sha256(s.path),
                     "label_source": Path(s.label_source).relative_to(ROOT).as_posix(),
                     "label_sha256": sha256(s.label_source)} for s in samples],
    }


# ---------- Chấm điểm ----------

def correct_solution(answer, clues):
    """Lời giải đúng khi thỏa luật Sudoku và giữ nguyên mọi số gợi ý (kiểm tra độc lập với validator)."""
    if answer is None:
        return False
    a = np.asarray(answer)
    if a.shape != (9, 9) or not np.all((a >= 1) & (a <= 9)):
        return False
    boxes = a.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
    target = np.arange(1, 10)
    return bool(np.all(np.sort(a, axis=1) == target) and np.all(np.sort(a.T, axis=1) == target)
                and np.all(np.sort(boxes, axis=1) == target) and np.all(a[clues > 0] == clues[clues > 0]))


def reference_geometry(sample, image):
    """Khung tham chiếu: góc gán nhãn (Wicht, Lexski) hoặc viền đen của ảnh chụp màn hình (Murphy)."""
    if sample.corners is not None:
        return sample.corners, "dataset_annotation"
    if sample.dataset == "Murphy":
        dark = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 80
        xs, ys = np.flatnonzero(dark.mean(axis=0) > .8), np.flatnonzero(dark.mean(axis=1) > .8)
        if len(xs) > 1 and len(ys) > 1:
            return np.float32([[xs.min(), ys.min()], [xs.max(), ys.min()],
                               [xs.max(), ys.max()], [xs.min(), ys.max()]]), "independent_screenshot_frame"
    return None, "unavailable"


def polygon_iou(actual, expected):
    a, b = cv2.convexHull(np.float32(actual)), cv2.convexHull(np.float32(expected))
    overlap, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - overlap
    return float(overlap / union) if union else 0.0


# ---------- Chạy một ảnh ----------

def run_image(sample, data, engine=None):
    """Chạy toàn bộ luồng trên một ảnh và đo thời gian; phần chấm điểm nằm ngoài vùng đo."""
    row = {"dataset": sample.dataset, "image": sample.name, "error": None, "failed_stage": None,
           **{key: None for key in STAGES}, "pipeline_ms": None}
    trace = {}
    start = perf_counter_ns()
    current = "decode"
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode image")
        trace["image"] = image
        row["decode_ms"] = (perf_counter_ns() - start) / 1e6
        row["preprocess_ms"], row["grid_ms"] = 0.0, 0.0

        # Giống app.py: thử lần 2 có nối nét nếu lần đầu không thấy lưới
        for connect in (False, True):
            current = "preprocess"
            at = perf_counter_ns()
            preprocessed = preprocess(image, connect)
            row["preprocess_ms"] += (perf_counter_ns() - at) / 1e6
            trace["mask"] = preprocessed["detection_mask"]
            current = "grid"
            at = perf_counter_ns()
            try:
                grid = detect_and_warp(preprocessed)
            except ValueError:
                row["grid_ms"] += (perf_counter_ns() - at) / 1e6
                if connect:
                    raise
                continue
            row["grid_ms"] += (perf_counter_ns() - at) / 1e6
            trace["grid"] = grid
            break

        current = "cells"
        at = perf_counter_ns()
        trace["cells"] = split_and_normalize(grid["board_gray"])
        row["cells_ms"] = (perf_counter_ns() - at) / 1e6
        row["pipeline_ms"] = (perf_counter_ns() - start) / 1e6

        if engine is not None:
            current = "ocr"
            at = perf_counter_ns()
            recognized = engine.predict_board(trace["cells"])
            row["ocr_ms"] = (perf_counter_ns() - at) / 1e6
            trace["recognized"] = np.asarray(recognized, dtype=int)
            current = "backtracking"
            at = perf_counter_ns()
            trace["solution"] = solve(recognized)
            row["backtracking_ms"] = (perf_counter_ns() - at) / 1e6
    except (ValueError, OSError, RuntimeError) as error:
        row["error"], row["failed_stage"] = str(error), current
    row["total_ms"] = (perf_counter_ns() - start) / 1e6

    # Chấm điểm (không tính vào thời gian)
    row["detected"] = "grid" in trace
    row["grid_iou"], row["grid_correct"] = None, None
    if row["detected"]:
        corners, method = reference_geometry(sample, image)
        trace["reference_corners"] = corners
        row["grid_reference_method"] = method
        if corners is not None:
            row["grid_iou"] = polygon_iou(trace["grid"]["corners_original"], corners)
            row["grid_correct"] = row["grid_iou"] >= IOU_THRESHOLD
    recognized = trace.get("recognized")
    cells = trace.get("cells")
    row["cells_valid"] = bool(cells is not None and cells.shape == (81, 28, 28, 1)
                              and cells.dtype == np.float32 and np.isfinite(cells).all()
                              and cells.min() >= 0 and cells.max() <= 1)
    row["pipeline_correct"] = bool(row["grid_correct"] and row["cells_valid"])
    row["ocr_exact"] = bool(recognized is not None and np.array_equal(recognized, sample.expected))
    row["wrong_cells"] = int(np.count_nonzero(recognized != sample.expected)) if recognized is not None else None
    row["ocr_correct_cells"] = 81 - row["wrong_cells"] if recognized is not None else 0
    row["ocr_cell_accuracy_percent"] = row["ocr_correct_cells"] / 81 * 100
    row["solution_returned"] = trace.get("solution") is not None
    row["solution_correct"] = correct_solution(trace.get("solution"), sample.expected)
    return row, trace


# ---------- Xuất ảnh minh họa ----------

def render_board(board, clues=None):
    """Vẽ bảng 9x9; số do chương trình điền (clues == 0) tô xanh."""
    canvas = np.full((470, 470, 3), 255, np.uint8)
    if board is None:
        cv2.putText(canvas, "No solution", (70, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 180), 2)
        return canvas
    for i in range(10):
        width = 3 if i % 3 == 0 else 1
        cv2.line(canvas, (10 + i * 50, 10), (10 + i * 50, 460), (40, 40, 40), width)
        cv2.line(canvas, (10, 10 + i * 50), (460, 10 + i * 50), (40, 40, 40), width)
    for r, c in np.ndindex(9, 9):
        n = int(board[r, c])
        if n:
            color = (0, 130, 0) if clues is not None and clues[r, c] == 0 else (0, 0, 0)
            cv2.putText(canvas, str(n), (25 + c * 50, 45 + r * 50), cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2,
                        cv2.LINE_AA)
    return canvas


def export_image(directory, sample, trace, row):
    """Lưu ảnh từng bước của một ảnh: mặt nạ, khung tìm được, bảng nắn, 81 ô, đối chiếu OCR, lời giải."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "result.json", row)
    if "mask" in trace:
        save_image(directory / "1_mask.png", trace["mask"])
    if "grid" in trace:
        annotated = trace["image"].copy()
        if trace.get("reference_corners") is not None:
            cv2.polylines(annotated, [np.int32(trace["reference_corners"])], True, (255, 0, 0), 3)
        cv2.polylines(annotated, [np.rint(trace["grid"]["corners_original"]).astype(np.int32)], True, (0, 180, 0), 3)
        save_image(directory / "2_detected_grid.png", annotated)
        save_image(directory / "3_warped_board.png", trace["grid"]["board_gray"])
    if "cells" in trace:
        cells = np.rint(trace["cells"][..., 0] * 255).astype(np.uint8)
        tile = np.vstack([np.hstack(cells[i:i + 9]) for i in range(0, 81, 9)])
        save_image(directory / "4_cnn_input.png", cv2.resize(tile, (756, 756), interpolation=cv2.INTER_NEAREST))
    if "recognized" in trace:
        predicted = trace["recognized"]
        write_csv(directory / "5_ocr_comparison.csv", [
            {"row": r + 1, "column": c + 1, "expected": int(sample.expected[r, c]),
             "predicted": int(predicted[r, c]), "correct": bool(sample.expected[r, c] == predicted[r, c])}
            for r, c in np.ndindex(9, 9)])
        save_image(directory / "6_solution.png", render_board(trace.get("solution"), predicted))


# ---------- Thống kê và môi trường ----------

def statistics_ms(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean_ms": None}
    return {"n": len(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "min_ms": min(values), "max_ms": max(values), "stdev_ms": statistics.pstdev(values)}


def environment():
    return {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "opencv_threads": cv2.getNumThreads()}
