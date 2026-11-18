"""Đánh giá toàn hệ thống trên 76 ảnh thật (40 Wicht test, 30 Lexski, 6 Murphy).

Đo độ chính xác và thời gian của từng bước: xử lý ảnh, nhận diện (CNN), giải (backtracking).
Mỗi ảnh chạy một lần. Chạy đủ 76 ảnh thì kết quả ghi vào results/;
chạy một phần (--dataset, --limit) thì ghi vào results/partial/.

Chạy: python tests/run_tests.py [--dataset all|wicht|lexski|murphy] [--limit N]
"""

import argparse
import random
import shutil
import sys
from time import perf_counter_ns
import traceback

from _support import (STAGES, audit_samples, correct_solution, create_run, environment, export_image,
                      image_directory, load_samples, run_image, sha256, statistics_ms, write_csv, write_json)
from sudoku_ai.solver import solve

IOU_CRITERION = "Grid IoU >= 0.90 and 81 valid normalized cells"
TIMES = (*STAGES, "pipeline_ms")
# Ảnh minh họa được chép sang results/examples/ (một ảnh đọc đúng, một ảnh đọc sai)
EXAMPLES = ("Wicht/image114.jpg", "Lexski/64580592jqzc1.webp")
EXAMPLE_FILES = ("2_detected_grid.png", "3_warped_board.png", "4_cnn_input.png",
                 "5_ocr_comparison.csv", "6_solution.png", "result.json")


def run_reference_solver(samples, directory):
    """Đo riêng bộ giải trên đề gốc (nhãn đúng), tách khỏi ảnh hưởng của lỗi OCR."""
    rows, answers = [], {}
    solve(samples[0].expected)  # Chạy khởi động, không tính thời gian
    order = list(samples)
    random.Random(43).shuffle(order)
    for sample in order:
        at = perf_counter_ns()
        answer = solve(sample.expected)
        elapsed = (perf_counter_ns() - at) / 1e6
        answers[sample.name] = answer
        rows.append({"dataset": sample.dataset, "image": sample.name, "solve_ms": elapsed,
                     "solution_returned": answer is not None,
                     "solvable": correct_solution(answer, sample.expected)})
    rows.sort(key=lambda r: r["image"])
    write_csv(directory / "per_image.csv", rows)

    solved = sum(r["solvable"] for r in rows)
    summary = {"images": len(rows), "solved": solved, "solution_rate_percent": solved / len(rows) * 100,
               "no_solution": [r["image"] for r in rows if not r["solvable"]],
               "invalid_solutions": [r["image"] for r in rows if r["solution_returned"] and not r["solvable"]],
               "timing": statistics_ms([r["solve_ms"] for r in rows])}
    write_json(directory / "summary.json", summary)
    print(f"Reference backtracking: {solved}/{len(rows)} boards solved", flush=True)
    return summary, answers


def run_images(samples, run, engine, reference):
    """Chạy toàn bộ luồng trên từng ảnh, đo thời gian, rồi xuất ảnh minh họa ngoài vùng đo."""
    data = {s.name: s.path.read_bytes() for s in samples}   # Đọc sẵn để không tính thời gian đọc đĩa

    # Chạy khởi động (không tính): một ảnh đầu tiên và một ảnh Lexski
    warmup = [samples[0]]
    if any(s.dataset == "Lexski" for s in samples) and samples[0].dataset != "Lexski":
        warmup.append(next(s for s in samples if s.dataset == "Lexski"))
    for sample in warmup:
        run_image(sample, data[sample.name], engine)

    rows = []
    order = list(samples)
    random.Random(42).shuffle(order)
    with (run / "run.log").open("w", encoding="utf-8") as log:
        for index, sample in enumerate(order, 1):
            row, trace = run_image(sample, data[sample.name], engine)
            row["reference_solvable"] = correct_solution(reference[sample.name], sample.expected)
            rows.append(row)
            export_image(image_directory(run, sample), sample, trace, row)
            log.write(f"{sample.name}: error={row['error']} OCR={row['ocr_exact']} "
                      f"solution={row['solution_correct']} {row['total_ms']:.2f} ms\n")
            if index % 10 == 0 or index == len(order):
                print(f"Images: {index}/{len(order)}", flush=True)
    rows.sort(key=lambda r: r["image"])
    write_csv(run / "per_image.csv", rows)

    def summarize(dataset=None):
        part = [r for r in rows if dataset is None or r["dataset"] == dataset]
        images = len(part)

        def accuracy(count, criterion):
            return {"correct": count, "total": images, "percent": count / images * 100, "criterion": criterion}

        summary = {"images": images, "detected": sum(r["detected"] for r in part),
                   "grid_reference_counts": {method: sum(r.get("grid_reference_method") == method for r in part)
                                             for method in ("dataset_annotation", "independent_screenshot_frame",
                                                            "unavailable")},
                   "grid_correct": sum(r["grid_correct"] is True for r in part),
                   "failed_images": [r["image"] for r in part if r["error"]],
                   "timing": {key: statistics_ms([r[key] for r in part]) for key in TIMES},
                   "ocr_exact": sum(r["ocr_exact"] for r in part),
                   "wrong_cells": sum(r["wrong_cells"] or 0 for r in part),
                   "reference_solvable": sum(r["reference_solvable"] for r in part),
                   "solution_returned": sum(r["solution_returned"] for r in part),
                   "solution_correct": sum(r["solution_correct"] for r in part)}
        cells_correct = sum(r["ocr_correct_cells"] for r in part)
        summary["accuracy"] = {
            "pipeline": accuracy(sum(r["pipeline_correct"] for r in part), IOU_CRITERION),
            "cnn": accuracy(summary["ocr_exact"], "All 81 cells equal the original givens, including blanks"),
            "backtracking_on_ocr": accuracy(summary["solution_correct"],
                                            "Solution is valid and preserves original givens; affected by OCR errors"),
            "full": accuracy(summary["solution_correct"], "Correct automatic solution from the input image"),
            "cnn_cells": {"correct": cells_correct, "total": images * 81,
                          "percent": cells_correct / (images * 81) * 100}}
        return summary

    return {"all": summarize(), "datasets": {d: summarize(d) for d in sorted({s.dataset for s in samples})}}


def write_report(run, report):
    """Ghi summary.json, stage_summary.csv và bảng tóm tắt summary.md."""
    write_json(run / "summary.json", report)
    stage_times = {"pipeline": "pipeline_ms", "cnn": "ocr_ms", "backtracking_on_ocr": "backtracking_ms",
                   "full": "total_ms"}
    stage_names = {"pipeline": "Xử lý ảnh (tìm lưới, tách ô)", "cnn": "Nhận diện CNN (đúng cả 81 ô)",
                   "backtracking_on_ocr": "Giải trên kết quả OCR", "full": "Toàn hệ thống"}
    results = report["all"]
    stage_rows = [{"stage": name, **results["accuracy"][name], "mean_ms": results["timing"][time]["mean_ms"]}
                  for name, time in stage_times.items()]
    write_csv(run / "stage_summary.csv", stage_rows)

    cells = results["accuracy"]["cnn_cells"]
    reference = report["backtracking_reference"]
    lines = ["# Kết quả đánh giá toàn hệ thống", "", f"Số ảnh: {results['images']}.", "",
             "| Giai đoạn | Đúng/tổng | Độ chính xác (%) | Thời gian TB (ms) |", "|---|---:|---:|---:|"]
    lines += [f"| {stage_names[r['stage']]} | {r['correct']}/{r['total']} | {r['percent']:.2f} | {r['mean_ms']:.3f} |"
              for r in stage_rows]
    lines += ["", f"CNN theo từng ô (gồm ô trống): {cells['correct']}/{cells['total']} = {cells['percent']:.2f}%.", "",
              "| Bộ dữ liệu | Số ảnh | OCR đúng cả bảng | Giải đúng tự động |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {d['images']} | {d['ocr_exact']}/{d['images']} | {d['solution_correct']}/{d['images']} |"
              for name, d in report["datasets"].items()]
    lines += ["", "| Bước | TB (ms) | Trung vị (ms) |", "|---|---:|---:|"]
    lines += [f"| {key} | {value['mean_ms']:.3f} | {value['median_ms']:.3f} |"
              for key, value in results["timing"].items() if value["mean_ms"] is not None]
    lines += ["", f"Bộ giải trên đề gốc: {reference['solved']}/{reference['images']} bảng có lời giải; "
                  f"TB {reference['timing']['mean_ms']:.3f} ms, trung vị {reference['timing']['median_ms']:.3f} ms.",
              "Bảng vô nghiệm: " + (", ".join(reference["no_solution"]) or "không có"), "",
              "Ghi chú: thời gian không gồm đọc/ghi đĩa, nạp mô hình và chạy khởi động. "
              "Xử lý ảnh đúng khi IoU khung >= 0.90 và đủ 81 ô hợp lệ. "
              "Lời giải đúng khi thỏa luật Sudoku và giữ nguyên số gợi ý gốc."]
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_examples(run, samples):
    """Chép ảnh từng bước của các ảnh trong EXAMPLES sang <run>/examples/<bộ dữ liệu>_<tên ảnh>/."""
    for sample in samples:
        if sample.name not in EXAMPLES:
            continue
        target = run / "examples" / sample.name.replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)
        for name in EXAMPLE_FILES:
            source = image_directory(run, sample) / name
            if source.is_file():
                shutil.copy2(source, target / name)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("all", "wicht", "lexski", "murphy"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Chỉ chạy N ảnh đầu (0 = tất cả)")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit cannot be negative")

    run = create_run(partial=args.dataset != "all" or args.limit > 0)
    print(f"Run folder: {run}", flush=True)
    phase = "load_references"
    try:
        samples = load_samples(args.dataset)
        if args.limit:
            samples = samples[:args.limit]
        write_json(run / "audit.json", audit_samples(samples))
        config = {**environment(), "dataset": args.dataset, "images": len(samples),
                  "included": "decode preloaded bytes, preprocessing, grid/warp, cells, OCR, solve on OCR",
                  "excluded": "disk reads/writes, model loading, warm-up, scoring, image export, UI",
                  "grid_iou_threshold": 0.90,
                  "label_sources": "Wicht .dat; Lexski manifest; Murphy TestData.txt cross-checked with manifest",
                  "grid_reference": "Annotated corners for Wicht/Lexski; black screenshot frame for Murphy"}

        phase = "backtracking_reference"
        directory = run / "backtracking_reference"
        directory.mkdir()
        summary, reference = run_reference_solver(samples, directory)
        report = {"backtracking_reference": summary}

        phase = "load_model"
        import tensorflow as tf
        from sudoku_ai.recognition import get_engine
        from sudoku_ai.recognition.cnn_model import MODEL_PATH
        at = perf_counter_ns()
        engine = get_engine()
        config.update({"tensorflow": tf.__version__, "model": MODEL_PATH.name, "model_sha256": sha256(MODEL_PATH),
                       "model_load_ms_excluded": (perf_counter_ns() - at) / 1e6})
        write_json(run / "config.json", config)

        phase = "images"
        report.update(run_images(samples, run, engine, reference))
        report["status"] = "completed"
        write_report(run, report)
        copy_examples(run, samples)
        print((run / "summary.md").read_text(encoding="utf-8"), flush=True)
        print(f"Saved: {run}", flush=True)
        failed = bool(report["all"]["failed_images"] or report["backtracking_reference"]["invalid_solutions"])
        return int(failed)
    except Exception as error:
        write_json(run / "error.json", {"phase": phase, "error": str(error), "traceback": traceback.format_exc()})
        print(f"Failed during {phase}: {error}. Details: {run / 'error.json'}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
