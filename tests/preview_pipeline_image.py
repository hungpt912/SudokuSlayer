import argparse
from pathlib import Path
from shutil import copy2

import cv2

from sudoku_ai.config import GRID_SIZE
from sudoku_ai.pipeline_image.cells import split_cells
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image


def save_image(image_path, image):
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Cannot save image: {image_path}")


def preview_image(image_path, output_directory):
    image = read_image(image_path)
    image_directory = output_directory / image_path.name
    image_directory.mkdir(parents=True, exist_ok=True)
    copy2(image_path, image_directory / f"input{image_path.suffix}")

    preprocessed = preprocess(image)
    preprocess_directory = image_directory / "preprocess"
    preprocess_directory.mkdir(parents=True, exist_ok=True)
    save_image(preprocess_directory / "01_grayscale.png", preprocessed["original_gray"])
    save_image(preprocess_directory / "02_resized.png", preprocessed["resized_gray"])
    save_image(preprocess_directory / "03_blurred.png", preprocessed["blurred"])
    save_image(preprocess_directory / "04_detection_mask.png", preprocessed["detection_mask"])

    grid_result = detect_and_warp(preprocessed)
    grid_directory = image_directory / "grid"
    grid_directory.mkdir(parents=True, exist_ok=True)
    detected_image = image.copy()
    corners = grid_result["corners_original"].round().astype("int32")
    cv2.polylines(detected_image, [corners.reshape(4, 1, 2)], True, (0, 255, 0), 3)

    for index, corner in enumerate(corners, start=1):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(detected_image, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(detected_image, str(index), (x + 8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    save_image(grid_directory / "01_detected_corners.png", detected_image)
    save_image(grid_directory / "02_warped_board.png", grid_result["board_gray"])

    cells = split_cells(grid_result["board_gray"])
    cell_directory = image_directory / "cell"
    cell_directory.mkdir(parents=True, exist_ok=True)

    for index, cell in enumerate(cells, start=1):
        cell_image = cv2.resize(cell, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        cell_path = cell_directory / f"cell_{index:02d}.png"

        save_image(cell_path, cell_image)

    for index in range(1, len(cells) + 1):
        old_cell_path = image_directory / f"cell_{index:02d}.png"

        if old_cell_path.is_file():
            old_cell_path.unlink()

    return image_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path(r"C:\Users\Admin\Downloads\sudoku_dataset-master\sudoku_dataset-master\images"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or greater.")

    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")

    image_paths = sorted(args.images.glob("*.jpg"))

    if not image_paths:
        parser.error("No JPG images found.")

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    processed_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            image_directory = preview_image(image_path, args.output)
            processed_count += 1
            print(f"[{index}/{len(image_paths)}] {image_path.name}: saved input, preprocess, grid, and {GRID_SIZE * GRID_SIZE} cells (6x preview) to {image_directory}")

        except Exception as error:
            print(f"[{index}/{len(image_paths)}] {image_path.name}: {type(error).__name__}: {error}")

    print(f"Processed: {processed_count}/{len(image_paths)}")
    print(f"Errors: {len(image_paths) - processed_count}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
