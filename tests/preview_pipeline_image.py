import argparse
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np

from sudoku_ai.config import GRID_SIZE, PROJECT_ROOT
from sudoku_ai.dataset import get_image_paths
from sudoku_ai.pipeline_image.cells import split_and_normalize
from sudoku_ai.pipeline_image.grid import detect_and_warp
from sudoku_ai.pipeline_image.preprocess import preprocess, read_image

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "pipeline"


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

    # Export the same normalized grayscale cells passed to the CNN.
    cells = split_and_normalize(grid_result["board_gray"])
    cell_images = np.rint(cells[..., 0] * 255).astype(np.uint8)
    cell_directory = image_directory / "cell"
    cell_directory.mkdir(parents=True, exist_ok=True)

    tiles = []
    for index, cell in enumerate(cell_images, start=1):
        save_image(cell_directory / f"cell_{index:02d}.png", cell)
        tiles.append(cv2.copyMakeBorder(cell, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=70))
    rows = [cv2.hconcat(tiles[start:start + GRID_SIZE]) for start in range(0, len(tiles), GRID_SIZE)]
    preview = cv2.resize(cv2.vconcat(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    save_image(image_directory / "cnn_input.png", preview)

    return image_directory


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--split", choices=("train", "test"), default="test")
    source.add_argument("--images", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or greater.")

    if args.images is None:
        image_paths = get_image_paths(args.split)
    else:
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
            print(f"[{index}/{len(image_paths)}] {image_path.name}: saved input, grid, and {GRID_SIZE * GRID_SIZE} CNN cells to {image_directory}")

        except Exception as error:
            print(f"[{index}/{len(image_paths)}] {image_path.name}: {type(error).__name__}: {error}")

    print(f"Processed: {processed_count}/{len(image_paths)}")
    print(f"Errors: {len(image_paths) - processed_count}")
    print(f"Output: {args.output.resolve()}")
    return 0 if processed_count == len(image_paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
