"""Shared project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WICHT_DIR = DATA_DIR / "wicht"

GRID_SIZE = 9
WARP_SIZE = 450                      # Rectified board side in pixels, a multiple of GRID_SIZE
CELL_SIZE = WARP_SIZE // GRID_SIZE   # Cell side length in pixels
CELL_MARGIN = 0.12                   # Fraction cropped from each cell edge to remove grid lines
DIGIT_SIZE = 28                      # Digit image size expected by the recognition engine
DIGIT_CONTENT_SIZE = 20              # Maximum glyph side inside the canvas, from 1 to DIGIT_SIZE
MAX_IMAGE_SIDE = 1200                # Downscale input images whose longest side exceeds this size

DEFAULT_ENGINE = "cnn"
AVAILABLE_ENGINES = ("cnn",)