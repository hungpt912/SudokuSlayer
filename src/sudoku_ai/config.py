"""Shared project constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WICHT_DIR = DATA_DIR / "wicht"

GRID_SIZE = 9
WARP_SIZE = 450                      # Rectified board side in pixels, a multiple of GRID_SIZE
CELL_SIZE = WARP_SIZE // GRID_SIZE   # Cell side length in pixels
CELL_MARGIN = 0.12                   # Maximum fraction inspected for grid strokes at each cell edge
DIGIT_SIZE = 28                      # Digit image size expected by the recognition engine
DIGIT_RECHECK_CONFIDENCE = 0.65      # Re-read occupied cells below this digit probability
MAX_IMAGE_SIDE = 1200                # Downscale input images whose longest side exceeds this size

DEFAULT_ENGINE = "cnn"
AVAILABLE_ENGINES = ("cnn",)
