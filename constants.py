"""Central configuration for Canalization Explorer."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"

APP_TITLE = "Canalization Explorer"
APP_ICON_PATH = ASSETS_DIR / "canalization_explorer_favicon.png"
CASCI_LOGO_PATH = ASSETS_DIR / "casci_canalization_explorer_logo_transparent.png"
MODEL_METADATA_PATH = PROJECT_ROOT / "model_metadata.json"
METRIC_CACHE_PATH = PROJECT_ROOT / "metric_cache.sqlite3"

RADIUS = 5.0
CANVAS_INCH = 5
NODE_PENWIDTH = 3.5
FONT_SIZE = "10"
ARROWSIZE = "0.7"
PENWIDTH_MAX = 2.5
MIN_WIDTH = 0.5
MAX_WIDTH = 4.0

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_NODES = 500
MAX_UPLOAD_EDGES = 5_000
MAX_UPLOAD_NODE_INPUTS = 18
MAX_CORRELATION_INPUTS = 16
SESSION_CACHE_MAX_ENTRIES = 24

DEFAULT_OUTLINE = "#ff9896"
SPECIAL_OUTLINE = "#ffdf0e"
ISOLATED_OUTLINE = "#888888"
POS_EDGE_COLOR = "black"
NEG_EDGE_COLOR = "black"
NEG_EDGE_STYLE = "dashed"
ZERO_EDGE_COLOR = "red"
ZERO_EDGE_STYLE = "dashed"

CELL_COLLECTIVE_DASHBOARD_URL = "https://research.cellcollective.org/research/dashboard/"
CELL_COLLECTIVE_MODEL_URL = "https://research.cellcollective.org/web/api/model/{}"
