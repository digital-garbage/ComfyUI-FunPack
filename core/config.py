"""Paths and constants. No logic, no imports from the rest of core."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_DIR = ROOT / "app"
MODULES_DIR = ROOT / "modules"

UI_PREFIX = "/funpack"

# Extension allowlists per served root. These are rule 1 ("a module never styles
# anything") enforced at the transport layer: a .css inside modules/ is a 404, so
# a module cannot ship a stylesheet even if someone writes one.
APP_EXTS = frozenset({".js", ".css", ".html", ".woff2", ".svg"})
MODULE_EXTS = frozenset({".js"})
