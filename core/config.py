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

# Where projects are kept. Inside the pack rather than ComfyUI's output tree: a
# project is FunPack's own state, and clearing outputs must not take a user's
# edits with it.
PROJECTS_DIR = ROOT / "projects"

# Uploaded media (reference images, imported clips) -- FunPack's own state for
# the same reason PROJECTS_DIR is: clearing ComfyUI's output tree must not take
# something the user deliberately imported with it.
MEDIA_DIR = ROOT / "media"

# The shortcut library: trigger -> replacement text. Global, not per-project --
# a shortcut is reused across every project, not redefined in each one, so it
# lives beside PROJECTS_DIR rather than inside any one project file.
SHORTCUTS_FILE = ROOT / "shortcuts.json"
