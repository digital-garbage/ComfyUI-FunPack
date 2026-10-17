"""Paths and constants. No logic, no imports from the rest of core."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_DIR = ROOT / "app"
MODULES_DIR = ROOT / "modules"

UI_PREFIX = "/funpack"

# Extension allowlist for the served app. modules/ is never served: a module is
# Python and a settings declaration, and draws nothing.
APP_EXTS = frozenset({".js", ".css", ".html", ".woff2", ".svg"})

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
# Categories a person made before putting anything in them (the picker offers
# them); ones a shortcut already names need no entry here.
# Shortcut revolver: its two switches and each shortcut's place in its cycle.
# Beside the library but NOT in its export -- it is state, not content.
REVOLVER_FILE = ROOT / "shortcut_revolver.json"
SHORTCUT_CATEGORIES_FILE = ROOT / "shortcut_categories.json"

# The words that cut a story into scenes (core/story.py). Global for the same
# reason the shortcut library is.
MARKERS_FILE = ROOT / "markers.json"

# Modules that failed and stay off until repaired (core/control.py). State, so beside the library.
QUARANTINE_FILE = ROOT / "quarantine.json"

# Optional features the user switched on (core/addons.py). Per machine: a new rental starts without them.
ADDONS_FILE = ROOT / "addons.json"

# LoRA training (modules/training/lora): training sets, tagger models, run history. Per machine, like media.
TRAINING_DIR = ROOT / "training"
