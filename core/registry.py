"""Finding modules.

Core scans a folder and imports what it finds. It holds no list, so adding a
module is adding a folder; and each import is guarded, so a module that raises
is absent and logged rather than taking the app down with it.
"""

import importlib
import importlib.util
import pkgutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config, log
from .contract import ModuleSpec
from .schema import SchemaError, validate

# Read off the module after import. Uppercase because they are declarations, not
# behaviour -- a module's announcement should be legible without running it.
FIELDS = {
    "id": "ID", "title": "TITLE", "mount": "MOUNT", "settings": "SETTINGS",
    "requires": "REQUIRES", "after": "AFTER", "before": "BEFORE",
    "stage": "STAGE", "status": "STATUS",
}


class Registry:
    def __init__(self) -> None:
        self.specs: Dict[str, ModuleSpec] = {}
        self.failed: List[Tuple[str, str]] = []      # (where, why)

    @property
    def ids(self) -> List[str]:
        return sorted(self.specs)

    def add(self, spec: ModuleSpec) -> None:
        if spec.id in self.specs:
            # Two modules answering to one id means one wins by import order,
            # and which one is nobody's decision.
            self.failed.append((spec.source, f"duplicate id {spec.id!r}, already declared by "
                                             f"{self.specs[spec.id].source}"))
            return
        self.specs[spec.id] = spec


def announcement_of(module, source: str) -> dict:
    raw = {}
    for key, const in FIELDS.items():
        if hasattr(module, const):
            raw[key] = getattr(module, const)
    return raw


def _feature_dirs(root: Path) -> List[Path]:
    """modules/<domain>/<feature>/ -- two levels, each a package."""
    if not root.is_dir():
        return []
    found = []
    for domain in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
        for feature in sorted(p for p in domain.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            if (feature / "__init__.py").is_file():
                found.append(feature)
    return found


def scan(root: Optional[Path] = None, package: str = "modules") -> Registry:
    """Import every module under `root` and collect the valid announcements."""
    root = Path(root or config.MODULES_DIR)
    registry = Registry()

    for feature in _feature_dirs(root):
        rel = feature.relative_to(root)
        dotted = f"{package}." + ".".join(rel.parts)
        try:
            module = importlib.import_module(dotted)
        except Exception as exc:                     # noqa: BLE001 -- any failure is absence
            log.failed(dotted, exc)
            registry.failed.append((dotted, f"{type(exc).__name__}: {exc}"))
            continue

        raw = announcement_of(module, dotted)
        if not raw.get("id"):
            # A folder that is not a module is not an error -- shared helpers
            # live in packages too.
            continue

        # A module's ui.js is optional, and the default is none: most modules
        # are a settings schema and no JavaScript at all.
        ui_file = feature / "ui.js"
        if ui_file.is_file():
            raw["ui"] = f"{config.UI_PREFIX}/modules/{'/'.join(rel.parts)}/ui.js"

        try:
            registry.add(validate(raw, source=dotted))
        except SchemaError as exc:
            log.note(f"{dotted} declares an invalid module: {exc}")
            registry.failed.append((dotted, str(exc)))

    return registry
