"""Keep collection scoped to the movie_editor sidecar.

The repo root is a ComfyUI custom-node package whose __init__ imports the full
ComfyUI/torch stack; without this guard pytest climbs into it during collection and
fails on heavy imports that aren't needed for these light, dependency-free tests.
"""
import sys
import types
from pathlib import Path

# Make `movie_editor` importable when pytest is run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
MOVIE_EDITOR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pytest package setup imports movie_editor as `<repo_dir>.movie_editor` and would
# execute the parent custom-node __init__ (torch/comfy/folder_paths). Stub the
# parent package so only the light movie_editor __init__ runs.
_parent_name = ROOT.name
if _parent_name not in sys.modules:
    _parent = types.ModuleType(_parent_name)
    _parent.__path__ = [str(ROOT)]
    _parent.__file__ = str(ROOT / "__init__.py")
    sys.modules[_parent_name] = _parent
