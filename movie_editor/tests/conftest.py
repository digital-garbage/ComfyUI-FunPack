"""Keep collection scoped to the movie_editor sidecar.

The repo root is a ComfyUI custom-node package whose __init__ imports the full
ComfyUI/torch stack; without this guard pytest climbs into it during collection and
fails on heavy imports that aren't needed for these light, dependency-free tests.
"""
import sys
from pathlib import Path

# Make `movie_editor` importable when pytest is run from the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
