"""Shared test setup for the whole tree.

Kept at the root so `modules/**/tests` get the same fixtures as `core/tests`
without each one re-deriving the path to ComfyUI.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _comfyui_root():
    candidates = [os.environ.get("COMFYUI_ROOT"), str(Path.home() / "Documents" / "ComfyUI")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "comfy" / "supported_models.py").is_file():
            if candidate not in sys.path:
                sys.path.append(candidate)
            return candidate
    return None


@pytest.fixture(scope="session")
def comfyui():
    """ComfyUI's source tree on sys.path, or skip.

    Everything that touches a node schema needs it, because a schema is only
    meaningful in terms of comfy_api's own types.
    """
    root = _comfyui_root()
    if root is None:
        pytest.skip("no ComfyUI source on this machine")
    return root
