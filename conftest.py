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


# Put ComfyUI on the path at IMPORT time, not only when a fixture asks. A module
# that ships a modifier imports comfy at module level -- as it must, since that is
# how it runs inside ComfyUI -- so collection touches comfy before any fixture has
# had a chance to run.
COMFYUI = _comfyui_root()


@pytest.fixture(scope="session")
def comfyui():
    """ComfyUI's source tree on sys.path, or skip.

    Everything that touches a node schema needs it, because a schema is only
    meaningful in terms of comfy_api's own types.
    """
    if COMFYUI is None:
        pytest.skip("no ComfyUI source on this machine")
    return COMFYUI
