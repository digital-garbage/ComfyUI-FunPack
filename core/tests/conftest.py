import sys
from pathlib import Path

# Tests import `core.*`; make the repo root importable however pytest was invoked.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _comfyui_root():
    """ComfyUI's source tree, if this machine has one.

    Tests that need real model configs or comfy_api import it; the rest of the
    suite must keep running without it, which is why this is a lookup and not a
    hard dependency.
    """
    import os
    candidates = [os.environ.get("COMFYUI_ROOT"), str(Path.home() / "Documents" / "ComfyUI")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "comfy" / "supported_models.py").is_file():
            if candidate not in sys.path:
                sys.path.append(candidate)
            return candidate
    return None


import pytest


@pytest.fixture(scope="session")
def comfyui():
    """ComfyUI's source tree on sys.path, or skip."""
    root = _comfyui_root()
    if root is None:
        pytest.skip("no ComfyUI source on this machine")
    return root
