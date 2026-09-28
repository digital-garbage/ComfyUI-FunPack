"""The step wrappers get a ModelPatcher. ComfyUI publishes the packed picture+sound
shapes on its BaseModel (CFGGuider.inner_sample: inner_model.latent_shapes), not under
an inner_model, so the video span must be found from there too."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_video_span_found_from_a_model_patcher():
    import samplers

    class Base:
        latent_shapes = [torch.Size([1, 24, 3, 4, 6]), torch.Size([1, 32, 5, 2])]

    class Patcher:                      # what model_function_wrapper builders receive
        model = Base()
        model_options = {}

    n = 24 * 3 * 4 * 6 + 32 * 5 * 2
    assert samplers._video_span(Patcher(), torch.zeros(1, 1, n)) == (0, 24 * 3 * 4 * 6, (1, 24, 3, 4, 6))
    # a stale shape list from another run does not match this tensor: no guess
    assert samplers._video_span(Patcher(), torch.zeros(1, 1, n + 1)) is None
    # single stream: nothing to split
    Base.latent_shapes = [torch.Size([1, 24, 3, 4, 6])]
    assert samplers._video_span(Patcher(), torch.zeros(1, 1, 24 * 3 * 4 * 6)) is None


def test_real_model_patcher_clones_share_the_base_model():
    """The wrapper closes over a clone of a clone; CFGGuider writes latent_shapes onto
    model_patcher.model. Real ModelPatcher.clone() must share that object. Run against
    the real ComfyUI in a subprocess: this suite stubs `comfy`."""
    import os
    import subprocess
    import pytest
    root = Path(os.environ.get("COMFYUI_DIR", Path.home() / "Documents" / "ComfyUI"))
    if not (root / "comfy" / "model_patcher.py").exists():
        pytest.skip("no ComfyUI checkout to test against")
    code = ("import torch, comfy.model_patcher as mp\n"
            "r = mp.ModelPatcher(torch.nn.Linear(2, 2), torch.device('cpu'), torch.device('cpu'))\n"
            "held = r.clone().clone(); sampled = held.clone()\n"
            "sampled.model.latent_shapes = [1, 2]\n"
            "assert held.model is sampled.model and held.model.latent_shapes == [1, 2]\n")
    done = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
