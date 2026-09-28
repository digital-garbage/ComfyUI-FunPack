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
