"""H3's picture, found in both shapes it reaches features in."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def test_packed_model_output_splits_and_rebuilds_exactly():
    from conftest import packed_av, unpacked
    from modules.models.minimax_h3.streams import video_stream
    video, audio = torch.randn(2, 4, 3, 6, 6), torch.randn(2, 8, 5)
    x, shapes = packed_av(video, audio)
    got, rebuild = video_stream(x, {"latent_shapes": shapes})
    assert torch.equal(got, video)
    out = rebuild(got * 2)
    assert torch.equal(unpacked(out, shapes)[0], video * 2)
    assert torch.equal(unpacked(out, shapes)[1], audio)


def test_packed_without_its_shapes_is_not_guessed():
    from conftest import packed_av
    from modules.models.minimax_h3.streams import video_stream
    x, _ = packed_av(torch.randn(1, 4, 3, 6, 6), torch.randn(1, 8, 5))
    assert video_stream(x, {}) is None


def test_nested_still_works():
    from comfy.nested_tensor import NestedTensor
    from modules.models.minimax_h3.streams import video_stream
    video, audio = torch.randn(1, 4, 3, 6, 6), torch.randn(1, 8, 5)
    got, rebuild = video_stream(NestedTensor([video, audio]))
    assert got is video and rebuild(video).tensors[1] is audio
