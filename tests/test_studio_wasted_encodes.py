"""Work whose only consumer has already declined must not be done.

Locating a phrase's token range costs one full pass of the text encoder per phrase — on
MiniMax H3 that is the same multi-billion-parameter encoder the prompt itself goes through.
The ranges feed the attn2 direction patch, which H3 refuses (no cross-attention). The refusal
used to happen after the ranges were computed, so every H3 run paid for up to eight encodes
whose result was dropped on the next line.
"""
import sys
import types

import pytest

sys.path.insert(0, ".")


@pytest.fixture
def refiner():
    from conditioning import FunPackVideoRefinerV2
    return FunPackVideoRefinerV2()


def _model(image_model):
    unet = {"image_model": image_model}
    cfg = types.SimpleNamespace(unet_config=unet)
    return types.SimpleNamespace(model=types.SimpleNamespace(model_config=cfg, diffusion_model=None))


def test_h3_does_not_read_attn2(refiner):
    assert refiner._v2_model_reads_attn2(_model("minimax_h3")) is False


def test_an_ltx_model_still_does(refiner):
    assert refiner._v2_model_reads_attn2(_model("ltxav")) is True


def test_no_model_reads_nothing(refiner):
    assert refiner._v2_model_reads_attn2(None) is False


def test_the_range_finder_and_the_patch_agree(refiner):
    """The early check and the patch's own guard must answer the same question, or the
    ranges get skipped for a model that would have used them."""
    h3 = _model("minimax_h3")
    assert refiner._v2_model_reads_attn2(h3) is False
    assert refiner._v2_apply_model_patches(h3, {}, {}, {}, [], 0.05) is None
