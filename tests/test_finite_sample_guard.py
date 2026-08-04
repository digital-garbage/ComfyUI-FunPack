"""A chunk that comes back non-finite stops the run where it happened.

Nothing downstream notices a NaN latent on its own: the scene blend spreads it into every
already-finished scene, the VAE decodes it, and the first thing to object is ffmpeg's AAC
encoder — naming nothing that produced it, after the whole montage has been paid for. The
video half never objects at all (NaN through astype(np.uint8) is undefined but silent).
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

from samplers import FunPackLTXAVSceneChainSampler  # noqa: E402


@pytest.fixture
def node():
    return FunPackLTXAVSceneChainSampler()


def test_a_clean_latent_passes_silently(node):
    node._assert_finite_sample(torch.zeros(1, 4, 2, 8, 8))


def test_nan_in_the_video_stream_raises(node):
    x = torch.zeros(1, 4, 2, 8, 8)
    x[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_sample(x)
    assert "non-finite latent" in str(e.value)
    assert "video" in str(e.value)


def test_inf_counts_too(node):
    """+-Inf is what the AAC encoder actually names, and it is just as unrecoverable."""
    x = torch.zeros(1, 4, 2, 8, 8)
    x[0, 0, 0, 0, 0] = float("inf")
    with pytest.raises(RuntimeError):
        node._assert_finite_sample(x)


def test_the_message_says_how_many_and_which_stream(node):
    x = torch.zeros(2, 4, 2, 4, 4)
    x[0, 0, 0, 0, 0] = float("nan")
    x[0, 0, 0, 0, 1] = float("nan")
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_sample(x)
    msg = str(e.value)
    assert "2/" in msg                      # count of bad values
    assert "LoRA" in msg                     # the first thing worth testing
    assert str(x.dtype) in msg


def test_the_check_never_fails_the_run_on_its_own(node):
    """A latent shape the helper cannot read is not a reason to kill a good render."""
    node._assert_finite_sample("not a tensor")
    node._assert_finite_sample(None)


# ── inputs ────────────────────────────────────────────────────────────────────
# The output check alone cannot tell "the model computed garbage" from "the model was handed
# garbage and propagated it faithfully", and those have disjoint suspect lists.

def _sigmas(*vals):
    return torch.tensor(list(vals), dtype=torch.float32)


def test_clean_inputs_pass(node):
    node._assert_finite_inputs(torch.zeros(1, 4, 2, 8, 8), _sigmas(1.0, 0.5, 0.0))


def test_a_latent_that_arrives_nan_blames_its_producer_not_the_model(node):
    x = torch.zeros(1, 4, 2, 8, 8)
    x[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_inputs(x, _sigmas(1.0, 0.0))
    msg = str(e.value)
    assert "ALREADY" in msg
    assert "has not run yet" in msg
    # It names the model-side suspects only to RULE THEM OUT, and points at the producer.
    assert "not the checkpoint, the LoRA or the sampler" in msg
    assert "empty-latent node" in msg


def test_nan_in_the_schedule_is_named_as_the_schedule(node):
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_inputs(torch.zeros(1, 4), _sigmas(1.0, float("nan"), 0.0))
    assert "sigma schedule contains NaN" in str(e.value)


def test_an_interior_zero_sigma_is_caught_before_it_divides(node):
    """Our solvers divide by sigma, so a 0 anywhere but the end is an instant Inf."""
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_inputs(torch.zeros(1, 4), _sigmas(1.0, 0.0, 0.5, 0.0))
    assert "before its last entry" in str(e.value)


def test_a_trailing_zero_is_fine(node):
    """The last sigma is only ever a target, never a divisor — the normal schedule shape."""
    node._assert_finite_inputs(torch.zeros(1, 4), _sigmas(1.0, 0.6, 0.3, 0.0))


def test_a_repeated_sigma_is_caught(node):
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_inputs(torch.zeros(1, 4), _sigmas(1.0, 0.5, 0.5, 0.0))
    assert "strictly descending" in str(e.value)


def test_an_ascending_schedule_is_caught(node):
    with pytest.raises(RuntimeError) as e:
        node._assert_finite_inputs(torch.zeros(1, 4), _sigmas(0.3, 0.9, 0.0))
    assert "strictly descending" in str(e.value)
