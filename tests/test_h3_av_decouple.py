"""H3 attention decoupling (_install_h3_av_decouple): damps cross-modal video<->audio
attention via comfy's optimized_attention_override extension point, without ever
materializing an S x S mask. Exercised at the install/override level with fake tensors --
no real attention backend or GPU needed."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


# rows: 0-1 video, 2 text, 3-4 audio, 5 video -- exercises a modality repeating
# non-contiguously, same as a real packed layout can produce.
_MOD_SEGMENTS = [(0, 2, 0), (2, 3, 1), (3, 5, 2), (5, 6, 0)]
_VIDEO_IDX = [0, 1, 5]
_AUDIO_IDX = [3, 4]
_OTHER_IDX = [2]


def _fire_capture_hook(patched, seq_len=6, mod_segments=_MOD_SEGMENTS):
    """Runs block 0's hook once, the way the real DiT forward loop would, so the override
    has a mod_segments layout to work with."""
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    hook = dit[("double_block", 0)]
    img = torch.ones(seq_len, 3)
    args = {"img": img, "mod_segments": mod_segments}
    extra = {"original_block": lambda a: {"img": a["img"]}}
    hook(args, extra)


def _qkv(seq_len=6, heads=1, head_dim=2):
    q = torch.arange(seq_len * head_dim, dtype=torch.float32).view(1, heads, seq_len, head_dim).clone()
    k = torch.ones(1, heads, seq_len, head_dim) * 2.0
    v = torch.zeros(1, heads, seq_len, head_dim)
    return q, k, v


def _fake_func_factory(calls):
    def fake_func(q, k, v, heads, mask=None, skip_reshape=True, **kw):
        # value encodes the bias so assertions can tell which group produced which output
        # without needing a real attention computation.
        marker = float(mask.sum()) if mask is not None else 0.0
        calls.append({"q_len": q.shape[-2], "mask": mask, "marker": marker})
        return torch.full((q.shape[0], q.shape[-2], heads * q.shape[-1]), marker)
    return fake_func


def test_zero_strength_is_a_true_noop():
    model = _FakeModel()
    node = S()
    assert node._install_h3_av_decouple(model, 0.0) is model
    assert node._install_h3_av_decouple(model, "") is model
    assert node._install_h3_av_decouple(model, "not-a-number") is model


def test_enabled_installs_override_and_block0_hook():
    node = S()
    patched = node._install_h3_av_decouple(_FakeModel(), 0.5)
    to = patched.model_options["transformer_options"]
    assert callable(to["optimized_attention_override"])
    assert ("double_block", 0) in to["patches_replace"]["dit"]


def test_before_mod_segments_captured_override_is_a_passthrough():
    node = S()
    patched = node._install_h3_av_decouple(_FakeModel(), 0.5)
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    calls = []
    q, k, v = _qkv()
    out = override(_fake_func_factory(calls), q, k, v, heads=1, mask=None, skip_reshape=True)
    assert len(calls) == 1, "no mod_segments yet -> single unmodified call, not split"
    assert calls[0]["mask"] is None
    assert torch.allclose(out, torch.zeros_like(out))


def test_an_explicit_mask_is_never_overridden():
    node = S()
    patched = node._install_h3_av_decouple(_FakeModel(), 0.5)
    _fire_capture_hook(patched)
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    calls = []
    q, k, v = _qkv()
    real_mask = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    override(_fake_func_factory(calls), q, k, v, heads=1, mask=real_mask, skip_reshape=True)
    assert len(calls) == 1 and calls[0]["mask"] is real_mask


def test_splits_into_three_groups_with_a_bias_penalizing_the_other_modality():
    node = S()
    patched = node._install_h3_av_decouple(_FakeModel(), 0.5)
    _fire_capture_hook(patched)
    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    calls = []
    q, k, v = _qkv(seq_len=6, heads=1, head_dim=2)
    out = override(_fake_func_factory(calls), q, k, v, heads=1, mask=None, skip_reshape=True)

    assert len(calls) == 3, "video-query, audio-query and other-query passes"
    by_len = {c["q_len"]: c for c in calls}
    assert set(by_len) == {len(_VIDEO_IDX), len(_AUDIO_IDX), len(_OTHER_IDX)}

    bias_val = -0.5 * 12.0  # strength * _BIAS_SCALE

    audio_call = by_len[len(_AUDIO_IDX)]
    audio_bias = audio_call["mask"].view(-1)
    assert torch.allclose(audio_bias[_VIDEO_IDX], torch.full((len(_VIDEO_IDX),), bias_val))
    assert torch.allclose(audio_bias[_AUDIO_IDX + _OTHER_IDX],
                          torch.zeros(len(_AUDIO_IDX) + len(_OTHER_IDX)))

    video_call = by_len[len(_VIDEO_IDX)]
    video_bias = video_call["mask"].view(-1)
    assert torch.allclose(video_bias[_AUDIO_IDX], torch.full((len(_AUDIO_IDX),), bias_val))
    assert torch.allclose(video_bias[_VIDEO_IDX + _OTHER_IDX],
                          torch.zeros(len(_VIDEO_IDX) + len(_OTHER_IDX)))

    other_call = by_len[len(_OTHER_IDX)]
    assert other_call["mask"] is None, "text/other rows ride the ordinary, unbiased pass"

    # scatter landed each group's output back at its own rows (marker = mask.sum())
    assert torch.allclose(out[0, _AUDIO_IDX], torch.full((len(_AUDIO_IDX), 2), audio_call["marker"]))
    assert torch.allclose(out[0, _VIDEO_IDX], torch.full((len(_VIDEO_IDX), 2), video_call["marker"]))
    assert torch.allclose(out[0, _OTHER_IDX], torch.zeros(len(_OTHER_IDX), 2))


def test_chains_through_an_already_installed_block0_hook():
    """If h3_repr_steering (or anything else) already hooked block 0, av_decouple must not
    silently replace it -- it should still see mod_segments AND still get the other hook's
    modified output."""
    node = S()
    model = _FakeModel()
    to = {"patches_replace": {"dit": {
        ("double_block", 0): lambda args, extra: {"img": args["img"] * 0.0}
    }}}
    model.model_options["transformer_options"] = to
    patched = node._install_h3_av_decouple(model, 0.5)
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    img = torch.ones(6, 3)
    out = dit[("double_block", 0)]({"img": img, "mod_segments": _MOD_SEGMENTS},
                                   {"original_block": lambda a: {"img": a["img"]}})
    assert torch.allclose(out["img"], torch.zeros(6, 3)), "existing hook's effect must survive"

    override = patched.model_options["transformer_options"]["optimized_attention_override"]
    calls = []
    q, k, v = _qkv()
    override(_fake_func_factory(calls), q, k, v, heads=1, mask=None, skip_reshape=True)
    assert len(calls) == 3, "mod_segments was captured despite chaining through the other hook"


if __name__ == "__main__":
    test_zero_strength_is_a_true_noop()
    test_enabled_installs_override_and_block0_hook()
    print("ok (run via pytest for the rest)")
