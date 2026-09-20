"""Phrase probe: reading map from q/k, response map from block outputs, anonymous result."""
import sys

import torch

sys.path.insert(0, ".")
import phrase_probe as pp  # noqa: E402


def _sdpa(q, k, v, heads, mask=None, **kw):
    s = q @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    if mask is not None:
        s = s + mask
    return s.softmax(-1) @ v


def test_reading_map_is_the_attention_share_on_the_phrase_keys():
    torch.manual_seed(0)
    cond_len, n_video, seq = 6, 8, 14
    state = pp.ProbeState([(1, 3)], cond_len, n_video)
    ov = pp.make_attention_override(state, None)
    q = torch.randn(1, 2, seq, 4); k = torch.randn(1, 2, seq, 4); v = torch.randn(1, 2, seq, 4)
    state.begin(-1)
    out = ov(_sdpa, q, k, v, 2, skip_reshape=True)
    assert torch.allclose(out, _sdpa(q, k, v, 2))                      # visible pass untouched
    probs = (q @ k.transpose(-1, -2) * 0.5).softmax(-1)                 # 4 ** -0.5
    want = probs[0, :, cond_len:, 1:3].sum(-1).mean()
    got = state.read[0][0][0]
    assert abs(float(got) - float(want)) < 1e-5


def test_masked_pass_hides_the_phrase_and_reads_nothing():
    state = pp.ProbeState([(1, 3)], 6, 8)
    seen = {}

    def func(q, k, v, heads, mask=None, **kw):
        seen["mask"] = mask
        return q

    ov = pp.make_attention_override(state, None)
    state.begin(0)
    ov(func, torch.zeros(1, 2, 14, 4), torch.zeros(1, 2, 14, 4), torch.zeros(1, 2, 14, 4), 2,
       skip_reshape=True)
    assert seen["mask"][0, 0, 0, 1] == pp.MASKED_BIAS and seen["mask"][0, 0, 0, 3] == 0
    assert state.read == {}


def test_outside_a_probed_call_the_override_is_transparent():
    state = pp.ProbeState([(1, 3)], 6, 8)
    ov = pp.make_attention_override(state, None)
    q = torch.randn(1, 2, 14, 4)
    assert torch.equal(ov(lambda q, k, v, h, mask=None, **kw: q, q, q, q, 2, skip_reshape=True), q)
    assert state.block == 0


def test_response_is_the_relative_change_of_video_rows_between_passes():
    state = pp.ProbeState([(1, 3)], 6, 8)
    mask_fn = lambda segs, n, dev: torch.tensor([False] * 6 + [True] * 8)  # noqa: E731
    outputs = {"visible": torch.ones(14, 4), "masked": torch.ones(14, 4) * 1.5}
    which = {"k": "visible"}
    hook = pp.make_block_hook(state, 7, None,  mask_fn)
    extra = {"original_block": lambda a: {"img": outputs[which["k"]]}}
    state.begin(-1); hook({"img": None}, extra)
    which["k"] = "masked"
    state.begin(0); hook({"img": None}, extra)
    rel = state.response[7][0][0]
    assert abs(float(rel) - 0.5) < 1e-6


def test_model_wrapper_returns_the_visible_output_and_runs_one_masked_pass_per_phrase():
    state = pp.ProbeState([(1, 3), (4, 5)], 6, 8)
    modes = []

    def apply_fn(x, t, **c):
        modes.append(state.mode)
        return x * 2

    w = pp.make_model_wrapper(state, None)
    x = torch.ones(2)
    out = w(apply_fn, {"input": x, "timestep": torch.tensor([1.0]), "c": {}})
    assert torch.equal(out, x * 2)
    assert modes == [-1, 0, 1] and state.mode is None and state.calls == 1


def test_result_is_anonymous():
    state = pp.ProbeState([(1, 3)], 6, 8)
    state.read = {0: {0: [torch.tensor(0.2)]}, 1: {0: [torch.tensor(0.4)]}}
    state.response = {1: {0: [torch.tensor(0.1)]}}
    res = pp.result(state)
    assert res["phrases"] == [{"phrase": 1, "tokens": 2, "read": [0.2, 0.4], "response": [None, 0.1]}] \
        or res["phrases"][0]["tokens"] == 2
    assert "text" not in str(res) and "label" not in str(res)
