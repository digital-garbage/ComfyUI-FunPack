"""H3 query steering (_install_h3_q_steering): REINS-style liked-minus-disliked direction
captured and applied on Q instead of the block's hidden state. Exercised at the install/
override level with fake tensors -- no real attention backend or GPU needed.

Covers the two places a silent bug would hide: the (heads, head_dim) <-> flat-vector
reshape used to reuse h3_repr_steering.capture()/direction() on Q, and that capture happens
on the PRE-injection Q (same ordering rule h3_repr_steering itself follows)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, ".")
import _comfy_stubs  # noqa: E402, F401
import h3_repr_steering as _rs  # noqa: E402
from samplers import FunPackLTXAVSceneChainSampler as S  # noqa: E402


class _FakeModel:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _FakeModel()
        m.model_options = dict(self.model_options)
        return m


# 4 rows: 0 and 2 are video, 1 is text, 3 is audio -- non-contiguous video, same as
# test_h3_av_decouple's layout.
_MOD_SEGMENTS = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 2)]


def _run_attention_inside_block(patched, block, attn_fn, mod_segments=_MOD_SEGMENTS, seq_len=4):
    """The active-block flag _install_h3_q_steering uses is only true WHILE the block's own
    forward is running (attention happens inside it) -- so `attn_fn` (whatever calls the
    optimized_attention_override) must run from inside `original_block`, not after the hook
    returns, or the override sees no active block at all (a real trap this test caught once)."""
    dit = patched.model_options["transformer_options"]["patches_replace"]["dit"]
    hook = dit[("double_block", block)]
    result = {}

    def _original_block(a):
        result["out"] = attn_fn()
        return {"img": a["img"]}

    args = {"img": torch.ones(seq_len, 3), "mod_segments": mod_segments}
    hook(args, {"original_block": _original_block})
    return result["out"]


def _install(monkeypatch, direction_vec, strength=1.0, block=0):
    """direction_vec: flat (heads*head_dim,) tensor, or None for 'not enough data yet'."""
    def _fake_direction(refinement_key, block=None, kind="repr_steer"):
        assert kind == "q_steer"
        if direction_vec is None:
            return None, 0, 0
        return direction_vec, 5, 5
    monkeypatch.setattr(_rs, "direction", _fake_direction)
    capture_holder = [{}]
    patched = S()._install_h3_q_steering(
        _FakeModel(), "fake_key", strength, capture_holder, steer_block=str(block))
    return patched, capture_holder


def _override_fn(patched):
    return patched.model_options["transformer_options"]["optimized_attention_override"]


def test_capture_and_inject_at_named_block(monkeypatch):
    # heads=2, head_dim=1, values chosen so row_norm and the injected shift are round numbers.
    q = torch.zeros(1, 2, 4, 1)
    q[0, 0, 0, 0] = 3.0
    q[0, 0, 2, 0] = 3.0
    q[0, 1, 0, 0] = 1.0
    q[0, 1, 2, 0] = 1.0
    k = torch.zeros(1, 2, 4, 1)
    v = torch.zeros(1, 2, 4, 1)

    patched, capture_holder = _install(monkeypatch, torch.tensor([1.0, -1.0]), strength=1.0)

    calls = []

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        calls.append(q_.clone())
        return torch.zeros(q_.shape[0], q_.shape[-2], heads_ * q_.shape[-1])

    _run_attention_inside_block(
        patched, block=0,
        attn_fn=lambda: _override_fn(patched)(fake_func, q, k, v, 2, skip_reshape=True))

    # Captured BEFORE injection: q_flat[video_rows].mean(dim=0) == [3.0, 1.0] (head0, head1).
    assert torch.allclose(capture_holder[0][0], torch.tensor([3.0, 1.0]))

    # row_norm = mean(|3|, |3|, |1|, |1|) = 2.0; shift = direction * strength * row_norm =
    # [1, -1] * 1.0 * 2.0 = [+2, -2], applied identically to both masked rows, K/V untouched.
    q_after = calls[0]
    assert torch.allclose(q_after[0, 0, 0, 0], torch.tensor(5.0))   # 3 + 2
    assert torch.allclose(q_after[0, 0, 2, 0], torch.tensor(5.0))   # 3 + 2
    assert torch.allclose(q_after[0, 1, 0, 0], torch.tensor(-1.0))  # 1 - 2
    assert torch.allclose(q_after[0, 1, 2, 0], torch.tensor(-1.0))  # 1 - 2
    # Unmasked (text/audio) rows are untouched.
    assert torch.allclose(q_after[0, :, 1, :], q[0, :, 1, :])
    assert torch.allclose(q_after[0, :, 3, :], q[0, :, 3, :])


def test_not_enough_data_captures_without_steering(monkeypatch):
    q = torch.ones(1, 2, 4, 1) * 7.0
    patched, capture_holder = _install(monkeypatch, None, strength=1.0)

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        return q_.clone()

    out = _run_attention_inside_block(
        patched, block=0,
        attn_fn=lambda: _override_fn(patched)(
            fake_func, q, torch.zeros_like(q), torch.zeros_like(q), 2, skip_reshape=True))
    # No direction -> Q passed through unmodified, but still captured for next time.
    assert torch.allclose(out, q)
    assert 0 in capture_holder[0]


def test_batch_greater_than_one_is_a_noop(monkeypatch, capsys):
    q = torch.ones(2, 2, 4, 1)
    patched, capture_holder = _install(monkeypatch, torch.tensor([1.0, -1.0]), strength=1.0)

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        return q_.clone()

    out = _run_attention_inside_block(
        patched, block=0,
        attn_fn=lambda: _override_fn(patched)(
            fake_func, q, torch.zeros_like(q), torch.zeros_like(q), 2, skip_reshape=True))
    assert torch.allclose(out, q)
    assert capture_holder[0] == {}
    assert "batch size > 1" in capsys.readouterr().out


if __name__ == "__main__":
    print("run via pytest -- these tests need the monkeypatch/capsys fixtures")
