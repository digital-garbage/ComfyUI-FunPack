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


def test_kind_isolation_uses_real_persistence(monkeypatch, tmp_path):
    """Real save_pending/commit/direction (no mocking of the persistence layer itself) for
    BOTH kind="repr_steer" (REINS' own default) and kind="q_steer" at the SAME block index,
    same refinement_key -- must never collide, even though a block index means a different
    vector space in each store."""
    monkeypatch.setattr(_rs, "state_path",
                        lambda key, kind="repr_steer": str(tmp_path / f"{key}.{kind}.pt"))
    key = "fake_key"
    block = 0
    # REINS' own (hidden-state) direction: liked cluster near [1,0], disliked near [-1,0].
    for i in range(3):
        _rs.save_pending(key, {block: torch.tensor([1.0 + i * 0.01, 0.0])})
        _rs.commit(key, reward=1.0)
        _rs.save_pending(key, {block: torch.tensor([-1.0 - i * 0.01, 0.0])})
        _rs.commit(key, reward=-1.0)
    # q_steer's own direction at the SAME block index: liked/disliked swapped on axis 2
    # instead, so if the two stores ever shared a file/rows the directions would collide.
    for i in range(3):
        _rs.save_pending(key, {block: torch.tensor([0.0, 1.0 + i * 0.01])}, kind="q_steer")
        _rs.commit(key, reward=1.0, kind="q_steer")
        _rs.save_pending(key, {block: torch.tensor([0.0, -1.0 - i * 0.01])}, kind="q_steer")
        _rs.commit(key, reward=-1.0, kind="q_steer")

    repr_dir, r_pos, r_neg = _rs.direction(key, block=block)
    q_dir, q_pos, q_neg = _rs.direction(key, block=block, kind="q_steer")
    assert r_pos == 3 and r_neg == 3
    assert q_pos == 3 and q_neg == 3
    # REINS' direction points along axis 1, q_steer's along axis 2 -- if the stores collided
    # (same file, or rows merged) both would reflect a mix of both axes instead.
    assert repr_dir[0].abs() > 0.9 and repr_dir[1].abs() < 0.1
    assert q_dir[1].abs() > 0.9 and q_dir[0].abs() < 0.1


def test_two_named_blocks_steer_with_independent_directions(monkeypatch):
    """steer_block='0,1' -- each block must apply its OWN direction, not the other's."""
    directions = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}

    def _fake_direction(refinement_key, block=None, kind="repr_steer"):
        assert kind == "q_steer"
        return directions[block], 5, 5
    monkeypatch.setattr(_rs, "direction", _fake_direction)
    capture_holder = [{}]
    patched = S()._install_h3_q_steering(
        _FakeModel(), "fake_key", 1.0, capture_holder, steer_block="0,1")

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        return q_.clone()

    results = {}
    for block in (0, 1):
        q = torch.zeros(1, 2, 4, 1)
        q[0, 0, 0, 0] = 5.0  # heads=2, head_dim=1; masked (video) row 0 only for simplicity
        results[block] = _run_attention_inside_block(
            patched, block=block,
            attn_fn=lambda q=q: _override_fn(patched)(
                fake_func, q, torch.zeros_like(q), torch.zeros_like(q), 2, skip_reshape=True))

    # Block 0's direction is [1,0] -> only head 0 shifts; block 1's is [0,1] -> only head 1
    # shifts. If the blocks shared one direction (a wiring bug), both would shift identically.
    assert results[0][0, 0, 0, 0] != 5.0
    assert torch.allclose(results[0][0, 1, 0, 0], torch.tensor(0.0))
    assert torch.allclose(results[1][0, 0, 0, 0], torch.tensor(5.0))
    assert results[1][0, 1, 0, 0] != 0.0
    assert 0 in capture_holder[0] and 1 in capture_holder[0]


def test_sample_signature_default_matches_input_types_default():
    """A ComfyUI 'optional' widget that predates a saved workflow is simply ABSENT from the
    call -- sample()'s own Python default fires instead of whatever INPUT_TYPES declares.
    h3_q_steer_block must default to blank (off) in BOTH places, or every pre-existing
    workflow silently turns this on (caught live: sample()'s default was "0,1" while
    INPUT_TYPES said "" -- an editor-side workflow, or ANY graph saved before this widget
    existed, would have started capturing/hooking/cloning on every H3 generation)."""
    import inspect
    sig_default = inspect.signature(S.sample).parameters["h3_q_steer_block"].default
    input_types_default = S.INPUT_TYPES()["optional"]["h3_q_steer_block"][1]["default"]
    assert sig_default == input_types_default == "", (
        f"sample() default {sig_default!r} != INPUT_TYPES default {input_types_default!r} "
        "-- must both be blank")
    strength_sig_default = inspect.signature(S.sample).parameters["h3_q_steer_strength"].default
    strength_it_default = S.INPUT_TYPES()["optional"]["h3_q_steer_strength"][1]["default"]
    assert strength_sig_default == strength_it_default == 0.0


def test_strength_zero_does_not_claim_to_be_applying(monkeypatch, capsys):
    """A learned direction existing (enough rating history) is not the same as it being
    APPLIED -- injection is separately gated on strength > 0. The console message must not
    claim "applying" when strength is 0, or a passively-accumulated rating history would
    print a false positive on every subsequent generation."""
    patched, capture_holder = _install(monkeypatch, torch.tensor([1.0, -1.0]), strength=0.0)
    q = torch.ones(1, 2, 4, 1)

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        return q_.clone()

    out = _run_attention_inside_block(
        patched, block=0,
        attn_fn=lambda: _override_fn(patched)(
            fake_func, q, torch.zeros_like(q), torch.zeros_like(q), 2, skip_reshape=True))
    assert torch.allclose(out, q)  # strength 0 -> no injection despite a real direction
    stdout = capsys.readouterr().out
    assert "applying learned direction" not in stdout
    assert "strength is 0" in stdout


def test_mismatched_direction_length_does_not_crash(monkeypatch, capsys):
    """A direction banked under a different heads*head_dim layout (a model swap on the same
    refinement_key, or a future H3 variant) cannot be reshaped onto this run's Q -- must
    degrade with a one-time notice, not raise, the same way batch>1 does above."""
    # direction has 3 values; this run's Q is heads=2, head_dim=1 -> needs exactly 2.
    patched, capture_holder = _install(monkeypatch, torch.tensor([1.0, 2.0, 3.0]), strength=1.0)
    q = torch.ones(1, 2, 4, 1) * 5.0

    def fake_func(q_, k_, v_, heads_, mask=None, skip_reshape=True, **kw):
        return q_.clone()

    out = _run_attention_inside_block(
        patched, block=0,
        attn_fn=lambda: _override_fn(patched)(
            fake_func, q, torch.zeros_like(q), torch.zeros_like(q), 2, skip_reshape=True))
    assert torch.allclose(out, q)  # not applied, but did not crash
    assert 0 in capture_holder[0]  # capture still happens
    assert "learned direction has 3 values" in capsys.readouterr().out


if __name__ == "__main__":
    print("run via pytest -- these tests need the monkeypatch/capsys fixtures")
