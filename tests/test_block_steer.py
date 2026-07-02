"""Block Steering: recorder/steer hooks on a toy AV model + profile/score/gain math.

The toy model mirrors the LTX-AV calling convention the hooks rely on: blocks held in
`model.model.diffusion_model.transformer_blocks` (a ModuleList of >= 28 entries, which
is what _funpack_locate_blocks requires), each called as block((vx, ax), ...) and
returning (vx_out, ax_out). Persistence tests point the refinement store at a tmp dir.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import block_steer as bs  # noqa: E402

N_BLOCKS = 28  # _funpack_locate_blocks' minimum believable transformer depth
DIM = 8


class _ToyAVBlock(torch.nn.Module):
    def __init__(self, v_gain, a_gain=0.5):
        super().__init__()
        self.v_gain = v_gain
        self.a_gain = a_gain

    def forward(self, x, **kwargs):
        vx, ax = x
        return vx + self.v_gain * torch.ones_like(vx), ax + self.a_gain * torch.ones_like(ax)


def _toy_model(v_gains):
    diff = torch.nn.Module()
    diff.transformer_blocks = torch.nn.ModuleList([_ToyAVBlock(g) for g in v_gains])
    inner = types.SimpleNamespace(diffusion_model=diff)
    return types.SimpleNamespace(model=inner, model_options={})


def _run(model, vx=None, ax=None):
    vx = vx if vx is not None else torch.ones(1, 4, DIM)
    ax = ax if ax is not None else torch.ones(1, 2, DIM)
    for blk in model.model.diffusion_model.transformer_blocks:
        vx, ax = blk((vx, ax))
    return vx, ax


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect refinement sidecars into a tmp dir (no conditioning.py import needed)."""
    def _path(key, name, prefix="refine_v2", extension="json"):
        return str(tmp_path / f"{key}.{name}.{extension}")
    monkeypatch.setattr(bs, "_state_path", lambda key, name: _path(key, name, extension="pt"))
    return tmp_path


def test_recorder_ranks_blocks_by_video_activity():
    gains = [0.01] * N_BLOCKS
    gains[5], gains[20] = 2.0, 1.0  # blocks 5 and 20 move the video stream hardest
    model = _toy_model(gains)
    rec = bs.BlockActivityRecorder(N_BLOCKS)
    handles = bs.install_recorder(model, rec)
    assert len(handles) == N_BLOCKS
    _run(model)
    _run(model)
    bs.remove_handles(handles)
    fp = rec.fingerprint()
    assert fp is not None and int(rec.counts[0]) == 2
    assert int(fp.argmax()) == 5
    assert fp[20] > fp[3]
    # hooks removed - a third run must not record
    _run(model)
    assert int(rec.counts[0]) == 2


def test_hooks_carry_leak_strip_tag():
    """Recorder AND steer hooks must be tagged so the sampler's defensive
    strip_funpack_block_hooks cleans them if a run crashes mid-sample."""
    model = _toy_model([1.0] * N_BLOCKS)
    rec = bs.BlockActivityRecorder(N_BLOCKS)
    h1 = bs.install_recorder(model, rec)
    h2 = bs.install_steer(model, [1.05] * N_BLOCKS)
    try:
        for blk in model.model.diffusion_model.transformer_blocks:
            fns = list(blk._forward_hooks.values())
            assert fns and all(getattr(f, bs._FUNPACK_HOOK_TAG, False) for f in fns)
    finally:
        bs.remove_handles(h1 + h2)


def test_steer_scales_video_residual_only():
    model = _toy_model([1.0] * N_BLOCKS)
    v_ref, a_ref = _run(model)
    gains = [1.0] * N_BLOCKS
    gains[3] = 1.10   # boost block 3's video contribution by 10%
    gains[7] = 0.90   # damp block 7's
    handles = bs.install_steer(model, gains)
    assert len(handles) == 2  # gain==1.0 blocks get NO hook (native path)
    v_out, a_out = _run(model)
    bs.remove_handles(handles)
    # video: each block adds +1; two blocks changed by +/-0.1 -> net exactly equal
    assert torch.allclose(v_out, v_ref, atol=1e-5)  # +0.1 and -0.1 cancel
    gains[7] = 1.0
    handles = bs.install_steer(model, gains)
    v_out2, _ = _run(model)
    bs.remove_handles(handles)
    assert torch.allclose(v_out2 - v_ref, torch.full_like(v_out2, 0.10), atol=1e-5)
    # audio stream byte-identical under steering
    assert torch.equal(a_out, a_ref)


def test_credit_wrapper_scores_and_credits():
    model = _toy_model([1.0] * N_BLOCKS)
    rec = bs.BlockActivityRecorder(N_BLOCKS)
    handles = bs.install_recorder(model, rec)

    class _VF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = iter([0.1, 0.5, 0.2])
        def compress(self, x):
            return x.mean().unsqueeze(0)
        def forward(self, c):
            return torch.tensor([next(self.scores)])

    def apply_fn(x, t, **c):
        _run(model)          # blocks fire -> recorder.cur populated
        return x * 0.9

    wrapper = bs.make_credit_wrapper(None, rec, _VF())
    args = {"input": torch.ones(1, 4, DIM), "timestep": torch.tensor([0.9]), "c": {}}
    for _ in range(3):
        wrapper(apply_fn, args)
    bs.remove_handles(handles)
    assert rec.credit_steps == 2  # first call only sets the baseline
    cv = rec.credit_vector()
    assert cv is not None and cv.shape == (N_BLOCKS,)
    # In the toy model vx grows by +1 per block, so relative deltas DECREASE with depth:
    # early blocks are the most active. Net score delta across the run is positive
    # (0.1 -> 0.5 -> 0.2), so credit must follow the centered activity gradient -
    # positive for the most active (early) blocks, negative for the least active (late).
    assert float(cv[0]) > 0 > float(cv[-1])
    assert int(cv.argmax()) == 0


def _rec_with(activity):
    r = bs.BlockActivityRecorder(activity.numel())
    r.sums = activity.clone().float(); r.counts = torch.ones(activity.numel())
    return r


def test_snapshot_profile_scores_gains_roundtrip(store):
    key = "toykey"
    base = torch.full((N_BLOCKS,), 0.5)
    hot = base.clone(); hot[10] = 1.5    # block 10 fires hard on the GOOD run
    cold = base.clone(); cold[2] = 1.5   # block 2 fires hard on the BAD run

    assert bs.save_run_snapshot(key, _rec_with(hot))
    assert bs.update_profile_with_rating(key, 1.0) == 1        # Perfect
    assert bs.update_profile_with_rating(key, 1.0) is None     # snapshot consumed - no double pair
    scores, status = bs.block_scores_with_status(key)
    assert scores is None and "have 1" in status               # 1 rated run < MIN_RATED_RUNS

    assert bs.save_run_snapshot(key, _rec_with(cold))
    assert bs.update_profile_with_rating(key, -0.9) == 2       # Awful
    scores, status = bs.block_scores_with_status(key)
    assert status == "ready"
    assert scores is not None and scores.shape == (N_BLOCKS,)
    assert int(scores.argmax()) == 10 and int(scores.argmin()) == 2
    assert abs(float(scores.mean())) < 1e-5                    # zero-mean: redistributes, no drift
    assert float(scores.abs().max()) == pytest.approx(1.0, abs=1e-5)

    gains = bs.gains_from_scores(scores, 0.05)
    assert max(gains) == pytest.approx(1.0 + 0.05, abs=1e-4)
    assert min(gains) == pytest.approx(1.0 - 0.05, abs=1e-4)
    # hard cap regardless of strength widget
    gains_hi = bs.gains_from_scores(scores, 5.0)
    assert max(gains_hi) <= 1.0 + bs.MAX_GAIN_DELTA + 1e-6
    assert min(gains_hi) >= 1.0 - bs.MAX_GAIN_DELTA - 1e-6


def test_same_pole_ratings_need_contrast_not_more_runs(store):
    """The v1 regression: MANY rated runs, all rated alike -> the profile must say WHY
    it can't steer (no reward contrast), not pretend it needs more runs."""
    key = "samepole"
    for _ in range(16):
        assert bs.save_run_snapshot(key, _rec_with(torch.rand(N_BLOCKS)))
        assert bs.update_profile_with_rating(key, 1.0) is not None
    scores, status = bs.block_scores_with_status(key)
    assert scores is None
    assert "rated alike" in status and "16" in status
    # one contrasting rating unlocks attribution
    assert bs.save_run_snapshot(key, _rec_with(torch.rand(N_BLOCKS)))
    bs.update_profile_with_rating(key, -0.9)
    scores, status = bs.block_scores_with_status(key)
    assert scores is not None and status == "ready"


def test_every_reward_value_contributes(store):
    """No dead band: mild rewards like Missing action (0.05) / Wrong details (0.20)
    count with their actual value instead of being dropped."""
    key = "mild"
    base = torch.full((N_BLOCKS,), 0.5)
    hi = base.clone(); hi[7] = 1.2
    lo = base.clone(); lo[21] = 1.2
    bs.save_run_snapshot(key, _rec_with(hi)); bs.update_profile_with_rating(key, 0.35)
    bs.save_run_snapshot(key, _rec_with(lo)); bs.update_profile_with_rating(key, 0.05)
    scores, status = bs.block_scores_with_status(key)
    assert status == "ready"
    assert int(scores.argmax()) == 7 and int(scores.argmin()) == 21


def test_depth_mismatch_and_v1_profile_reset(store):
    key = "depth"
    bs.save_run_snapshot(key, _rec_with(torch.rand(N_BLOCKS)))
    bs.update_profile_with_rating(key, 1.0)
    r2 = torch.rand(N_BLOCKS + 4)  # different model depth
    bs.save_run_snapshot(key, _rec_with(r2))
    assert bs.update_profile_with_rating(key, 1.0) == 1  # fresh profile, not a mixed one
    prof = torch.load(bs._state_path(key, "block_profile"), weights_only=False)
    assert prof["n_blocks"] == N_BLOCKS + 4
    # a v1 pole-EMA profile on disk (liked_n, no history) is discarded, not mixed in
    torch.save({"n_blocks": N_BLOCKS, "liked": torch.rand(N_BLOCKS), "liked_n": 16},
               bs._state_path(key, "block_profile"))
    bs.save_run_snapshot(key, _rec_with(torch.rand(N_BLOCKS)))
    assert bs.update_profile_with_rating(key, 1.0) == 1
