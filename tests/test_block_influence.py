"""Per-block influence probe: the pending->rating pairing, the profile maths, and the
flatness readout that decides whether rating-driven block weighting is worth pursuing."""
import sys

import pytest
import torch

sys.path.insert(0, ".")
import block_influence as bi  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch, tmp_path):
    monkeypatch.setattr(bi, "state_path", lambda key: str(tmp_path / f"{key}.pt"))


def _run(key, profile, reward):
    bi.save_pending(key, profile)
    return bi.commit(key, reward)


# --- pending / commit contract -------------------------------------------------

def test_commit_reports_recorded_on_a_real_pairing():
    assert _run("k", {0: 0.1, 1: 0.2}, 1.0) == "recorded"


def test_commit_reports_no_pending_with_nothing_to_pair():
    assert bi.commit("k", 1.0) == "no_pending"


def test_commit_reports_no_key():
    assert bi.commit("", 1.0) == "no_key"


def test_pending_overwrites_rather_than_accumulating():
    bi.save_pending("k", {0: 0.1})
    bi.save_pending("k", {0: 0.9})  # a second run before any rating
    bi.commit("k", 1.0)
    rows = bi._load("k")["rows"]
    assert len(rows) == 1 and rows[0]["profile"][0] == 0.9


def test_clear_all_removes_the_state():
    _run("k", {0: 0.1}, 1.0)
    bi.clear_all("k")
    assert bi.profile("k")["overall"] == {}


def test_no_key_is_a_safe_no_op():
    bi.save_pending("", {0: 0.1})
    bi.commit("", 1.0)
    bi.clear_all("")


# --- profile maths -------------------------------------------------------------

def test_overall_is_the_mean_across_runs_regardless_of_rating():
    _run("k", {0: 0.2, 1: 0.4}, 1.0)
    _run("k", {0: 0.4, 1: 0.6}, -1.0)
    got = bi.profile("k")["overall"]
    assert got[0] == pytest.approx(0.3)
    assert got[1] == pytest.approx(0.5)


def test_difference_needs_min_per_group_on_each_side():
    for _ in range(bi.MIN_PER_GROUP):
        _run("k", {0: 0.5}, 1.0)
    _run("k", {0: 0.1}, -1.0)  # only one disliked
    p = bi.profile("k")
    assert p["difference"] is None
    assert p["n_liked"] == bi.MIN_PER_GROUP and p["n_disliked"] == 1


def test_difference_is_liked_mean_minus_disliked_mean():
    for _ in range(bi.MIN_PER_GROUP):
        _run("k", {0: 0.8, 1: 0.2}, 1.0)
        _run("k", {0: 0.4, 1: 0.2}, -1.0)
    diff = bi.profile("k")["difference"]
    assert diff[0] == pytest.approx(0.4)   # block 0 runs hotter on liked runs
    assert diff[1] == pytest.approx(0.0)   # block 1 is indifferent to the rating


def test_a_block_missing_from_a_row_is_skipped_not_counted_as_zero():
    """A run that hooked fewer blocks must not drag the absent block's mean toward 0 --
    the same rule h3_repr_steering.direction() follows for a missing descriptor."""
    _run("k", {0: 0.4, 1: 0.4}, 1.0)
    _run("k", {0: 0.6}, 1.0)  # no block 1 this run
    got = bi.profile("k")["overall"]
    assert got[0] == pytest.approx(0.5)
    assert got[1] == pytest.approx(0.4)


def test_neutral_weight_rows_count_toward_overall_but_neither_group():
    _run("k", {0: 1.0}, 0.0)
    p = bi.profile("k")
    assert p["n_liked"] == 0 and p["n_disliked"] == 0
    assert p["overall"][0] == pytest.approx(1.0)


# --- flatness: the number the whole probe exists to produce ---------------------

def test_a_perfectly_flat_profile_reports_zero_flatness():
    """Every block moving the stream by the same amount is the null result that says
    rating-driven block weighting has nothing to grip."""
    _run("k", {b: 0.5 for b in range(10)}, 1.0)
    assert bi.profile("k")["flatness"] == pytest.approx(0.0)


def test_a_structured_profile_reports_nonzero_flatness():
    _run("k", {0: 0.1, 1: 0.5, 2: 1.5}, 1.0)
    assert bi.profile("k")["flatness"] > 0.5


def test_flatness_is_scale_invariant():
    """Coefficient of variation, not raw spread: doubling every block's delta describes the
    same shape and must not read as twice as structured."""
    _run("a", {0: 0.1, 1: 0.2, 2: 0.4}, 1.0)
    _run("b", {0: 0.2, 1: 0.4, 2: 0.8}, 1.0)
    assert bi.profile("a")["flatness"] == pytest.approx(bi.profile("b")["flatness"])


def test_flatness_is_none_with_no_data():
    assert bi.profile("k")["flatness"] is None


def test_corrupt_state_file_degrades_to_empty(tmp_path, monkeypatch):
    path = tmp_path / "junk.pt"
    path.write_bytes(b"not a torch file")
    monkeypatch.setattr(bi, "state_path", lambda key: str(path))
    assert bi.profile("k")["overall"] == {}
    assert bi.commit("k", 1.0) == "no_pending"


def test_rows_survive_a_reload_as_plain_floats():
    """weights_only=True load: the state must never depend on unpickling a custom class."""
    _run("k", {0: 0.25}, 1.0)
    rows = bi._load("k")["rows"]
    assert isinstance(rows[0]["profile"][0], float)
    assert isinstance(rows[0]["weight"], float)
    assert not isinstance(rows[0]["profile"][0], torch.Tensor)


# --- the sampler hook itself ---------------------------------------------------
#
# The module above is only half the probe. The half that can silently record nothing is the
# hook: if the video mask misses, the input tensor is not where it is expected, or the
# swallow-everything guard eats a real error, the profile comes back empty and looks like
# "this model has no block structure" rather than "the probe never ran". These exercise the
# registered hook directly, the way the model calls it.

import types  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402


class _Model:
    def __init__(self):
        self.model_options = {}

    def clone(self):
        m = _Model()
        m.model_options = {k: dict(v) if isinstance(v, dict) else v
                           for k, v in self.model_options.items()}
        return m


def _install(capture, n_blocks=2, max_rows=512):
    s = samplers.FunPackLTXAVSceneChainSampler()
    patched = s._install_block_influence(_Model(), capture, n_blocks=n_blocks,
                                          max_rows=max_rows)
    return patched.model_options["transformer_options"]["patches_replace"]["dit"]


def _call(hook, src, out, mod_segments):
    """Drive the hook the way minimax/model.py does."""
    return hook({"img": src, "mod_segments": mod_segments},
                {"original_block": lambda a: {"img": out}})


def test_hook_records_the_relative_delta_on_video_rows():
    capture = [{}]
    dit = _install(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 3.0                       # ||video rows|| = 6.0 over 4 rows
    out = src.clone()
    out[:, 1] = 3.0                       # delta has the same norm as the input
    res = _call(dit[("double_block", 0)], src, out, [(0, 4, 6)])  # tag 6 % 3 == 0 -> video
    assert torch.equal(res["img"], out)   # measurement must not modify the stream
    recorded = torch.stack(capture[0][0]).mean().item()
    assert recorded == pytest.approx(1.0, rel=1e-4)


def test_hook_averages_across_steps():
    capture = [{}]
    dit = _install(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    for scale in (1.0, 3.0):              # two "steps" with different deltas
        out = src.clone()
        out[:, 1] = scale
        _call(dit[("double_block", 0)], src, out, [(0, 4, 6)])
    assert torch.stack(capture[0][0]).mean().item() == pytest.approx(2.0, rel=1e-4)


def test_text_and_audio_rows_are_not_measured():
    """Only video rows count -- a huge delta on text rows must not register."""
    capture = [{}]
    dit = _install(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    out = src.clone()
    out[2:, 1] = 50.0                     # rows 2-3 are text below
    _call(dit[("double_block", 0)], src, out, [(0, 2, 6), (2, 4, 7)])
    recorded = torch.stack(capture[0][0]).mean().item()
    assert recorded == pytest.approx(0.0, abs=1e-6)


def test_no_video_segment_records_nothing_rather_than_zero():
    capture = [{}]
    dit = _install(capture)
    src = torch.ones(4, 8)
    _call(dit[("double_block", 0)], src, src * 2, [(0, 4, 7)])  # text only
    assert capture[0] == {}


def test_each_block_accumulates_separately():
    capture = [{}]
    dit = _install(capture, n_blocks=3)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    for block, scale in ((0, 1.0), (2, 4.0)):
        out = src.clone()
        out[:, 1] = scale
        _call(dit[("double_block", block)], src, out, [(0, 4, 6)])
    assert set(capture[0]) == {0, 2}      # block 1 was never called
    assert torch.stack(capture[0][0]).mean().item() == pytest.approx(1.0, rel=1e-4)
    assert torch.stack(capture[0][2]).mean().item() == pytest.approx(4.0, rel=1e-4)


def test_row_subsampling_bounds_the_reduction():
    """The stride keeps the cost flat as the sequence grows; the estimate must stay close."""
    capture = [{}]
    dit = _install(capture, max_rows=16)
    n = 4096
    src = torch.zeros(n, 8)
    src[:, 0] = 1.0
    out = src.clone()
    out[:, 1] = 2.0
    _call(dit[("double_block", 0)], src, out, [(0, n, 6)])
    assert torch.stack(capture[0][0]).mean().item() == pytest.approx(2.0, rel=1e-3)


def test_an_existing_patch_at_the_same_block_is_chained_not_replaced():
    """Installing over another mechanism's hook must call it, not drop it."""
    calls = []

    class _M(_Model):
        pass

    s = samplers.FunPackLTXAVSceneChainSampler()
    base = _M()

    def _inner(args, extra):
        calls.append(True)
        return {"img": args["img"] * 2}

    base.model_options["transformer_options"] = {
        "patches_replace": {"dit": {("double_block", 0): _inner}}}
    capture = [{}]
    patched = s._install_block_influence(base, capture, n_blocks=1)
    hook = patched.model_options["transformer_options"]["patches_replace"]["dit"][
        ("double_block", 0)]
    src = torch.ones(4, 8)
    res = _call(hook, src, src, [(0, 4, 6)])
    assert calls == [True]                     # inner ran
    assert torch.equal(res["img"], src * 2)    # and its output is what propagates


def test_a_broken_hook_never_breaks_sampling():
    capture = [{}]
    dit = _install(capture)
    src = torch.ones(4, 8)
    # mod_segments of the wrong shape entirely: must pass the block output through anyway.
    res = _call(dit[("double_block", 0)], src, src * 3, "not-segments")
    assert torch.equal(res["img"], src * 3)
    assert capture[0] == {}


# --- the collection switch ------------------------------------------------------
#
# Research data is opt-in from Settings > Refinement & Taste. This switch gates COLLECTION
# only -- it must never be wired to anything that records a rating (see the 2026-09-03
# session: a "pause capture" toggle that silently stopped writing ratings).

@pytest.fixture
def _switch(monkeypatch, tmp_path):
    monkeypatch.delenv(bi._ENV_SWITCH, raising=False)
    monkeypatch.setattr(bi, "_switch_dir", lambda: str(tmp_path / "sw"))
    return tmp_path


def test_collection_is_off_by_default(_switch):
    assert bi.collection_enabled() is False


def test_setting_it_on_persists_to_disk_and_to_this_process(_switch, monkeypatch):
    assert bi.set_collection_enabled(True) is True
    assert bi.collection_enabled() is True
    # A restarted process has no env var; the on-disk copy must still say on.
    monkeypatch.delenv(bi._ENV_SWITCH, raising=False)
    assert bi.collection_enabled() is True


def test_turning_it_off_persists_too(_switch, monkeypatch):
    bi.set_collection_enabled(True)
    bi.set_collection_enabled(False)
    monkeypatch.delenv(bi._ENV_SWITCH, raising=False)
    assert bi.collection_enabled() is False


def test_env_var_wins_over_disk(_switch, monkeypatch):
    bi.set_collection_enabled(False)
    monkeypatch.setenv(bi._ENV_SWITCH, "1")
    assert bi.collection_enabled() is True


def test_unwritable_switch_dir_still_sets_this_session(_switch, monkeypatch):
    monkeypatch.setattr(bi, "_switch_dir", lambda: "/nonexistent-root/nope")
    assert bi.set_collection_enabled(True) is True
    assert bi.collection_enabled() is True   # env var carries it for this process


def test_the_switch_does_not_gate_recording(_switch, monkeypatch, tmp_path):
    """HARD RULE: this toggle controls whether the PROBE is installed, nothing else. It must
    never stand between a rating and the data it belongs to -- save_pending/commit are
    unconditional, exactly as h3_repr_steering's are."""
    monkeypatch.setattr(bi, "state_path", lambda key: str(tmp_path / f"{key}.pt"))
    bi.set_collection_enabled(False)
    bi.save_pending("k", {0: 0.5})
    assert bi.commit("k", 1.0) == "recorded"
    assert bi.profile("k")["overall"][0] == pytest.approx(0.5)


# --- novelty: what each block ADDS that the one before it didn't ------------------
#
# Magnitude alone cannot tell "this block added something new" from "this block pushed
# harder in the direction the last one already went". The cosine with the previous block's
# delta is what separates those, and it is the only number here that speaks to whether the
# 50 blocks are doing 50 different things or one thing 50 times.

def test_novelty_is_recorded_and_averaged():
    bi.save_pending("k", {1: 0.1, 2: 0.2}, novelty={2: 0.0})
    bi.commit("k", 1.0)
    bi.save_pending("k", {1: 0.1, 2: 0.2}, novelty={2: 1.0})
    bi.commit("k", 1.0)
    p = bi.profile("k")
    assert p["novelty"][2] == pytest.approx(0.5)
    assert p["mean_novelty"] == pytest.approx(0.5)


def test_novelty_is_optional_and_old_rows_stay_readable():
    """A row recorded before novelty existed carries none; it must not break the readout or
    be counted as a 0.0 cosine (which would read as 'perfectly orthogonal')."""
    bi.save_pending("k", {1: 0.5})               # no novelty
    bi.commit("k", 1.0)
    bi.save_pending("k", {1: 0.5}, novelty={1: 0.8})
    bi.commit("k", 1.0)
    p = bi.profile("k")
    assert p["novelty"][1] == pytest.approx(0.8)  # averaged over the row that HAS it only


def test_no_novelty_anywhere_reports_none_not_zero():
    bi.save_pending("k", {1: 0.5})
    bi.commit("k", 1.0)
    p = bi.profile("k")
    assert p["novelty"] == {}
    assert p["mean_novelty"] is None


def test_negative_novelty_survives_the_round_trip():
    """Blocks partly undoing each other is a real outcome and the most interesting one --
    it must not be clamped or dropped on the way through."""
    bi.save_pending("k", {3: 0.4}, novelty={3: -0.6})
    bi.commit("k", 1.0)
    assert bi.profile("k")["novelty"][3] == pytest.approx(-0.6)


# --- the hook's novelty half ------------------------------------------------------

def _install2(capture, n_blocks=4, max_rows=512):
    s = samplers.FunPackLTXAVSceneChainSampler()
    patched = s._install_block_influence(_Model(), capture, n_blocks=n_blocks,
                                          max_rows=max_rows)
    return patched.model_options["transformer_options"]["patches_replace"]["dit"]


def _push(dit, block, src, delta_vec, segs=None):
    out = src + delta_vec
    return _call(dit[("double_block", block)], src, out, segs or [(0, 4, 6)])


def test_hook_records_orthogonal_deltas_as_zero_novelty():
    capture = [{}, {}]
    dit = _install2(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    d0 = torch.zeros(4, 8); d0[:, 1] = 1.0
    d1 = torch.zeros(4, 8); d1[:, 2] = 1.0     # orthogonal to d0
    _push(dit, 0, src, d0)
    _push(dit, 1, src, d1)
    assert torch.stack(capture[1][1]).mean().item() == pytest.approx(0.0, abs=1e-5)


def test_hook_records_a_repeated_direction_as_novelty_one():
    capture = [{}, {}]
    dit = _install2(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    d = torch.zeros(4, 8); d[:, 1] = 1.0
    _push(dit, 0, src, d)
    _push(dit, 1, src, d * 3.0)                # same direction, bigger
    assert torch.stack(capture[1][1]).mean().item() == pytest.approx(1.0, rel=1e-4)


def test_hook_records_an_undoing_block_as_negative_novelty():
    capture = [{}, {}]
    dit = _install2(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    d = torch.zeros(4, 8); d[:, 1] = 1.0
    _push(dit, 0, src, d)
    _push(dit, 1, src, -d)
    assert torch.stack(capture[1][1]).mean().item() == pytest.approx(-1.0, rel=1e-4)


def test_the_first_block_of_a_step_has_no_predecessor():
    """Block 0 of step 2 must not be compared against block 49 of step 1 -- that pairs the
    end of one denoise step with the start of the next, which is not a depth relationship."""
    capture = [{}, {}]
    dit = _install2(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    d = torch.zeros(4, 8); d[:, 1] = 1.0
    for _step in range(2):                     # two steps, blocks 0..2 each
        for b in range(3):
            _push(dit, b, src, d)
    assert 0 not in capture[1]                 # never a novelty entry for the first block
    assert len(capture[1][1]) == 2             # one per step, not three


def test_novelty_is_skipped_when_the_holder_has_no_slot_for_it():
    """Back-compat: a caller passing a 1-slot holder still gets magnitudes and no crash."""
    capture = [{}]
    dit = _install2(capture)
    src = torch.zeros(4, 8)
    src[:, 0] = 1.0
    d = torch.zeros(4, 8); d[:, 1] = 1.0
    _push(dit, 0, src, d)
    _push(dit, 1, src, d)
    assert set(capture[0]) == {0, 1}
    assert len(capture) == 1
