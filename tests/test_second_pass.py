"""second_pass — a general two-pass split on the Chain Sampler, independent of anchor_shift.

anchor_shift built the machinery (stop the schedule, hand the state over, re-enter exactly);
this exposes it on its own so a scene can be finished differently from how it was started.
The invariants that matter here are the ones a split can silently get wrong:

1. Pass 1 stops on a REAL step of the user's schedule, found by walking it.
2. Pass 2 re-enters at or above the cut — never below, because noise can be added back but
   never removed (the same rule anchor_shift's rewind lives by).
3. A connected second schedule replaces pass 2 wholesale, and its own first sigma becomes
   the entry point, so the sampler's phases are measured against IT and not the main one.
4. Nothing is ever a silent no-op: every refusal comes back with a reason naming the fix.
5. The i2v anchor stays PINNED across the split — the opposite of anchor_shift, which
   deletes the pin on purpose. Without this pass 2 re-denoises the reference frame.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402


# The user's own 9-step schedule.
SCHEDULE = torch.tensor([1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0])


def _node():
    return samplers.FunPackLTXAVSceneChainSampler()


def _plan(cut, restart=0.0, alt=None):
    return _node()._second_pass_plan(SCHEDULE, cut, restart, alt)


# ── the split ───────────────────────────────────────────────────────────────

def test_continue_splits_the_schedule_without_adding_a_step():
    """restart 0 is behaviour-neutral: the two halves are the whole schedule, cut once, and
    pass 2 resumes from the exact handed-over state (resume=True -> no re-noise)."""
    plan, reason = _plan(0.603)
    assert reason is None
    first, second, cut_at, entry_at, resume, ctx = plan
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.955, 0.893, 0.812, 0.715, 0.603]
    assert [round(v, 3) for v in second.tolist()] == [0.603, 0.482, 0.241, 0.121, 0.0]
    assert abs(cut_at - 0.603) < 1e-5 and abs(entry_at - 0.603) < 1e-5
    assert resume is True
    # No step is run twice: 5 + 4 == the schedule's own 9.
    assert (first.numel() - 1) + (second.numel() - 1) == SCHEDULE.numel() - 1
    # Pass 2's phases must still be measured against the whole schedule.
    assert ctx["sigmas"] is SCHEDULE and ctx["offset"] == 5


def test_rewind_re_enters_higher_and_replays_those_steps():
    plan, reason = _plan(0.482, restart=0.812)
    assert reason is None
    first, second, cut_at, entry_at, resume, ctx = plan
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.955, 0.893, 0.812, 0.715, 0.603, 0.482]
    assert [round(v, 3) for v in second.tolist()] == [0.812, 0.715, 0.603, 0.482, 0.241, 0.121, 0.0]
    assert resume is False          # the caller must rebuild the state at 0.812 first
    assert ctx["offset"] == 3


def test_restart_below_the_cut_is_refused_with_the_fix_named():
    plan, reason = _plan(0.812, restart=0.482)
    assert plan is None
    assert "never a cleaner one" in reason and "second_pass_restart_sigma" in reason


def test_a_cut_outside_the_schedule_is_refused_not_ignored():
    for cut in (1.0, 0.9999, 0.0005):
        plan, reason = _plan(cut)
        assert plan is None and reason, cut


# ── a separate schedule for pass 2 ──────────────────────────────────────────

def test_a_connected_schedule_replaces_pass_two_wholesale():
    """The point of the feature: the second half can have its own step count and spacing.
    Its first sigma is the entry point, and the sampler must be told THAT is its schedule —
    otherwise it measures its quality phase and correction window against the main one."""
    alt = torch.tensor([0.715, 0.5, 0.3, 0.15, 0.0])
    plan, reason = _plan(0.603, restart=0.0, alt=alt)
    assert reason is None
    first, second, cut_at, entry_at, resume, ctx = plan
    assert second is alt
    assert abs(entry_at - 0.715) < 1e-5      # the alt schedule's own first sigma
    assert resume is False                   # 0.603 -> 0.715 needs the state rebuilt
    assert ctx["sigmas"] is alt and ctx["offset"] == 0
    # ...and pass 1 is unaffected by it.
    assert [round(v, 3) for v in first.tolist()] == [1.0, 0.955, 0.893, 0.812, 0.715, 0.603]


def test_a_connected_schedule_starting_at_the_cut_is_an_exact_handover():
    alt = torch.tensor([0.603, 0.4, 0.2, 0.0])
    plan, reason = _plan(0.603, alt=alt)
    assert reason is None and plan[4] is True     # resume -> nothing added


def test_a_connected_schedule_starting_below_the_cut_is_refused():
    alt = torch.tensor([0.3, 0.15, 0.0])
    plan, reason = _plan(0.603, alt=alt)
    assert plan is None
    assert "second_pass_sigmas starts at" in reason


def test_a_degenerate_connected_schedule_falls_back_to_the_restart_sigma():
    """A single-sigma (or empty) SIGMAS input is not a schedule — it must not be treated as
    one, and the restart sigma still governs."""
    for alt in (None, torch.tensor([0.5]), torch.tensor([])):
        plan, reason = _plan(0.603, restart=0.812, alt=alt)
        assert reason is None
        assert plan[1].tolist()[0] == SCHEDULE[3].item()   # a slice of the MAIN schedule


# ── the pin survives the split ──────────────────────────────────────────────

def test_the_i2v_anchor_is_re_pinned_for_pass_two():
    """_sample_chunk drops noise_mask from what it returns, so without this pass 2 runs
    unpinned and re-denoises the reference frame. anchor_shift wants that; this does not."""
    frames = 6
    clean = torch.arange(frames, dtype=torch.float32).view(1, 1, frames, 1, 1).repeat(1, 4, 1, 2, 2)
    mask = torch.ones_like(clean)
    mask[:, :, :1] = 0.0                       # one pinned anchor frame
    chunk = {"samples": clean, "noise_mask": mask}
    # A "state" where the anchor frame has been polluted (as a rewind re-noise would).
    state = {"samples": torch.full_like(clean, -99.0)}

    out = _node()._restore_pinned_prefix(state, chunk)
    assert out.get("noise_mask") is not None
    # The pinned frame is back to the clean anchor...
    assert torch.equal(out["samples"][:, :, :1], clean[:, :, :1])
    # ...and nothing else was touched.
    assert torch.equal(out["samples"][:, :, 1:], state["samples"][:, :, 1:])


def test_restore_is_a_no_op_on_a_t2v_scene():
    """No pinned prefix = no anchor to protect; the state must come back untouched."""
    clean = torch.zeros(1, 4, 6, 2, 2)
    state = {"samples": torch.full_like(clean, 3.0)}
    for chunk in ({"samples": clean, "noise_mask": torch.ones_like(clean)},
                  {"samples": clean}):
        assert _node()._restore_pinned_prefix(state, chunk) is state
