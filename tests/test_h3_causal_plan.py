"""The chunk plan: which packed rows each chunk owns.

The plan is the join between our rollout and upstream's packed sequence, so what these tests
guard is agreement rather than any behaviour of our own. A plan that tiles the sequence wrongly
does not raise — it hands the model rows from the wrong moment of the clip, and the only
symptom is a video that is subtly incoherent.

The suite stubs `comfy`, so the plan is exercised here against a fake layout with upstream's
segment structure, and against the REAL PackedLayout in a clean subprocess at the bottom.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_causal as hc  # noqa: E402

TEXT_LEN, LATENT_T, AUDIO_T, FRAME_ROWS = 120, 57, 320, 6


class _Layout:
    """Upstream's segment structure, without upstream. Order is the contract: the prefix
    first, then the target audio, then the target video."""

    def __init__(self, prefix=(("text", TEXT_LEN),)):
        segments, row = [], 0
        for kind, n in prefix:
            segments.append((row, row + n, kind))
            row += n
        segments.append((row, row + AUDIO_T * 2, "audio"))
        row += AUDIO_T * 2
        segments.append((row, row + LATENT_T * FRAME_ROWS, "video"))
        self.seq_len = row + LATENT_T * FRAME_ROWS
        self.segments = segments


def _plan(prefix=(("text", TEXT_LEN),)):
    return hc.build_plan(_Layout(prefix), LATENT_T, AUDIO_T)


def _all_rows(plan):
    return torch.cat([plan.prefix_rows] + [plan.chunk(i)[0] for i in range(plan.n_chunks)])


# ── the cut ─────────────────────────────────────────────────────────────────

def test_a_real_clip_cuts_into_the_expected_number_of_chunks():
    """192 pixel frames is 57 video latents: eleven 17-frame chunks and a 5-frame tail."""
    assert _plan().n_chunks == 12


def test_the_chunks_and_the_prefix_tile_the_whole_sequence_exactly():
    """No row generated twice, none missed. An overlap would denoise a moment twice from two
    different contexts; a gap would leave noise in the finished clip."""
    plan = _plan()
    assert sorted(_all_rows(plan).tolist()) == list(range(plan.layout.seq_len))


def test_the_prefix_is_everything_before_the_target_streams():
    plan = _plan()
    assert [k for _, _, k in plan.prefix_runs] == ["text"]
    assert plan.prefix_rows.tolist() == list(range(TEXT_LEN))


def test_a_chunk_packs_its_audio_before_its_video():
    """Upstream packs the target audio segment before the target video one, and the final
    layer reads both back by contiguous slice."""
    assert [k for _, _, k in _plan().chunk(3)[1]] == ["audio", "video"]


def test_a_chunks_audio_rows_are_two_spans_not_one():
    """`pack_audio` is channel-major over the WHOLE clip — all of the left channel, then all
    of the right — so a chunk owns a slice of each, not one contiguous run."""
    rows, (audio_run, _) = _plan().chunk(2)
    audio = rows[audio_run[0]:audio_run[1]].tolist()
    half = len(audio) // 2
    left, right = audio[:half], audio[half:]
    assert right[0] - left[0] == AUDIO_T
    assert [r - left[0] for r in left] == [r - right[0] for r in right]


def test_the_video_rows_of_a_chunk_are_its_own_latent_frames():
    plan = _plan()
    v_start, v_stop, _, _ = plan.bounds[4]
    _, (_, video_run) = plan.chunk(4)
    assert video_run[1] - video_run[0] == (v_stop - v_start) * FRAME_ROWS


def test_the_tail_chunk_owns_whatever_audio_is_left():
    """The clip's audio must be covered exactly — a soundtrack that stops short of the
    picture is the failure this rules out."""
    plan = _plan()
    assert plan.bounds[-1][3] == AUDIO_T


def test_media_chunk_zero_is_cache_chunk_two():
    """The prompt holds 0 and the conditioning 1, so the media chunks start at 2."""
    assert hc.ChunkPlan.cache_index(0) == 2
    assert hc.ChunkPlan.cache_index(5) == 7


def test_the_prompt_and_the_conditioning_are_separate_cache_chunks():
    """They are not the same kind of context. The prompt is every moment of the clip; the
    anchor is one moment of it, and pinning it into a 3-block context makes the model compose
    from it instead of continuing."""
    plan = _plan((("text", TEXT_LEN), ("cond", 6), ("ref_img", 12)))
    assert [k for _, _, k in plan.text_runs] == ["text"]
    assert [k for _, _, k in plan.cond_runs] == ["cond", "ref_img"]
    assert plan.text_rows.tolist() + plan.cond_rows.tolist() == plan.prefix_rows.tolist()


def test_the_conditioning_runs_are_local_to_their_own_sequence():
    """They are prefilled as a sequence of their own, so a run still starting at the text's
    row count would write past the end of the buffer."""
    plan = _plan((("text", TEXT_LEN), ("cond", 6)))
    assert plan.cond_runs == [(0, 6, "cond")]


def test_a_clip_with_no_conditioning_still_reserves_the_slot():
    """An index that moved with the presence of an anchor would make `sink` mean a different
    thing on a t2v run than on an i2v one."""
    plan = _plan()
    assert plan.cond_rows.numel() == 0
    assert hc.ChunkPlan.cache_index(0) == 2


# ── conditioning rows keep upstream's layout ────────────────────────────────

def test_reference_rows_land_in_the_prefix_not_in_a_media_chunk():
    """This is what makes r2v work here: the reference keeps the rows upstream packed for it
    instead of being re-packed into a chunk that has no place for them."""
    plan = _plan((("text", TEXT_LEN), ("ref_img", 90)))
    assert [k for _, _, k in plan.prefix_runs] == ["text", "ref_img"]
    for i in range(plan.n_chunks):
        assert {k for _, _, k in plan.chunk(i)[1]} == {"audio", "video"}


def test_the_tiling_still_holds_with_conditioning_rows_present():
    plan = _plan((("text", TEXT_LEN), ("cond", 6), ("ref_audio", 40)))
    assert sorted(_all_rows(plan).tolist()) == list(range(plan.layout.seq_len))


def test_a_clip_whose_streams_disagree_about_its_length_is_refused():
    """A chunk with no audio rows is a StopIteration fifty blocks deep, because the final
    layer reads one contiguous span of each stream. Say what is actually wrong instead."""
    layout = _Layout()
    with pytest.raises(hc.CacheError, match="does not cover the picture"):
        hc.ChunkPlan(layout, hc.chunk_bounds(LATENT_T, 12), 12)


def test_a_layout_with_no_target_streams_is_refused():
    class Bare:
        segments = [(0, 10, "text")]

    with pytest.raises(hc.CacheError):
        hc.ChunkPlan(Bare(), [(0, 1, 0, 1)], 1)


# ── against the real ComfyUI, if one is here ────────────────────────────────

REAL_COMFY = Path.home() / "Documents" / "ComfyUI"


@pytest.mark.skipif(not (REAL_COMFY / "comfy" / "ldm" / "minimax" / "model.py").exists(),
                    reason="no local ComfyUI with MiniMax H3 to check against")
def test_the_plan_and_both_lanes_agree_with_the_real_model():
    """The one that catches a ComfyUI update moving the layout out from under us. It also
    drives the REAL DiT at toy width through both lanes, which is where the dense-lane safety
    property is actually checked: with no cache passed, the causal model IS the stock one."""
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "_h3_causal_upstream_check.py"),
         str(root), str(REAL_COMFY)],
        capture_output=True, text=True, cwd=str(REAL_COMFY),
        env=dict(os.environ, PYTHONPATH=""))
    assert "OK" in out.stdout, out.stderr[-3000:]
