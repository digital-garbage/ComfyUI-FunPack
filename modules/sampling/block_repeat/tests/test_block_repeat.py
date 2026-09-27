"""Block repeat on a real H3 forward."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, patcher=None, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    return FunPackLoadModifiers.execute(
        patcher or tiny.patcher, {"block_repeat": {"enabled": True, **values}}).result


def _count_block_calls(tiny, patched, **run):
    """How many times each real block ran during one forward."""
    counts = {}
    handles = []
    for i, block in enumerate(tiny.patcher.model.diffusion_model.blocks):
        handles.append(block.register_forward_pre_hook(
            lambda m, a, i=i: counts.__setitem__(i, counts.get(i, 0) + 1)))
    try:
        tiny.run(patched, **run)
    finally:
        for h in handles:
            h.remove()
    return counts


def test_each_runs_every_named_block_times_plus_one(tiny_h3):
    patched, status = _load(tiny_h3, blocks="1-2", times=2)
    assert "block_repeat: blocks 1,2 run 3x each" in status
    assert _count_block_calls(tiny_h3, patched) == {0: 1, 1: 3, 2: 3, 3: 1}


def test_span_loops_the_range_and_runs_each_block_once_per_loop(tiny_h3):
    patched, status = _load(tiny_h3, blocks="1-2", shape="span", times=1)
    assert "blocks 1-2 loop 2x as one span" in status
    assert _count_block_calls(tiny_h3, patched) == {0: 1, 1: 2, 2: 2, 3: 1}


def test_span_output_equals_running_the_span_twice_by_hand(tiny_h3):
    """The loop really feeds block 2's output back into block 1."""
    patched, _ = _load(tiny_h3, blocks="1-2", shape="span")
    looped = tiny_h3.run(patched)
    # By hand: blocks 0,1,2,1,2,3 in order.
    from core import dit_hooks
    manual = tiny_h3.patcher.clone()
    real = tiny_h3.patcher.model.diffusion_model.blocks

    def twice(args, extra):
        h = args["img"]
        for i in (1, 2, 1, 2):
            h = real[i](h, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                        transformer_options=args["transformer_options"])
        return {"img": h}

    dit_hooks.add_block_hook(manual, "t", 1, twice)
    dit_hooks.add_block_hook(manual, "t", 2, lambda a, e: {"img": a["img"]})
    assert torch.allclose(looped[0], tiny_h3.run(manual)[0])


def test_a_scattered_list_is_refused_for_span(tiny_h3):
    _p, status = _load(tiny_h3, blocks="0,2", shape="span")
    assert "block_repeat: failed to install" in status


def test_step_window_repeats_only_on_the_last_steps(tiny_h3):
    patched, _ = _load(tiny_h3, blocks="1", steps=1)
    sched = torch.tensor([1.0, 0.7, 0.3, 0.0])
    early = _count_block_calls(tiny_h3, patched, sigma=0.7, sigmas=sched)
    last = _count_block_calls(tiny_h3, patched, sigma=0.3, sigmas=sched)
    assert early[1] == 1 and last[1] == 2


def test_video_only_leaves_audio_as_a_single_pass_would(tiny_h3):
    """Repeating the LAST block with video only: the audio branch must match a
    run with no repeat at all, the video branch must not."""
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3, blocks="3", video_only=True)
    out = tiny_h3.run(patched)
    assert torch.allclose(out[1], base[1], atol=1e-6), "audio got the repeat"
    assert not torch.allclose(out[0], base[0]), "video did not"


def test_video_only_span_leaves_audio_as_a_single_pass_would(tiny_h3):
    base = tiny_h3.run()
    patched, _ = _load(tiny_h3, blocks="2-3", shape="span", video_only=True)
    out = tiny_h3.run(patched)
    assert torch.allclose(out[1], base[1], atol=1e-6), "audio got the loop"
    assert not torch.allclose(out[0], base[0])
