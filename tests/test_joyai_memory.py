"""Unit tests for the JoyAI-Echo cross-shot memory bank policy on the Scene Chain sampler
(_JoyAIMemoryBank). Covers the fixed-anchor pinning, rolling most-recent window, and the
max_size cap — the bookkeeping that drives which prior-shot frames get injected each scene.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Minimal comfy stubs so `import samplers` works without a full ComfyUI env.
for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402

Bank = samplers._JoyAIMemoryBank


def test_keeps_everything_under_capacity():
    b = Bank(max_size=7, num_fix=3)
    for i in range(5):
        b.add(i)
    assert b.frames() == [0, 1, 2, 3, 4]


def test_pins_fixed_anchors_and_rolls_recent():
    b = Bank(max_size=5, num_fix=2)
    for i in range(8):  # 0..7
        b.add(i)
    # First 2 (the opening shots) pinned forever; the rest is the most-recent window.
    frames = b.frames()
    assert len(frames) == 5
    assert frames[:2] == [0, 1]          # anchors never pruned
    assert frames[2:] == [5, 6, 7]       # rolling most-recent fills the remaining 3 slots


def test_no_fixed_anchors_is_pure_rolling():
    b = Bank(max_size=3, num_fix=0)
    for i in range(6):
        b.add(i)
    assert b.frames() == [3, 4, 5]


def test_num_fix_clamped_to_max_size():
    b = Bank(max_size=3, num_fix=10)   # nonsensical: more anchors than capacity
    for i in range(6):
        b.add(i)
    # num_fix clamps to max_size, so the first 3 pin and nothing rolls.
    assert b.frames() == [0, 1, 2]


def test_add_none_is_ignored():
    b = Bank(max_size=4, num_fix=1)
    b.add(0)
    b.add(None)
    b.add(1)
    assert b.frames() == [0, 1]


def test_frames_returns_a_copy():
    b = Bank(max_size=4, num_fix=1)
    b.add(0)
    out = b.frames()
    out.append(99)
    assert b.frames() == [0]   # internal state not mutated by caller
