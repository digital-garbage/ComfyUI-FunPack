"""A keyframe pin can sit at any pixel frame, not only the first or the last.

Upstream ``PackedLayout`` computes a pin's time coordinate in two branches and raises for
anything between them. Those two branches are the endpoints of one straight line — the video
time axis advances ``FRAME_RESCALE`` per PIXEL frame — so an interior pin is a different point
on an axis the model already uses, not a new kind of conditioning.

What is tested here:

* the linear rule reproduces upstream's own two branches (against the REAL ComfyUI when one is
  present, in a subprocess, because the suite stubs ``comfy``);
* the patch rewrites ONLY the condition rows' time column;
* an endpoint-only run never enters the rewrite, so existing behaviour is untouched;
* a changed upstream frame grid turns the feature OFF rather than mis-placing pins.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import minimax_h3 as h3

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _frames_for(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


class _FakePackedLayout:
    """Upstream's keyframe branch, faithfully — including the raise."""

    ROWS = 6                      # frame_rows, one latent frame's spatial grid

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t,
                 keyframes=None, refs=None, frame_count=None):
        segments = [("text", text_len)]
        pos = [torch.zeros(text_len, 3, dtype=torch.float64)]
        for kf in (keyframes or ()):
            pixel_index = kf["resolved_frame_index"]
            if pixel_index == 0:
                cond_t = float(text_len)
            elif frame_count is not None and pixel_index == frame_count - 1:
                cond_t = float(text_len) + sum(_spans(latent_t)) - FRAME_RESCALE
            else:
                raise ValueError("only first/last keyframe anchors are supported")
            g = torch.zeros(self.ROWS, 3, dtype=torch.float64)
            g[:, 0] = cond_t
            g[:, 1] = 1.0                     # a spatial column, to prove it is left alone
            segments.append(("cond", self.ROWS))
            pos.append(g)
        segments.append(("video", latent_t * self.ROWS))
        pos.append(torch.full((latent_t * self.ROWS, 3), 9.0, dtype=torch.float64))

        self.position_ids = torch.cat(pos)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        seg_abs, off = [], 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


@pytest.fixture
def upstream(monkeypatch):
    """Install a fake comfy.ldm.minimax.model and reset the patch cache."""
    mod = types.ModuleType("comfy.ldm.minimax.model")
    mod.PackedLayout = type("PackedLayout", (_FakePackedLayout,), {})
    mod.FRAME_PER_TOKEN = FRAME_PER_TOKEN
    mod.FRAME_RESCALE = FRAME_RESCALE
    mod._video_t_spans = _spans
    for name in ("comfy.ldm.minimax", "comfy.ldm.minimax.model"):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(h3._INTERIOR_PINS, "state", None)
    return mod


def _build(mod, keyframes, text_len=7, latent_t=6):
    return mod.PackedLayout(text_len, latent_t, 4, 4, 3,
                            keyframes=keyframes, frame_count=_frames_for(latent_t))


def _cond_t(layout):
    return [float(layout.position_ids[a, 0]) for a, _b, kind in layout.segments if kind == "cond"]


# ── the rule ────────────────────────────────────────────────────────────────

def test_the_rule_reproduces_both_upstream_branches(upstream):
    """The whole feature rests on this: first/last are two points on one line."""
    assert h3._linear_rule_matches_upstream()


def test_the_rule_is_linear_in_the_pixel_index():
    step = h3.keyframe_cond_t(7, 1) - h3.keyframe_cond_t(7, 0)
    assert step == pytest.approx(FRAME_RESCALE)
    assert h3.keyframe_cond_t(7, 12) == pytest.approx(7.0 + FRAME_RESCALE * 12)


def test_a_changed_frame_grid_disables_the_feature(upstream, capsys):
    """Drift must turn pins off, not move them. FunPack patches ComfyUI by name and has
    been silently broken by an upstream grid change before."""
    upstream.FRAME_RESCALE = 2.0
    assert h3.install_interior_keyframes() is False
    assert h3.keyframe_indices_supported(8, 18) is False
    assert "DISABLED" in capsys.readouterr().out


def test_a_comfy_without_h3_declines_quietly(monkeypatch, capsys):
    monkeypatch.setitem(h3._INTERIOR_PINS, "state", None)
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.model", None)
    assert h3.install_interior_keyframes() is False
    assert "unavailable" in capsys.readouterr().out


# ── the patch ───────────────────────────────────────────────────────────────

def test_an_interior_pin_is_refused_before_the_patch(upstream):
    with pytest.raises(ValueError):
        _build(upstream, [{"resolved_frame_index": 8, "latent": None}])


def test_an_interior_pin_lands_on_its_own_coordinate(upstream):
    assert h3.install_interior_keyframes() is True
    layout = _build(upstream, [{"resolved_frame_index": 8, "latent": None}])
    assert _cond_t(layout) == [pytest.approx(h3.keyframe_cond_t(7, 8))]


def test_the_pin_sits_between_the_two_endpoints(upstream):
    h3.install_interior_keyframes()
    first = _cond_t(_build(upstream, [{"resolved_frame_index": 0}]))[0]
    mid = _cond_t(_build(upstream, [{"resolved_frame_index": 8}]))[0]
    last = _cond_t(_build(upstream, [{"resolved_frame_index": 17}]))[0]
    assert first < mid < last


def test_only_the_time_column_is_rewritten(upstream):
    """The pin keeps the target's spatial grid — that is how it addresses a frame at all."""
    h3.install_interior_keyframes()
    layout = _build(upstream, [{"resolved_frame_index": 8}])
    start, stop = [(a, b) for a, b, k in layout.segments if k == "cond"][0]
    assert torch.all(layout.position_ids[start:stop, 1] == 1.0)
    assert torch.all(layout.position_ids[start:stop, 2] == 0.0)


def test_rows_outside_the_pin_are_untouched(upstream):
    h3.install_interior_keyframes()
    layout = _build(upstream, [{"resolved_frame_index": 8}])
    video = [(a, b) for a, b, k in layout.segments if k == "video"][0]
    assert torch.all(layout.position_ids[video[0]:video[1]] == 9.0)
    assert torch.all(layout.position_ids[:7] == 0.0)


def test_several_pins_each_get_their_own_coordinate(upstream):
    """Segments are emitted in `keyframes` order, so the k-th cond span is the k-th pin."""
    h3.install_interior_keyframes()
    layout = _build(upstream, [{"resolved_frame_index": 0},
                               {"resolved_frame_index": 8},
                               {"resolved_frame_index": 17}])
    assert _cond_t(layout) == [pytest.approx(h3.keyframe_cond_t(7, i)) for i in (0, 8, 17)]


def test_an_endpoint_only_run_is_bit_identical(upstream):
    """The patch must be invisible to every run that does not use it."""
    before = _build(upstream, [{"resolved_frame_index": 0},
                               {"resolved_frame_index": 17}]).position_ids.clone()
    h3.install_interior_keyframes()
    after = _build(upstream, [{"resolved_frame_index": 0},
                              {"resolved_frame_index": 17}]).position_ids
    assert torch.equal(before, after)


def test_a_pinless_run_is_bit_identical(upstream):
    before = _build(upstream, []).position_ids.clone()
    h3.install_interior_keyframes()
    assert torch.equal(before, _build(upstream, []).position_ids)


def test_patching_twice_does_not_stack(upstream):
    assert h3.install_interior_keyframes() is True
    first = upstream.PackedLayout.__init__
    h3._INTERIOR_PINS["state"] = None
    assert h3.install_interior_keyframes() is True
    assert upstream.PackedLayout.__init__ is first


# ── the gate callers use ────────────────────────────────────────────────────

def test_endpoints_never_need_the_patch(monkeypatch):
    """A first/last pin must keep working on a ComfyUI where the patch cannot install."""
    monkeypatch.setitem(h3._INTERIOR_PINS, "state", False)
    assert h3.keyframe_indices_supported(0, 18) is True
    assert h3.keyframe_indices_supported(17, 18) is True
    assert h3.keyframe_indices_supported(8, 18) is False


def test_out_of_range_indices_are_still_refused(upstream):
    h3.install_interior_keyframes()
    assert h3.keyframe_indices_supported(-1, 18) is False
    assert h3.keyframe_indices_supported(18, 18) is False
    assert h3.keyframe_indices_supported(17, 18) is True


# ── against the real ComfyUI, if one is here ────────────────────────────────

REAL_COMFY = Path.home() / "Documents" / "ComfyUI"


@pytest.mark.skipif(not (REAL_COMFY / "comfy" / "ldm" / "minimax" / "model.py").exists(),
                    reason="no local ComfyUI with MiniMax H3 to check against")
def test_the_rule_matches_the_real_upstream_layout():
    """The suite stubs `comfy`, so this re-derives the equality in a clean subprocess from
    upstream's own constants. This is the test that catches a real ComfyUI update."""
    script = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE, _video_t_spans\n"
        "RATE = 5.0 / 3.0\n"
        "assert abs(float(FRAME_RESCALE) - RATE) < 1e-9, 'FRAME_RESCALE moved'\n"
        "for latent_t in range(1, 40):\n"
        "    spans = _video_t_spans(latent_t)\n"
        "    frames = sum(FRAME_PER_TOKEN[k %% 5] for k in range(latent_t))\n"
        "    last = 7.0 + sum(spans) - FRAME_RESCALE\n"
        "    assert abs(7.0 - (7.0 + RATE * 0)) < 1e-9\n"
        "    assert abs(last - (7.0 + RATE * (frames - 1))) < 1e-9, (latent_t, last)\n"
        "print('OK')\n" % str(REAL_COMFY))
    env = dict(os.environ, PYTHONPATH="")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env=env, cwd=str(REAL_COMFY))
    assert "OK" in out.stdout, out.stderr[-2000:]
