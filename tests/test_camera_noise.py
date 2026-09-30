"""Camera noise (camera_noise.py): a pan or zoom is written into the starting noise as the
same pattern travelling, and the noise stays unit Gaussian."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import camera_noise as cn  # noqa: E402


def _travel(move, t=9, h=12, w=20, c=3, seed=0):
    return cn.travel(move, c, t, h, w, torch.Generator().manual_seed(seed))


def test_pan_right_moves_the_picture_left_by_the_stated_share():
    move = cn.Move(pan_x=0.5)                    # half the frame width over the clip
    out = _travel(move, t=9, w=20)
    last = out[:, -1]                            # shifted by 10 cells
    assert torch.equal(last[:, :, :10], out[:, 0][:, :, 10:])
    assert not torch.equal(last[:, :, 10:], out[:, 0][:, :, :10])   # uncovered edge is fresh
    mid = out[:, 4]                              # halfway: 5 cells
    assert torch.equal(mid[:, :, :15], out[:, 0][:, :, 5:])


def test_pan_down_moves_the_picture_up():
    out = _travel(cn.Move(pan_y=0.5), t=9, h=12)
    assert torch.equal(out[:, -1][:, :6, :], out[:, 0][:, 6:, :])


def test_zoom_in_magnifies_the_focus_and_zoom_out_shrinks_it():
    zin = _travel(cn.Move(zoom=2.0), t=5, h=12, w=20)
    # at the end (scale 2) output cell (0, 0) copies frame-0 cell (3, 5): (0.5 - 6)/2 + 6 = 3.25
    assert torch.equal(zin[:, -1][:, 0, 0], zin[:, 0][:, 3, 5])
    assert torch.equal(zin[:, -1][:, 11, 19], zin[:, 0][:, 8, 14])   # 6 + 5.5/2 = 8.75, 10 + 9.5/2 = 14.75
    assert torch.equal(zin[:, -1][:, 6, 10], zin[:, 0][:, 6, 10])    # the focus itself stays
    zout = _travel(cn.Move(zoom=0.5), t=5, h=12, w=20)
    # zoomed out (scale 0.5) the picture shrinks toward the focus: output cell (6, 11) shows
    # frame-0 cell (7, 13), twice as far from the middle: (6.5 - 6) * 2 + 6 = 7, (11.5 - 10) * 2 + 10 = 13
    assert torch.equal(zout[:, -1][:, 6, 11], zout[:, 0][:, 7, 13])
    assert zout.shape == (3, 5, 12, 20)


def test_focus_point_moves_where_the_zoom_aims():
    left = _travel(cn.Move(zoom=2.0, focus_x=0.0), t=5)
    # aimed at the left edge, that edge column stays put while the rest stretches away from it
    assert torch.equal(left[:, -1][:, 6, 0], left[:, 0][:, 6, 0])
    assert torch.equal(left[:, -1][:, 6, 8], left[:, 0][:, 6, 4])    # (8.5)/2 = 4.25


def test_every_frame_is_unit_gaussian_and_the_blend_keeps_it():
    move = cn.Move(pan_x=0.3, zoom=1.4, amount=0.8)
    noise = torch.randn(1, 24, 9, 12, 20, generator=torch.Generator().manual_seed(1))
    out, note = cn.shape(noise, torch.zeros_like(noise), move, seed=5)
    assert note.startswith("Active")
    per_frame = out[0].std(dim=(0, 2, 3))
    assert per_frame.min() > 0.9 and per_frame.max() < 1.1
    assert abs(float(out.mean())) < 0.05


def test_the_travelling_pattern_is_really_shared_between_frames():
    """Line frame 8 up with frame 0 by the pan (6 cells) and they correlate ~amount squared;
    plain noise does not."""
    noise = torch.randn(1, 24, 9, 12, 20, generator=torch.Generator().manual_seed(2))
    out, _ = cn.shape(noise, torch.zeros_like(noise), cn.Move(pan_x=0.3, amount=0.8), seed=5)
    aligned = float((out[0, :, 8, :, :14] * out[0, :, 0, :, 6:]).mean())
    plain = float((noise[0, :, 8, :, :14] * noise[0, :, 0, :, 6:]).mean())
    assert aligned > 0.5 and abs(plain) < 0.1


def test_a_move_that_cannot_apply_is_left_alone_and_said():
    noise = torch.randn(1, 24, 9, 12, 20)
    empty = torch.zeros_like(noise)
    out, note = cn.shape(noise, empty, cn.Move(), seed=1)
    assert out is noise and note.startswith("Inactive") and "no move" in note
    out, note = cn.shape(noise, torch.ones_like(noise), cn.Move(pan_x=0.3), seed=1)
    assert out is noise and note.startswith("Inactive") and "picture" in note
    one = torch.randn(1, 24, 1, 12, 20)
    out, note = cn.shape(one, torch.zeros_like(one), cn.Move(pan_x=0.3), seed=1)
    assert out is one and note.startswith("Inactive")


def test_same_seed_same_move_is_reproducible_and_seeds_differ():
    noise = torch.randn(1, 24, 9, 12, 20)
    empty = torch.zeros_like(noise)
    a, _ = cn.shape(noise, empty, cn.Move(pan_x=0.3), seed=7)
    b, _ = cn.shape(noise, empty, cn.Move(pan_x=0.3), seed=7)
    c, _ = cn.shape(noise, empty, cn.Move(pan_x=0.3), seed=8)
    assert torch.equal(a, b) and not torch.equal(a, c)


class _Nested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors


def test_audio_noise_is_never_touched(monkeypatch):
    monkeypatch.setattr(sys.modules["comfy.nested_tensor"], "NestedTensor", _Nested, raising=False)
    video, audio = torch.randn(1, 24, 9, 12, 20), torch.randn(1, 32, 2, 40)
    noise = _Nested([video, audio])
    empty = _Nested([torch.zeros_like(video), torch.zeros_like(audio)])
    out, note = cn.shape(noise, empty, cn.Move(pan_x=0.3), seed=1)
    assert note.startswith("Active") and torch.equal(out.tensors[1], audio)
    assert not torch.equal(out.tensors[0], video)


def test_describe_reads_back_the_move():
    m = cn.Move(pan_x=-0.3, zoom=1.5, focus_x=0.2, amount=0.8)
    assert m.describe() == "pan left 0.30, zoom in x1.50 toward (0.20, 0.50), 80% carried"


def _adjacent_correlation(x):
    a, b = x[..., :-1], x[..., 1:]
    return float((a * b).mean() / (x.std() ** 2))


def test_zoom_in_makes_neighbours_alike_but_pan_and_zoom_out_do_not():
    """The cost is stated in the docstring and tooltip; this pins its size."""
    last = lambda m: _travel(m, t=5, h=48, w=84, c=8)[:, -1]
    assert abs(_adjacent_correlation(last(cn.Move(pan_x=0.5)))) < 0.05
    assert abs(_adjacent_correlation(last(cn.Move(zoom=0.5)))) < 0.05
    assert 0.2 < _adjacent_correlation(last(cn.Move(zoom=1.5))) < 0.5


def test_pan_is_clamped_and_nan_is_no_pan():
    assert cn.clamp_pan(7) == 1.0 and cn.clamp_pan(-7) == -1.0
    assert cn.clamp_pan(float("nan")) == 0.0 and cn.clamp_pan(0.25) == 0.25


def test_the_sampler_inputs_agree_with_the_editor_defaults():
    import samplers
    from movie_editor.backend import settings_card
    opt = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()["optional"]
    names = ["camera_noise", "camera_pan_x", "camera_pan_y", "camera_zoom",
             "camera_focus_x", "camera_focus_y", "camera_amount"]
    for name in names:
        assert opt[name][1]["default"] == settings_card._EDITOR_DEFAULTS[name], name
    assert opt["camera_noise"][1]["default"] == "off" and settings_card._EDITOR_DEFAULTS["camera_noise"] == "off"
