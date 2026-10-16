"""Clip effects as ffmpeg filters: the order matters, and junk must read as 'not set'."""

import pytest

from modules.output.render import effects


def test_geometry_runs_flips_then_crop_then_fit():
    f = effects.geometry({"flip_h": True, "flip_v": True, "crop_inset": 0.1}, 320, 180)
    assert f[:2] == ["hflip", "vflip"] and f[2].startswith("crop=trunc(") and f[3].startswith("scale=320:180")
    assert f[-1] == "pad=320:180:-1:-1:color=black"


def test_fill_crops_to_the_canvas_instead_of_letterboxing():
    assert effects.geometry({"fit": "fill"}, 320, 180)[-2:] == [
        "scale=320:180:force_original_aspect_ratio=increase", "crop=320:180"]


def test_crop_is_clamped_and_even():
    assert effects.crop_inset({"crop_inset": 5}) == 0.4
    assert effects.crop_inset({"crop_inset": -1}) == 0.0
    assert effects.crop_inset({"crop_inset": "x"}) == 0.0
    assert "/2)*2" in effects.geometry({"crop_inset": 0.1}, 100, 100)[0]


@pytest.mark.parametrize("junk", ["abc", None, float("nan"), float("inf"), [], {}])
def test_a_junk_effect_value_is_not_set_not_a_crash(junk):
    fx = {"blur": junk, "fade_in": junk, "fade_out": junk, "zoom": "in", "zoom_ratio": junk,
          "zoom_frames": junk, "zoom_start_frame": junk, "crop_inset": junk}
    vf = effects.clip_filters(fx, 160, 120, 25.0, 2.0)
    assert not any(x.startswith(("gblur", "fade")) for x in vf)
    assert any(x.startswith("zoompan") for x in vf)       # the zoom itself still runs on defaults


def test_filters_order_reverse_geometry_fps_zoom_blur_fade():
    vf = effects.clip_filters({"reverse": True, "zoom": "out", "blur": 0.1, "fade_in": 0.5, "fade_out": 0.5},
                              160, 120, 25.0, 2.0)
    names = [x.split("=")[0] for x in vf]
    assert names.index("reverse") < names.index("scale") < names.index("fps") < names.index("zoompan") \
        < names.index("gblur") < names.index("fade")
    assert vf[-2:] == ["format=yuv420p", "setsar=1"]
    assert "fade=t=out:st=1.500:d=0.500" in vf


def test_reverse_too_long_is_refused_in_words_not_rendered_forwards():
    with pytest.raises(ValueError, match="every frame in memory"):
        effects.clip_filters({"reverse": True}, 160, 120, 25.0, 60.0)
    assert effects.reverse_refusal(10, 25) is None


def test_the_zoom_ramp_clamps_to_the_clip_and_matches_its_expression():
    ratio, start, length = effects.zoom_params({"zoom_start_frame": 999, "zoom_frames": 999, "zoom_ratio": 9}, 40)
    assert (ratio, start, length) == (0.5, 39, 1)
    assert effects.zoom_scale_at("in", {}, 0, 40) == 1.0
    assert effects.zoom_scale_at("in", {}, 39, 40) == pytest.approx(1.15)
    assert effects.zoom_scale_at("out", {}, 0, 40) == pytest.approx(1.15)
    assert effects.zoom_scale_at("none", {}, 5, 40) == 1.0
    assert "if(lt(on," in effects.zoompan_z("in", {}, 40)
