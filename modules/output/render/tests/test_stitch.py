"""The final cut. The filter graph is checked as text; the render and export are run on real ffmpeg."""

import os

import pytest

from modules.output.render.tests.clips import comfy, make_clip, needs_ffmpeg, probe  # noqa: F401
from core import projects
from modules.output.render import files, overlays, stitch


def clip(name, dur=2.0, **extra):
    return {"filename": name, "subfolder": "", "type": "output", "in": 0, "dur": dur,
            "fx": {}, "fps": 25, "w": 160, "h": 120, "transition": "", "tdur": 0, "volume": 1.0, **extra}


# --- the graph, as text ---------------------------------------------------------


def test_a_crossfade_overlaps_and_a_hard_cut_concatenates():
    a = clip("a.mp4", transition="crossfade", tdur=0.5)
    g, audio = stitch.build_filter([a, clip("b.mp4"), clip("c.mp4")])
    assert audio and "xfade=transition=fade:duration=0.500:offset=1.500" in g
    assert "acrossfade=d=0.500" in g
    assert "[vx1][v2]concat=n=2:v=1:a=0[vc2]" in g.replace("[v2]", "[v2]")


def test_a_seam_longer_than_a_clip_is_a_hard_cut_not_a_broken_graph():
    g, _ = stitch.build_filter([clip("a.mp4", transition="crossfade", tdur=5.0), clip("b.mp4")])
    assert "xfade" not in g and "concat=n=2:v=1:a=0[vc1]" in g


def test_a_gap_adds_black_and_silence_of_that_length():
    g, _ = stitch.build_filter([clip("a.mp4", gap_after=0.75), clip("b.mp4")])
    assert "color=c=black:s=160x120:r=25:d=0.750" in g and "anullsrc=r=48000:cl=stereo:d=0.750" in g


def test_a_clip_with_no_sound_contributes_silence_instead_of_sinking_the_render():
    g, _ = stitch.build_filter([clip("a.mp4", has_audio=False), clip("b.mp4")])
    assert "[0:a:0]" not in g and "anullsrc=r=48000:cl=stereo:d=2.000" in g and "[1:a:0]" in g


def test_audio_is_resampled_so_clips_of_different_rates_can_be_joined():
    g, _ = stitch.build_filter([clip("a.mp4"), clip("b.mp4")])
    assert g.count("sample_rates=48000") >= 2


def test_reversed_clip_reverses_its_sound_too_and_volume_applies():
    g, _ = stitch.build_filter([clip("a.mp4", fx={"reverse": True}, volume=0.5)])
    assert "areverse,volume=0.500" in g


def test_keep_original_off_drops_the_clips_sound_and_extra_tracks_still_mix():
    g, has_audio = stitch.build_filter([clip("a.mp4")], keep_original=False)
    assert not has_audio and "[a0]" not in g
    g, has_audio = stitch.build_filter([clip("a.mp4")], [{"start_sec": 1.5, "volume": 0.5}],
                                       keep_original=False, base_input=1)
    assert has_audio and "[1:a:0]" in g and "adelay=1500|1500" in g and "volume=0.500" in g


def test_a_reverse_that_cannot_run_names_the_clip():
    with pytest.raises(stitch.RenderError, match="Clip 2: Reverse needs every frame"):
        stitch.build_filter([clip("a.mp4"), clip("b.mp4", dur=90.0, fx={"reverse": True})])


def test_no_clips_and_no_canvas_is_refused():
    with pytest.raises(stitch.RenderError, match="nothing to render"):
        stitch.build_filter([])


def test_junk_clip_numbers_do_not_crash_the_graph():
    g, _ = stitch.build_filter([clip("a.mp4", dur="x", tdur=float("nan"), volume=None, w="wide", fps=None)])
    assert "[0:v:0]" in g


# --- overlays ---------------------------------------------------------------------


def test_each_overlay_reads_its_own_picture_even_when_an_earlier_one_is_not_drawn():
    ovs = [{"kind": "image", "duration_sec": 0}, {"kind": "image", "duration_sec": 2, "start_sec": 1, "x": 0.25}]
    lines, final = overlays.composite("[vbase]", ovs, canvas_w=160, image_labels=["[5:v:0]", "[6:v:0]"])
    assert final == "[vov1]" and "[6:v:0]scale=" in lines[0] and "[5:v:0]" not in "".join(lines)
    assert "enable='between(t,1.000,3.000)'" in lines[-1] and "overlay=x=0.250000*W-w/2" in lines[-1]


def test_overlays_stack_by_lane_then_start():
    lanes = [{"id": "low"}, {"id": "high"}]
    ovs = [{"id": 1, "lane_id": "high", "start_sec": 0}, {"id": 2, "lane_id": "low", "start_sec": 5},
           {"id": 3, "lane_id": "low", "start_sec": 1}]
    assert [o["id"] for o in overlays.in_stacking_order(ovs, lanes)] == [3, 2, 1]


# --- real ffmpeg ------------------------------------------------------------------


@needs_ffmpeg
def test_two_clips_with_a_crossfade_and_effects_render_to_the_right_length_and_size(comfy):
    make_clip(comfy.output / "a.mp4", 2.0, size="160x120")
    make_clip(comfy.output / "b.mp4", 2.0, size="320x180")           # a different native size
    proj = projects.Project(width=160, height=120, frame_rate=25)
    out = stitch.render(proj, [clip("a.mp4", transition="crossfade", tdur=0.5, fx={"flip_h": True, "blur": 0.1}),
                               clip("b.mp4", fx={"fit": "fill", "fade_out": 0.3})])
    assert out["clips"] == 2 and out["media"]["type"] == "temp"
    dur, w, h, audio = probe(comfy.temp / out["media"]["filename"])
    assert (w, h) == (160, 120) and audio
    assert dur == pytest.approx(3.5, abs=0.15)                       # 2 + 2 - the 0.5 s overlap


@needs_ffmpeg
def test_a_silent_clip_among_clips_with_sound_still_renders(comfy):
    make_clip(comfy.output / "a.mp4", 1.0, audio=False)
    make_clip(comfy.output / "b.mp4", 1.0)
    out = stitch.render(projects.Project(width=160, height=120), [clip("a.mp4", dur=1.0), clip("b.mp4", dur=1.0)])
    dur, _w, _h, audio = probe(comfy.temp / out["media"]["filename"])
    assert audio and dur == pytest.approx(2.0, abs=0.15)


@needs_ffmpeg
def test_a_trim_window_and_a_gap_are_honoured(comfy):
    make_clip(comfy.output / "a.mp4", 4.0)
    out = stitch.render(projects.Project(width=160, height=120),
                        [clip("a.mp4", dur=1.0, **{"in": 1.5}, gap_after=0.5), clip("a.mp4", dur=1.0)])
    dur, *_ = probe(comfy.temp / out["media"]["filename"])
    assert dur == pytest.approx(2.5, abs=0.15)


@needs_ffmpeg
def test_a_reversed_clip_renders(comfy):
    make_clip(comfy.output / "a.mp4", 1.0)
    out = stitch.render(projects.Project(width=160, height=120), [clip("a.mp4", dur=1.0, fx={"reverse": True})])
    assert probe(comfy.temp / out["media"]["filename"])[0] == pytest.approx(1.0, abs=0.15)


@needs_ffmpeg
def test_overlays_and_audio_lanes_render_with_no_clips_at_all(comfy):
    from core import media
    wav = comfy.root / "tone.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=330:duration=2", str(wav)],
                   check=True, capture_output=True)
    entry = media.save_upload("tone.wav", wav.read_bytes())
    proj = projects.Project(width=160, height=120, frame_rate=25, audio_tracks=[
        {"id": "a1", "kind": "overlay", "media_ref": entry["id"], "start_sec": 0.5, "volume": 1.0,
         "source_in_sec": 0, "source_dur": 1.5}],
        overlay_lanes=[{"id": "l1"}],
        overlay_tracks=[{"id": "t1", "lane_id": "l1", "kind": "text", "text": "hello", "start_sec": 0,
                         "duration_sec": 1.5, "x": 0.5, "y": 0.5, "font_size": 24, "color": "#ffcc00",
                         "bold": True, "italic": True, "shadow": True, "bg_enabled": True}])
    out = stitch.render(proj, [])
    dur, w, h, audio = probe(comfy.temp / out["media"]["filename"])
    assert (w, h) == (160, 120) and audio and dur == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_nothing_to_render_says_so(comfy):
    with pytest.raises(stitch.RenderError, match="Nothing to render"):
        stitch.render(projects.Project(), [])


@needs_ffmpeg
def test_a_missing_clip_file_is_a_sentence_not_a_traceback(comfy):
    with pytest.raises(stitch.RenderError, match="not on disk"):
        stitch.render(projects.Project(), [clip("gone.mp4")])


@needs_ffmpeg
def test_a_clip_name_that_climbs_out_of_the_output_folder_is_not_found(comfy):
    (comfy.root / "secret.mp4").write_bytes(b"x")
    with pytest.raises(stitch.RenderError):
        stitch.render(projects.Project(), [{**clip("secret.mp4"), "subfolder": ".."}])


@needs_ffmpeg
def test_export_one_clip_trims_and_several_hard_cut(comfy):
    make_clip(comfy.output / "a.mp4", 3.0)
    one = stitch.concat([{"filename": "a.mp4", "in": 1.0, "dur": 1.0}])
    assert probe(comfy.temp / one["media"]["filename"])[0] == pytest.approx(1.0, abs=0.15)
    two = stitch.concat([{"filename": "a.mp4", "in": 0, "dur": 1.0}, {"filename": "a.mp4", "in": 1.0, "dur": 1.0}])
    assert two["clips"] == 2 and probe(comfy.temp / two["media"]["filename"])[0] == pytest.approx(2.0, abs=0.2)
    assert not [p for p in os.listdir(comfy.temp) if p.startswith(("funpack_seg_", "funpack_concat_"))]
    with pytest.raises(stitch.RenderError):
        stitch.concat([])


@needs_ffmpeg
def test_a_separated_audio_lane_plays_the_clips_own_sound_from_its_file(comfy):
    make_clip(comfy.output / "a.mp4", 2.0)
    proj = projects.Project(width=160, height=120, keep_original_audio=False, audio_tracks=[
        {"id": "s", "kind": "separated", "scene_id": "sc1", "start_sec": 0}])
    out = stitch.render(proj, [clip("a.mp4", dur=2.0, scene_id="sc1")])
    assert probe(comfy.temp / out["media"]["filename"])[3] is True
