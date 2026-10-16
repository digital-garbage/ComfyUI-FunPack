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


@needs_ffmpeg
def test_export_keeps_the_sound_when_the_first_clip_is_silent(comfy):
    make_clip(comfy.output / "quiet.mp4", 1.0, audio=False)
    make_clip(comfy.output / "loud.mp4", 1.0, audio=True)
    out = stitch.concat([{"filename": "quiet.mp4", "dur": 1.0}, {"filename": "loud.mp4", "dur": 1.0}])
    assert files.has_audio(str(comfy.temp / out["media"]["filename"]))


@needs_ffmpeg
def test_export_of_a_window_past_the_end_says_so_and_leaves_nothing(comfy):
    make_clip(comfy.output / "a.mp4", 1.0)
    with pytest.raises(stitch.RenderError, match="shorter than the window|past"):
        stitch.concat([{"filename": "a.mp4", "in": 5, "dur": 1.0}])
    with pytest.raises(stitch.RenderError, match="Clip 1"):
        stitch.concat([{"filename": "a.mp4", "in": 0, "dur": 3.0}])
    assert not os.listdir(comfy.temp)


@needs_ffmpeg
def test_two_exports_in_one_second_do_not_share_a_file(comfy):
    make_clip(comfy.output / "a.mp4", 1.0)
    first = stitch.concat([{"filename": "a.mp4", "dur": 1.0}])["media"]["filename"]
    second = stitch.concat([{"filename": "a.mp4", "dur": 1.0}])["media"]["filename"]
    assert first != second and (comfy.temp / first).is_file() and (comfy.temp / second).is_file()


def _audio_seconds(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=duration", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True).stdout.split()
    return float(out[0])


@needs_ffmpeg
def test_a_clip_whose_sound_is_shorter_than_its_picture_does_not_pull_later_sound_early(comfy):
    import subprocess
    short = comfy.output / "short.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(short)], check=True, capture_output=True)
    out = stitch.render(projects.Project(width=160, height=120),
                        [clip("short.mp4", dur=2.0), clip("short.mp4", dur=2.0), clip("short.mp4", dur=2.0)])
    assert _audio_seconds(comfy.temp / out["media"]["filename"]) == pytest.approx(6.0, abs=0.15)


@needs_ffmpeg
def test_an_odd_canvas_renders(comfy):
    make_clip(comfy.output / "a.mp4", 1.0, size="160x120")
    out = stitch.render(projects.Project(width=161, height=121), [clip("a.mp4", dur=1.0, w=161, h=121)])
    assert probe(comfy.temp / out["media"]["filename"])[1:3] == (160, 120)


@needs_ffmpeg
def test_export_of_clips_with_different_frame_rates_keeps_picture_and_sound_together(comfy):
    make_clip(comfy.output / "a.mp4", 2.0, rate=25)
    make_clip(comfy.output / "b.mp4", 2.0, rate=30)
    out = stitch.concat([{"filename": "a.mp4", "dur": 2.0}, {"filename": "b.mp4", "dur": 2.0}])
    path = comfy.temp / out["media"]["filename"]
    assert probe(path)[0] == pytest.approx(4.0, abs=0.25)
    assert _audio_seconds(path) == pytest.approx(4.0, abs=0.25)


@needs_ffmpeg
def test_a_render_says_what_it_left_out(comfy):
    make_clip(comfy.output / "a.mp4", 1.0)
    proj = projects.Project(width=160, height=120,
                            overlay_tracks=[{"id": "o1", "kind": "image", "media_ref": "aaaaaaaaaaaa",
                                             "start_sec": 0, "duration_sec": 1}],
                            audio_tracks=[{"id": "t1", "media_ref": "bbbbbbbbbbbb", "start_sec": 0}])
    out = stitch.render(proj, [clip("a.mp4", dur=1.0)])
    assert len(out["warnings"]) == 2 and "overlay" in out["warnings"][0] and "audio" in out["warnings"][1]


@needs_ffmpeg
def test_a_crossfade_after_a_hard_cut_or_a_gap_renders(comfy):
    make_clip(comfy.output / "a.mp4", 2.0, size="160x120")
    c = lambda **k: clip("a.mp4", dur=2.0, **k)            # noqa: E731
    for first in ({}, {"gap_after": 0.5}):
        out = stitch.render(projects.Project(width=160, height=120),
                            [c(**first), c(transition="crossfade", tdur=0.5), c()])
        assert probe(comfy.temp / out["media"]["filename"])[0] > 5.0


@needs_ffmpeg
def test_a_clip_whose_render_is_shorter_than_its_window_is_refused(comfy):
    make_clip(comfy.output / "a.mp4", 1.0)
    with pytest.raises(stitch.RenderError, match="Clip 1's render has"):
        stitch.render(projects.Project(width=160, height=120), [clip("a.mp4", dur=2.0)])


@needs_ffmpeg
def test_an_audio_lane_with_no_recorded_length_renders_as_long_as_its_file(comfy, monkeypatch):
    import subprocess
    from core import media
    wav = comfy.root / "song.wav"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=duration=3", str(wav)], check=True, capture_output=True)
    monkeypatch.setattr(media, "path_for", lambda ref: wav)
    monkeypatch.setattr(media, "is_id", lambda ref: True)
    proj = projects.Project(width=160, height=120, audio_tracks=[{"id": "t", "media_ref": "aaaaaaaaaaaa", "start_sec": 1.0}])
    assert stitch.graphics_duration(proj) == pytest.approx(4.0, abs=0.2)


def _video_seconds(path):
    import subprocess
    return float(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=duration", "-of", "default=nw=1:nk=1", str(path)],
                                capture_output=True, text=True).stdout.split()[0])


@needs_ffmpeg
def test_picture_and_sound_stay_together_when_a_render_runs_a_little_short(comfy):
    make_clip(comfy.output / "a.mp4", 1.6)
    c = lambda **k: clip("a.mp4", dur=2.0, **k)            # noqa: E731
    out = stitch.render(projects.Project(width=160, height=120), [c(), c(), c()])
    path = comfy.temp / out["media"]["filename"]
    assert _video_seconds(path) == pytest.approx(6.0, abs=0.1)
    assert _audio_seconds(path) == pytest.approx(6.0, abs=0.1)


@needs_ffmpeg
def test_an_audio_lane_whose_file_has_no_sound_is_left_out_and_said(comfy):
    make_clip(comfy.output / "quiet.mp4", 1.0, audio=False)
    proj = projects.Project(width=160, height=120, audio_tracks=[{"id": "t", "kind": "separated", "scene_id": "s1"}])
    out = stitch.render(proj, [clip("quiet.mp4", dur=1.0, scene_id="s1")])
    assert any("audio lane" in w for w in out["warnings"])


@needs_ffmpeg
def test_the_render_is_made_at_the_clips_own_size_not_the_projects(comfy):
    make_clip(comfy.output / "a.mp4", 1.0, size="160x120")
    make_clip(comfy.output / "b.mp4", 1.0, size="96x96")
    spec = lambda name, **k: clip(name, dur=1.0, w=100, h=100, **k)          # noqa: E731  (scene settings say 100x100)
    size = lambda proj, clips: probe(comfy.temp / stitch.render(proj, clips)["media"]["filename"])[1:3]   # noqa: E731
    clips = [spec("a.mp4", scene_id="s1"), spec("b.mp4", scene_id="s2")]
    assert size(projects.Project(width=100, height=100), clips) == (160, 120)              # the first clip
    assert size(projects.Project(width=100, height=100, export_size_from="s2"), clips) == (96, 96)
    assert size(projects.Project(width=100, height=100, export_size_from="project"), clips) == (100, 100)
    assert size(projects.Project(width=100, height=100, export_size_from="gone"), clips) == (160, 120)


def test_a_separated_lane_goes_quiet_with_a_clip_that_is_not_in_the_cut():
    proj = projects.Project(width=160, height=120, audio_tracks=[
        {"id": "a", "kind": "separated", "scene_id": "gone"}, {"id": "b", "kind": "separated", "scene_id": "here"},
        {"id": "c", "media_ref": "aaaaaaaaaaaa", "start_sec": 0}])
    assert [t["id"] for t in stitch._live_tracks(proj, {"here": {}})] == ["b", "c"]


def test_a_separated_lane_starts_where_its_clip_is_in_this_render_plus_its_slide(tmp_path, monkeypatch):
    f = tmp_path / "x.mp4"
    f.write_bytes(b"")
    monkeypatch.setattr(stitch.files, "has_audio", lambda p: True)
    proj = projects.Project(width=160, height=120, audio_tracks=[
        {"id": "a", "kind": "separated", "scene_id": "s2", "start_sec": 99, "offset_sec": 0.5},       # start_sec is the editor's guess, the render's own timeline wins
        {"id": "b", "kind": "separated", "scene_id": "s2", "offset_sec": -9}])
    clips = [{"scene_id": "s1", "dur": 2.0}, {"scene_id": "s2", "dur": 3.0}]
    by = {c["scene_id"]: c for c in clips}
    out = stitch._audio_tracks(proj, by, lambda c: str(f), stitch._clip_starts(clips))
    assert [round(t["start_sec"], 2) for t in out] == [2.5, 0.0]


def test_clip_starts_follow_the_renders_own_clock_gaps_and_overlapping_seams():
    clips = [{"scene_id": "a", "dur": 2.0, "gap_after": 1.0, "transition": "crossfade", "tdur": 0.5},
             {"scene_id": "b", "dur": 3.0, "transition": "crossfade", "tdur": 0.5},
             {"scene_id": "c", "dur": 1.0}]
    assert stitch._clip_starts(clips) == {"a": 0.0, "b": 2.5, "c": 5.0}
