"""Overlay compositing filter builder."""

from movie_editor.backend.nle_overlays import build_overlay_video_filter


def test_build_overlay_text_filter():
    parts, final = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "text", "text": "Hello", "start_sec": 1, "duration_sec": 2, "x": 0.5, "y": 0.5, "font_size": 36}],
        canvas_w=768,
        canvas_h=512,
        image_input_labels=[],
    )
    assert len(parts) == 1
    assert "drawtext" in parts[0]
    assert "Hello" in parts[0]
    assert "between(t,1.000,3.000)" in parts[0]
    assert final.startswith("[vov")


def test_build_overlay_text_with_font():
    parts, _ = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "text", "text": "Hi", "font_family": "system-ui", "start_sec": 0, "duration_sec": 1}],
        canvas_w=768,
        canvas_h=512,
        image_input_labels=[],
    )
    assert "drawtext" in parts[0]
    assert "fontfile=" not in parts[0]


def test_resolve_fontfile_optional():
    from movie_editor.backend.nle_overlays import _resolve_fontfile
    assert _resolve_fontfile("system-ui") is None
    assert _resolve_fontfile(None) is None


def test_sort_overlays_by_lane():
    from movie_editor.backend.nle_overlays import sort_overlays_for_composite
    lanes = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    tracks = [
        {"id": "1", "lane_id": "c", "start_sec": 0},
        {"id": "2", "lane_id": "a", "start_sec": 1},
        {"id": "3", "lane_id": "b", "start_sec": 0},
    ]
    out = sort_overlays_for_composite(tracks, lanes)
    assert [t["id"] for t in out] == ["2", "3", "1"]


def test_build_overlay_image_filter():
    parts, final = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "image", "start_sec": 0, "duration_sec": 5, "x": 0.5, "y": 0.5, "width_px": 400, "opacity": 0.8}],
        canvas_w=1000,
        canvas_h=600,
        image_input_labels=["[3:v:0]"],
    )
    assert len(parts) == 2
    assert "scale=400:-1" in parts[0]
    assert "overlay=" in parts[1]
    assert final.startswith("[vov")


def test_build_overlay_image_legacy_scale():
    parts, _ = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "image", "start_sec": 0, "duration_sec": 5, "scale": 0.4}],
        canvas_w=1000,
        canvas_h=600,
        image_input_labels=["[3:v:0]"],
    )
    assert "scale=400:-1" in parts[0]


def test_build_overlay_image_flip():
    parts, _ = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "image", "start_sec": 0, "duration_sec": 2, "width_px": 200, "flip_h": True, "flip_v": True}],
        canvas_w=800,
        canvas_h=600,
        image_input_labels=["[2:v:0]"],
    )
    joined = "\n".join(parts)
    assert "hflip" in joined
    assert "vflip" in joined


def test_build_overlay_text_flip():
    parts, _ = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "text", "text": "Hi", "start_sec": 0, "duration_sec": 1, "flip_h": True}],
        canvas_w=768,
        canvas_h=512,
        image_input_labels=[],
    )
    joined = "\n".join(parts)
    assert "drawtext" in joined
    assert "hflip" in joined
