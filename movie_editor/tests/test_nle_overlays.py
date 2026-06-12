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


def test_build_overlay_image_filter():
    parts, final = build_overlay_video_filter(
        "[vbase]",
        [{"kind": "image", "start_sec": 0, "duration_sec": 5, "x": 0.5, "y": 0.5, "scale": 0.4, "opacity": 0.8}],
        canvas_w=1000,
        canvas_h=600,
        image_input_labels=["[3:v:0]"],
    )
    assert len(parts) == 2
    assert "scale=400" in parts[0]
    assert "overlay=" in parts[1]
    assert final.startswith("[vov")
