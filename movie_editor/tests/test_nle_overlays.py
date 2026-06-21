"""Overlay compositing filter builder."""

import os
import tempfile

from movie_editor.backend.nle_overlays import build_overlay_video_filter, sort_overlays_for_composite

try:
    import PIL  # noqa: F401
    from movie_editor.backend.nle_overlays import prepare_overlay_export, render_text_overlay_png
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def test_render_text_overlay_png():
    if not _HAS_PIL:
        import pytest
        pytest.skip("Pillow not installed")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "text.png")
        w, h = render_text_overlay_png(
            {"text": "Hello", "font_size": 36, "color": "#ffffff", "opacity": 1},
            768,
            512,
            path,
        )
        assert os.path.isfile(path)
        assert w > 0 and h > 0


def test_prepare_overlay_export_rasterizes_text():
    if not _HAS_PIL:
        import pytest
        pytest.skip("Pillow not installed")
    with tempfile.TemporaryDirectory() as tmp:
        export, paths = prepare_overlay_export(
            [{"id": "t1", "kind": "text", "text": "Hi", "start_sec": 0, "duration_sec": 2}],
            [{"id": "lane1"}],
            canvas_w=768,
            canvas_h=512,
            tempdir=tmp,
            resolve_image_path=lambda _ref: None,
        )
        assert len(export) == 1
        assert len(paths) == 1
        assert export[0]["kind"] == "image"
        assert os.path.isfile(paths[0])


def test_render_text_overlay_styled():
    """All rich-text style props render without error and grow the PNG vs. plain."""
    if not _HAS_PIL:
        import pytest
        pytest.skip("Pillow not installed")
    with tempfile.TemporaryDirectory() as tmp:
        plain = render_text_overlay_png(
            {"text": "Style\nme", "font_size": 40}, 768, 512, os.path.join(tmp, "plain.png"),
        )
        styled = render_text_overlay_png(
            {
                "text": "Style\nme", "font_size": 40, "color": "#ffcc00",
                "bold": True, "italic": True, "text_align": "left", "line_spacing": 1.6,
                "stroke_width": 4, "stroke_color": "#000000",
                "shadow": True, "shadow_color": "#222222",
                "bg_enabled": True, "bg_color": "#003355", "bg_opacity": 0.6, "opacity": 1.0,
            },
            768, 512, os.path.join(tmp, "styled.png"),
        )
        assert os.path.isfile(os.path.join(tmp, "styled.png"))
        # Background box + outline + shadow + italic shear must enlarge the raster.
        assert styled[0] >= plain[0] and styled[1] >= plain[1]


def test_resolve_fontfile_optional():
    from movie_editor.backend.nle_overlays import _resolve_fontfile
    assert _resolve_fontfile("system-ui") is not None or _resolve_fontfile("arial") is not None


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
    assert "drawtext" not in "\n".join(parts)
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


def test_prepare_overlay_text_flip_baked_into_png():
    if not _HAS_PIL:
        import pytest
        pytest.skip("Pillow not installed")
    with tempfile.TemporaryDirectory() as tmp:
        export, paths = prepare_overlay_export(
            [{"id": "t2", "kind": "text", "text": "Flip", "flip_h": True, "start_sec": 0, "duration_sec": 1}],
            [{"id": "lane1"}],
            canvas_w=768,
            canvas_h=512,
            tempdir=tmp,
            resolve_image_path=lambda _ref: None,
        )
        assert export[0]["flip_h"] is False
        assert os.path.isfile(paths[0])
