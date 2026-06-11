import json

from movie_editor.backend.chain_layout import (
    latent_overlap_frames,
    latent_to_pixel_frame,
    layout_from_boundaries_json,
    layout_from_history_entry,
    scene_playback_layout,
    scene_start_pixels,
)


def test_latent_to_pixel_frame_time_scale():
    assert latent_to_pixel_frame(13, 8) == 97
    assert latent_to_pixel_frame(24, 8) == 185


def test_scene_starts_match_sampler_boundaries():
    # Two-scene chain: scene 2 starts at pixel 97 (full first scene), not 81 (97 - 16 overlap).
    starts = scene_start_pixels(
        3,
        num_frames_per_scene=97,
        frame_overlap=16,
        time_scale=8,
        boundaries=[
            {"between": [1], "pixel_frame": 97},
            {"between": [2], "pixel_frame": 185},
        ],
    )
    assert starts == [0, 97, 185]


def test_scene_starts_without_boundaries_uses_latent_math():
    latent_frames = 13
    latent_overlap = latent_overlap_frames(16, 8)
    assert latent_overlap == 2
    starts = scene_start_pixels(
        2,
        num_frames_per_scene=97,
        frame_overlap=16,
        time_scale=8,
        boundaries=None,
    )
    assert starts[0] == 0
    assert starts[1] == latent_to_pixel_frame(latent_frames, 8)


def test_layout_from_boundaries_json():
    data = {
        "scene_count": 2,
        "frames_per_scene": 13,
        "frames_per_scene_pixel": 97,
        "pixel_overlap": 16,
        "time_scale": 8,
        "boundaries": [{"between_scenes": [1, 2], "pixel_frame": 97, "latent_frame": 13}],
    }
    layout = layout_from_boundaries_json(data, fps=25.0)
    assert layout is not None
    assert layout[0]["in_sec"] == 0.0
    assert layout[1]["start_frame"] == 97
    assert abs(layout[1]["in_sec"] - 97 / 25.0) < 1e-6


def test_scene_playback_layout_fps():
    layout = scene_playback_layout(
        2,
        fps=24.0,
        num_frames_per_scene=97,
        frame_overlap=16,
        time_scale=8,
        boundaries=[{"between": [1], "pixel_frame": 97}],
    )
    assert len(layout) == 2
    assert layout[1]["in_sec"] == round(97 / 24.0, 6)


def test_three_scene_playback_in_sec():
    layout = scene_playback_layout(
        3,
        fps=25.0,
        num_frames_per_scene=97,
        frame_overlap=16,
        time_scale=8,
        boundaries=[
            {"between": [1], "pixel_frame": 97},
            {"between": [2], "pixel_frame": 185},
        ],
    )
    assert len(layout) == 3
    assert layout[2]["start_frame"] == 185
    assert abs(layout[2]["in_sec"] - 185 / 25.0) < 1e-6


def test_layout_from_history_picks_longest():
    two_scene = {
        "scene_count": 2,
        "frames_per_scene_pixel": 97,
        "pixel_overlap": 16,
        "time_scale": 8,
        "boundaries": [{"between_scenes": [1, 2], "pixel_frame": 97}],
    }
    three_scene = {
        "scene_count": 3,
        "frames_per_scene_pixel": 97,
        "pixel_overlap": 16,
        "time_scale": 8,
        "boundaries": [
            {"between_scenes": [1, 2], "pixel_frame": 97},
            {"between_scenes": [2, 3], "pixel_frame": 185},
        ],
    }
    entry = {
        "outputs": {
            "a": {"text": [json.dumps(two_scene)]},
            "b": {"text": [json.dumps(three_scene)]},
        }
    }
    layout = layout_from_history_entry(entry, fps=25.0)
    assert layout is not None
    assert len(layout) == 3
    assert layout[2]["start_frame"] == 185
