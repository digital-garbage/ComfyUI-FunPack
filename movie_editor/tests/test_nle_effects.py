import shutil
import subprocess

import pytest

from movie_editor.backend.nle_effects import (
    crop_inset_fraction,
    geometry_filters,
    zoom_scale_at_frame,
    zoompan_z_expr,
)


def test_zoom_in_ramp_window():
    fx = {"zoom_ratio": 0.2, "zoom_start_frame": 10, "zoom_frames": 20}
    assert zoom_scale_at_frame("in", fx, 0, 100) == 1.0
    assert zoom_scale_at_frame("in", fx, 10, 100) == 1.0
    assert abs(zoom_scale_at_frame("in", fx, 20, 100) - 1.1) < 1e-6
    assert abs(zoom_scale_at_frame("in", fx, 30, 100) - 1.2) < 1e-6
    assert abs(zoom_scale_at_frame("in", fx, 50, 100) - 1.2) < 1e-6


def test_zoom_out_ramp_window():
    fx = {"zoom_ratio": 0.2, "zoom_start_frame": 0, "zoom_frames": 10}
    assert abs(zoom_scale_at_frame("out", fx, 0, 50) - 1.2) < 1e-6
    assert abs(zoom_scale_at_frame("out", fx, 5, 50) - 1.1) < 1e-6
    assert abs(zoom_scale_at_frame("out", fx, 10, 50) - 1.0) < 1e-6


def test_zoompan_expr_contains_window():
    fx = {"zoom_ratio": 0.15, "zoom_start_frame": 5, "zoom_frames": 15}
    z = zoompan_z_expr("in", fx, 80)
    assert "if(lt(on,5)" in z
    assert "if(lt(on,20)" in z


# ── clip geometry: flip / crop / fit ──────────────────────────────────────────

def test_no_effects_is_the_plain_letterbox_chain():
    """The default must stay byte-identical to the chain that predates these effects —
    every existing project renders through this path."""
    assert geometry_filters({}, 768, 768) == [
        "scale=768:768:force_original_aspect_ratio=decrease",
        "pad=768:768:-1:-1:color=black",
    ]
    assert geometry_filters(None, 768, 768) == geometry_filters({}, 768, 768)


def test_flips_precede_the_fit():
    assert geometry_filters({"flip_h": True}, 640, 480)[0] == "hflip"
    assert geometry_filters({"flip_v": True}, 640, 480)[0] == "vflip"
    both = geometry_filters({"flip_h": True, "flip_v": True}, 640, 480)
    assert both[:2] == ["hflip", "vflip"]


def test_fill_crops_instead_of_padding():
    out = geometry_filters({"fit": "fill"}, 1280, 720)
    assert "scale=1280:720:force_original_aspect_ratio=increase" in out
    assert "crop=1280:720" in out
    assert not any(f.startswith("pad=") for f in out)


def test_crop_trims_the_source_before_fitting():
    out = geometry_filters({"crop_inset": 0.1}, 768, 768)
    assert out[0].startswith("crop=trunc(iw*0.800000/2)*2")
    assert out.index(out[0]) < out.index("scale=768:768:force_original_aspect_ratio=decrease")


def test_crop_dimensions_are_even():
    """Odd intermediate sizes are rejected downstream by yuv420p chroma subsampling."""
    for f in geometry_filters({"crop_inset": 0.13}, 768, 768):
        if f.startswith("crop=trunc"):
            assert f.count("/2)*2") == 2


@pytest.mark.parametrize("raw,expect", [
    (None, 0.0), ({}, 0.0), ({"crop_inset": 0}, 0.0), ({"crop_inset": -1}, 0.0),
    ({"crop_inset": "junk"}, 0.0), ({"crop_inset": float("nan")}, 0.0),
    ({"crop_inset": 9.9}, 0.4), ({"crop_inset": 0.25}, 0.25),
])
def test_crop_fraction_is_clamped(raw, expect):
    assert crop_inset_fraction(raw) == expect


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("fx", [
    {}, {"flip_h": True}, {"flip_v": True}, {"flip_h": True, "flip_v": True},
    {"fit": "fill"}, {"crop_inset": 0.1}, {"crop_inset": 0.4, "flip_h": True, "fit": "fill"},
])
def test_ffmpeg_accepts_every_chain(fx, tmp_path):
    """A malformed filter string only fails at render time, on the user's machine, after a
    generation. Run each chain through real ffmpeg on a synthetic clip instead."""
    out = tmp_path / "out.mp4"
    chain = ",".join(geometry_filters(fx, 320, 240) + ["setsar=1", "fps=12", "format=yuv420p"])
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=200x150:rate=12:duration=0.5",
         "-vf", chain, "-frames:v", "6", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-1500:]
    assert out.stat().st_size > 0


# ── reverse ───────────────────────────────────────────────────────────────────

def test_reverse_allowed_for_ordinary_clips():
    from movie_editor.backend.nle_effects import reverse_refusal
    assert reverse_refusal(20, 24) is None      # a typical FunPack clip
    assert reverse_refusal(49, 24) is None      # just under the cap
    assert reverse_refusal(None, None) is None  # unknown length is not a refusal


def test_reverse_refused_past_the_frame_cap():
    from movie_editor.backend.nle_effects import REVERSE_MAX_FRAMES, reverse_refusal
    msg = reverse_refusal(300, 24)
    assert msg and str(REVERSE_MAX_FRAMES) in msg
    # The message has to say what to do about it, not just that it failed.
    assert "Split" in msg


def test_reverse_cap_is_frames_not_seconds():
    """A high-fps clip hits the memory limit sooner in wall-clock terms."""
    from movie_editor.backend.nle_effects import reverse_refusal
    assert reverse_refusal(45, 24) is None
    assert reverse_refusal(45, 60) is not None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_ffmpeg_reverse_chain_runs_and_flips_order(tmp_path):
    """reverse+areverse must survive the same chain the render builds around them."""
    src, out = tmp_path / "src.mp4", tmp_path / "out.mp4"
    # A clip whose brightness ramps up, so reversing is detectable rather than assumed.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=1",
         "-vf", "geq=lum='N*20':cb=128:cr=128", "-frames:v", "10", str(src)],
        capture_output=True, text=True, check=True, timeout=120)
    chain = ",".join(["reverse"] + geometry_filters({}, 64, 64) + ["setsar=1", "fps=10"])
    r = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-vf", chain, str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]

    def first_frame_luma(path):
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=120)
        return sum(p.stdout[:64]) / 64.0

    # Source starts dark and ends bright; reversed, frame 0 must be the bright end.
    assert first_frame_luma(out) > first_frame_luma(src) + 20
