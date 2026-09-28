"""video_upscale.upscale_file: streams a real clip through an upscaler in chunks and keeps
its frame count, frame rate and sound; a failed run leaves no half-written file."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import video_upscale as vu  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _clip(path, frames=21, fps=24, sound=True):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
           f"testsrc=size=64x48:rate={fps}:duration={frames / fps}"]
    if sound:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={frames / fps}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def _count(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def _twice(x):
    return x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def test_upscales_every_frame_and_keeps_fps_and_sound(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _clip(src)
    calls = []
    frames = vu.upscale_file(str(src), str(dst), lambda x: calls.append(x.shape[0]) or _twice(x))
    assert frames == 21 == _count(dst)
    assert max(calls) <= vu.CHUNK                         # streamed, not the whole clip
    w, h, fps, sound = vu.probe(str(dst))
    assert (w, h, fps, sound) == (128, 96, "24/1", True)


def test_a_silent_clip_stays_silent(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _clip(src, sound=False)
    vu.upscale_file(str(src), str(dst), _twice)
    assert vu.probe(str(dst))[3] is False


def test_a_stopped_run_leaves_no_file(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _clip(src)
    seen = []

    def stop():
        seen.append(1)
        if len(seen) > 1:
            raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        vu.upscale_file(str(src), str(dst), _twice, stop)
    assert not dst.exists()
