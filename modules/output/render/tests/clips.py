"""Real ffmpeg, a fake ComfyUI folder pair, and clips made on the spot."""

import shutil
import subprocess

import pytest

from core import config

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


@pytest.fixture
def comfy(tmp_path, monkeypatch):
    """ComfyUI's output and temp folders, plus FunPack's own stores, all under tmp_path."""
    from modules.output.render import files
    out, temp = tmp_path / "output", tmp_path / "temp"
    out.mkdir(), temp.mkdir()
    monkeypatch.setattr(files, "comfy_dir", lambda kind: str(out if kind == "output" else temp))
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    return type("Comfy", (), {"output": out, "temp": temp, "root": tmp_path})


def make_clip(path, seconds=2.0, size="160x120", rate=25, audio=True, faststart=True):
    """A real mp4: a test pattern, with a sine tone unless `audio` is False."""
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def probe(path):
    """(duration seconds, width, height, has audio) of a real file."""
    import json
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                          "-of", "json", str(path)], check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return (float(data["format"]["duration"]), video["width"], video["height"],
            any(s["codec_type"] == "audio" for s in data["streams"]))
