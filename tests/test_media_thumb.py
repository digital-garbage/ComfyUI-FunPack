"""media.thumb_path_for: cached, downscaled thumbnails instead of serving the full original
file for every media-bin grid render (see mediabrowser.js's own client-side history of the
same problem, fixed alongside this on the frontend side)."""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_editor.backend import config, media  # noqa: E402

pytest.importorskip("PIL", reason="Pillow is a real dependency of this project")
from PIL import Image  # noqa: E402

FFMPEG = shutil.which("ffmpeg")


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path)
    monkeypatch.setattr(media, "_load_index", lambda: media._items)
    monkeypatch.setattr(media, "_save_index", lambda items: media._items.__setitem__(slice(None), items))
    media._items = []
    return tmp_path


def _add_image(media_dir, name="big.png", size=(1200, 600)):
    Image.new("RGB", size, (200, 80, 40)).save(media_dir / name)
    mid = "img" + name.replace(".", "")
    media._items.append({"id": mid, "name": name, "filename": name, "kind": "image",
                          "size": 0, "added": 0})
    return mid


def test_generates_a_downscaled_jpeg_for_an_oversized_image(media_dir):
    mid = _add_image(media_dir, "big.png", (1200, 600))
    out = media.thumb_path_for(mid)
    assert out is not None and out.is_file()
    assert out.suffix == ".jpg"
    with Image.open(out) as im:
        assert max(im.size) <= media.THUMB_MAX_DIM
        assert im.size[0] / im.size[1] == pytest.approx(1200 / 600, rel=0.02)


def test_a_small_image_is_not_upscaled(media_dir):
    mid = _add_image(media_dir, "small.png", (64, 32))
    out = media.thumb_path_for(mid)
    with Image.open(out) as im:
        assert im.size == (64, 32)


def test_second_call_reuses_the_cached_file_instead_of_regenerating(media_dir, monkeypatch):
    mid = _add_image(media_dir)
    first = media.thumb_path_for(mid)
    first_mtime = first.stat().st_mtime
    monkeypatch.setattr(media, "_make_image_thumb",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not regenerate")))
    second = media.thumb_path_for(mid)
    assert second == first
    assert second.stat().st_mtime == first_mtime


def test_audio_and_other_kinds_have_no_thumbnail(media_dir):
    media._items.append({"id": "a1", "name": "x.mp3", "filename": "x.mp3", "kind": "audio",
                          "size": 0, "added": 0})
    (media_dir / "x.mp3").write_bytes(b"\0")
    assert media.thumb_path_for("a1") is None


def test_missing_source_file_returns_none(media_dir):
    media._items.append({"id": "gone", "name": "gone.png", "filename": "gone.png",
                          "kind": "image", "size": 0, "added": 0})
    assert media.thumb_path_for("gone") is None


def test_unknown_id_returns_none(media_dir):
    assert media.thumb_path_for("nope") is None


def test_delete_removes_the_cached_thumbnail_too(media_dir):
    mid = _add_image(media_dir)
    out = media.thumb_path_for(mid)
    assert out.is_file()
    assert media.delete(mid)
    assert not out.is_file()


@pytest.mark.skipif(not FFMPEG, reason="needs a real ffmpeg on PATH")
def test_generates_a_frame_grab_for_video(media_dir, tmp_path):
    import subprocess
    src = media_dir / "clip.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=blue:s=640x360:d=1", "-frames:v", "3", str(src)],
        capture_output=True, check=True,
    )
    media._items.append({"id": "vid1", "name": "clip.mp4", "filename": "clip.mp4",
                          "kind": "video", "size": 0, "added": 0})
    out = media.thumb_path_for("vid1")
    assert out is not None and out.is_file()
    with Image.open(out) as im:
        assert max(im.size) <= media.THUMB_MAX_DIM
        assert im.size[0] / im.size[1] == pytest.approx(640 / 360, rel=0.05)
