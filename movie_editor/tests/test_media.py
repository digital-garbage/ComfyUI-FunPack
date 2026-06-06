"""Tests for the media bin store (upload/list/delete round-trip)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import config, media  # noqa: E402


def test_media_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    assert media.list_media() == []

    entry = media.save_upload("shot one.png", b"\x89PNG\r\n\x1a\nfake")
    assert entry["kind"] == "image"
    assert entry["name"] == "shot one.png"
    listed = media.list_media()
    assert len(listed) == 1 and listed[0]["id"] == entry["id"]

    p = media.path_for(entry["id"])
    assert p is not None and p.read_bytes().startswith(b"\x89PNG")
    assert media.content_type(entry["id"]).startswith("image/")

    vid = media.save_upload("clip.mp4", b"\x00\x00\x00")
    assert vid["kind"] == "video"
    assert len(media.list_media()) == 2

    assert media.delete(entry["id"]) is True
    assert media.path_for(entry["id"]) is None
    assert [m["id"] for m in media.list_media()] == [vid["id"]]
    assert media.delete("nope") is False
