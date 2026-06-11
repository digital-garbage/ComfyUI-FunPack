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
    aud = media.save_upload("bed.wav", b"RIFF")
    assert aud["kind"] == "audio"
    assert len(media.list_media()) == 3

    assert media.delete(entry["id"]) is True
    assert media.path_for(entry["id"]) is None
    assert sorted(m["id"] for m in media.list_media()) == sorted([vid["id"], aud["id"]])
    assert media.delete("nope") is False


def test_media_rename(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "MEDIA_DIR", tmp_path / "media")
    entry = media.save_upload("old name.png", b"data")
    updated = media.rename(entry["id"], "new label.png")
    assert updated is not None
    assert updated["name"] == "new label.png"
    assert updated["filename"] == entry["filename"]
    assert media.get(entry["id"])["name"] == "new label.png"
    assert media.rename(entry["id"], "   ") is None
    assert media.rename("missing", "x") is None
