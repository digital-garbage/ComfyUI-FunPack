"""Character library store tests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from movie_editor.backend import characters  # noqa: E402


def test_character_library_crud(tmp_path, monkeypatch):
    db = tmp_path / "characters.json"
    monkeypatch.setattr(characters, "CHARACTERS_PATH", db)

    assert characters.list_characters()["characters"] == []

    characters.save_character({"name": "Nicole", "appearance": "red hair", "face_ref": "img1"})
    items = characters.list_characters()["characters"]
    assert len(items) == 1
    assert items[0]["name"] == "Nicole"
    assert items[0]["face_ref"] == "img1"
    cid = items[0]["id"]

    characters.save_character({"id": cid, "name": "Nicole V2", "appearance": "auburn hair"})
    items = characters.list_characters()["characters"]
    assert len(items) == 1
    assert items[0]["name"] == "Nicole V2"

    characters.delete_character(cid)
    assert characters.list_characters()["characters"] == []
    assert json.loads(db.read_text(encoding="utf-8"))["characters"] == {}