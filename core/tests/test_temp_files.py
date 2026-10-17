"""The gallery lists what ComfyUI saved, and deletes only media inside that folder."""

from core import temp_files


def test_saved_files_are_listed_from_output_not_temp(monkeypatch, tmp_path):
    (tmp_path / "FunPack_00001_.mp4").write_bytes(b"x")
    monkeypatch.setattr(temp_files, "output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(temp_files, "temp_dir", lambda: None)
    assert [f["filename"] for f in temp_files.listing(where="output")["files"]] == ["FunPack_00001_.mp4"]
    assert temp_files.listing()["files"] == []


def test_delete_removes_a_media_file_and_nothing_else(monkeypatch, tmp_path):
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    pic, weights, outside = out / "sub" / "a.png", out / "model.safetensors", tmp_path / "secret.png"
    for f in (pic, weights, outside):
        f.write_bytes(b"x")
    monkeypatch.setattr(temp_files, "output_dir", lambda: str(out))

    assert temp_files.delete("a.png", "sub", "output") == {"deleted": True} and not pic.exists()
    assert not temp_files.delete("model.safetensors", "", "output")["deleted"] and weights.exists()
    assert not temp_files.delete("secret.png", "..", "output")["deleted"] and outside.exists(), "left the folder"
    assert not temp_files.delete("../secret.png", "", "output")["deleted"] and outside.exists()
    assert not temp_files.delete("a.png", "sub", "output")["deleted"], "a second delete should say it is gone"
    assert not temp_files.delete("a.png", "", "input")["deleted"]
