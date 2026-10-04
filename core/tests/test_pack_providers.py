import json

from core import nodes_manager


def test_providers_reads_managers_node_map(tmp_path, monkeypatch):
    manager = tmp_path / "ComfyUI-Manager"
    manager.mkdir()
    (manager / "extension-node-map.json").write_text(json.dumps({
        "https://github.com/a/pack-a": [["NodeOne", "NodeTwo"], {"title": "A"}],
        "https://github.com/b/pack-b": [["NodeTwo"], {}],
    }))
    monkeypatch.setattr(nodes_manager, "custom_nodes_root", lambda: tmp_path)
    got = nodes_manager.providers(["NodeOne", "NodeTwo", "Nobody"])
    assert got == {"NodeOne": "https://github.com/a/pack-a", "NodeTwo": "https://github.com/a/pack-a", "Nobody": None}


def test_without_manager_nothing_is_known(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes_manager, "custom_nodes_root", lambda: tmp_path)
    assert nodes_manager.providers(["X"]) == {"X": None}
    (tmp_path / "ComfyUI-Manager").mkdir()
    (tmp_path / "ComfyUI-Manager" / "extension-node-map.json").write_text("not json")
    assert nodes_manager.providers(["X"]) == {"X": None}
