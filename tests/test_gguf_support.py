"""GGUF backend selection and file discovery.

Nothing here needs a .gguf file or the gguf package: what is testable is the routing —
which backend gets picked, what happens when none is available, and whether .gguf files
are found at all (core's extension set excludes them, which is the whole reason this
module exists).
"""
import sys
import types

import pytest

sys.path.insert(0, ".")
import gguf_support  # noqa: E402


@pytest.fixture
def fake_folders(tmp_path, monkeypatch):
    """A stand-in folder_paths with two model roots on disk."""
    diff = tmp_path / "diffusion_models"
    enc = tmp_path / "text_encoders"
    sub = diff / "nested"
    for d in (diff, enc, sub):
        d.mkdir(parents=True, exist_ok=True)
    (diff / "ltx.safetensors").write_bytes(b"x")
    (diff / "ltx-Q4_K_M.gguf").write_bytes(b"x")
    (sub / "h3-Q8_0.gguf").write_bytes(b"x")
    (enc / "gemma.safetensors").write_bytes(b"x")
    (enc / "gemma-Q5_K.gguf").write_bytes(b"x")

    mod = types.ModuleType("folder_paths")
    mod.base_path = str(tmp_path)
    roots = {"diffusion_models": [str(diff)], "text_encoders": [str(enc)]}
    mod.get_folder_paths = lambda name: roots.get(name, [])
    monkeypatch.setitem(sys.modules, "folder_paths", mod)
    return tmp_path


def test_is_gguf_is_case_insensitive():
    assert gguf_support.is_gguf("model.gguf")
    assert gguf_support.is_gguf("MODEL.GGUF")
    assert not gguf_support.is_gguf("model.safetensors")
    assert not gguf_support.is_gguf("")
    assert not gguf_support.is_gguf(None)


def test_finds_gguf_files_core_cannot_list(fake_folders):
    names = gguf_support.gguf_names("diffusion_models")
    assert "ltx-Q4_K_M.gguf" in names
    # Nested, because model folders are routinely organised into subdirectories.
    assert any(n.endswith("h3-Q8_0.gguf") for n in names)
    # Only .gguf — the ordinary files still come from core's own listing.
    assert not any(n.endswith(".safetensors") for n in names)


def test_encoders_are_listed_separately(fake_folders):
    assert gguf_support.gguf_names("text_encoders") == ["gemma-Q5_K.gguf"]


def test_missing_folder_is_empty_not_an_error(fake_folders):
    assert gguf_support.gguf_names("vae") == []


def test_path_resolves_a_listed_name(fake_folders):
    p = gguf_support.gguf_path("diffusion_models", "ltx-Q4_K_M.gguf")
    assert p and p.endswith("ltx-Q4_K_M.gguf")
    assert gguf_support.gguf_path("diffusion_models", "not-there.gguf") is None


def test_no_backend_refuses_with_both_remedies(monkeypatch):
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: None)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: None)
    assert gguf_support.backend() is None
    with pytest.raises(RuntimeError) as e:
        gguf_support.load_state_dict("/nope.gguf")
    msg = str(e.value)
    # Both ways out must be named: they have different consequences for VRAM.
    assert "ComfyUI-GGUF" in msg
    assert "pip install gguf" in msg


def test_the_pack_wins_when_both_are_available(monkeypatch):
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: "/fake/ComfyUI-GGUF")
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: object())
    # Quantized-in-VRAM is the point of GGUF; the dequantizing fallback is second best.
    assert gguf_support.backend() == "pack"


def test_native_is_used_when_only_the_library_is_present(monkeypatch):
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: None)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec",
                        lambda name: object() if name == "gguf" else None)
    assert gguf_support.backend() == "native"


def test_pack_is_found_by_content_not_by_folder_name(tmp_path, monkeypatch):
    """Manager, forks and hand-installs all rename the folder; the marker is the code."""
    cn = tmp_path / "custom_nodes"
    good = cn / "some-renamed-fork"
    decoy = cn / "ComfyUI-GGUF-Themed-Nodes"   # named right, contains nothing
    for d in (good, decoy):
        d.mkdir(parents=True)
    (good / "loader.py").write_text("def gguf_sd_loader(path):\n    return {}\n")
    (good / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (decoy / "loader.py").write_text("# nothing useful\n")

    mod = types.ModuleType("folder_paths")
    mod.base_path = str(tmp_path)
    mod.get_folder_paths = lambda name: []
    monkeypatch.setitem(sys.modules, "folder_paths", mod)

    assert gguf_support._pack_dir() == str(good)


def test_pack_load_passes_custom_operations_through(tmp_path, monkeypatch):
    """The quantized weights are unusable without their ops — losing them would produce a
    matmul against block-quantized storage rather than a clear failure."""
    d = tmp_path / "pack"
    d.mkdir()
    (d / "loader.py").write_text(
        "def gguf_sd_loader(path):\n    return {'w': 'quantized'}\n")
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))

    sd, options, note = gguf_support.load_state_dict("/whatever.gguf")
    assert sd == {"w": "quantized"}
    assert "custom_operations" in options
    assert type(options["custom_operations"]).__name__ == "GGMLOps"
    assert "quantized" in note
