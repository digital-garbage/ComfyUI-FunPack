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


def test_container_magic_beats_the_extension(tmp_path):
    """A .gguf renamed to .safetensors reached the safetensors parser, which read its binary
    header as UTF-8 JSON and failed with a decode error — true, and no help at all."""
    renamed = tmp_path / "ltx-Q4_K_M.safetensors"
    renamed.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x80" * 200)
    assert gguf_support.has_gguf_magic(str(renamed))
    # ...and the name alone still says nothing.
    assert not gguf_support.is_gguf(renamed.name)


def test_a_real_safetensors_is_not_mistaken_for_gguf(tmp_path):
    st = tmp_path / "real.safetensors"
    st.write_bytes((8).to_bytes(8, "little") + b'{"a":{}}')
    assert not gguf_support.has_gguf_magic(str(st))


def test_magic_check_survives_a_missing_file():
    assert gguf_support.has_gguf_magic("/definitely/not/here.safetensors") is False


def test_pack_modules_can_use_relative_imports(tmp_path, monkeypatch):
    """ComfyUI-GGUF's loader.py does `from .ops import ...`. Loading it as a standalone
    module by file path raised "attempted relative import with no known parent package";
    mounting the directory as a package is what makes it resolve."""
    d = tmp_path / "ComfyUI-GGUF"
    d.mkdir()
    (d / "__init__.py").write_text("raise AssertionError('the pack __init__ must not run')")
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "dequant.py").write_text("MARKER = 'dequant'\n")
    (d / "loader.py").write_text(
        "from .ops import GGMLOps\n"
        "from .dequant import MARKER\n"
        "def gguf_sd_loader(path):\n"
        "    return {'w': MARKER}\n"
    )
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)

    sd, options, note = gguf_support.load_state_dict("/whatever.gguf")
    assert sd == {"w": "dequant"}
    assert type(options["custom_operations"]).__name__ == "GGMLOps"


def test_switching_pack_directories_remounts(tmp_path, monkeypatch):
    """The synthesized package is cached in sys.modules; a different directory must not keep
    resolving to the first one."""
    for name, marker in (("a", "first"), ("b", "second")):
        d = tmp_path / name
        d.mkdir()
        (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
        (d / "loader.py").write_text(
            f"def gguf_sd_loader(path):\n    return {{'w': '{marker}'}}\n")
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)

    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(tmp_path / "a"))
    assert gguf_support.load_state_dict("/x.gguf")[0] == {"w": "first"}

    # No manual cache clearing: remounting a different directory must purge it by itself.
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(tmp_path / "b"))
    assert gguf_support.load_state_dict("/x.gguf")[0] == {"w": "second"}
