"""GGUF backend selection and file discovery.

Nothing here needs a .gguf file or the gguf package: what is testable is the
routing -- which backend gets picked, what happens when none is available, and
whether .gguf files are found at all (core's extension set excludes them,
which is the whole reason this module exists).
"""
import sys
import types

import pytest

from modules.loaders import gguf_support


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
    # Only .gguf -- the ordinary files still come from core's own listing.
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
    # `gguf` is a FunPack requirement now, so the first is a re-run of them
    # rather than a lone package.
    assert "ComfyUI-GGUF" in msg
    assert "requirements.txt" in msg
    assert "quantized in VRAM" in msg


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
    """The quantized weights are unusable without their ops -- losing them
    would produce a matmul against block-quantized storage rather than a clear
    failure."""
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
    """A .gguf renamed to .safetensors reached the safetensors parser, which
    read its binary header as UTF-8 JSON and failed with a decode error --
    true, and no help at all."""
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
    """ComfyUI-GGUF's loader.py does `from .ops import ...`. Loading it as a
    standalone module by file path raised "attempted relative import with no
    known parent package"; mounting the directory as a package is what makes
    it resolve."""
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
    """The synthesized package is cached in sys.modules; a different directory
    must not keep resolving to the first one."""
    for name, marker in (("a", "first"), ("b", "second")):
        d = tmp_path / name
        d.mkdir()
        (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
        (d / "loader.py").write_text(
            f"def gguf_sd_loader(path):\n    return {{'w': '{marker}'}}\n")
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)

    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(tmp_path / "a"))
    assert gguf_support.load_state_dict("/x.gguf")[0] == {"w": "first"}

    # No manual cache clearing: remounting a different directory must purge it
    # by itself.
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(tmp_path / "b"))
    assert gguf_support.load_state_dict("/x.gguf")[0] == {"w": "second"}


def test_pack_refusing_an_architecture_falls_back(tmp_path, monkeypatch):
    """ComfyUI-GGUF raises "Unexpected architecture type in GGUF file:
    'minimax_h3'" for architectures it has no handling for. That is a fair
    refusal on its part and a dead end for us -- but the pack DEPENDS on the
    gguf package, so the fallback is always there."""
    d = tmp_path / "pack"
    d.mkdir()
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "loader.py").write_text(
        "def gguf_sd_loader(path):\n"
        "    raise ValueError(\"Unexpected architecture type in GGUF file: 'minimax_h3'\")\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec",
                        lambda name: object() if name == "gguf" else None)
    monkeypatch.setattr(gguf_support, "_load_native",
                        lambda p, pack=False: ({"w": 1}, {}, "gguf: dequantized at load"))

    sd, options, note = gguf_support.load_state_dict("/h3.gguf")
    assert sd == {"w": 1}
    # Both halves must be in the note: what refused, and what happened instead.
    assert "minimax_h3" in note
    assert "dequantized at load" in note
    # The quantized ops must NOT be carried over -- these tensors are plain.
    assert "custom_operations" not in options


def test_a_pack_refusal_without_the_library_still_raises(tmp_path, monkeypatch):
    """No silent success: with nothing to fall back to, the original refusal stands."""
    d = tmp_path / "pack"
    d.mkdir()
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "loader.py").write_text(
        "def gguf_sd_loader(path):\n    raise ValueError('nope')\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="nope"):
        gguf_support.load_state_dict("/h3.gguf")


def test_gguf_is_a_declared_requirement():
    """The advice in UNAVAILABLE is "re-run FunPack's requirements", which is
    only true while gguf is actually in them."""
    import pathlib
    reqs = pathlib.Path(__file__).resolve().parents[3] / "requirements.txt"
    lines = [ln.strip() for ln in reqs.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    assert any(ln == "gguf" or ln.startswith("gguf") for ln in lines)


def test_a_text_encoder_gguf_the_pack_refuses_is_refused_not_loaded_as_the_wrong_model(tmp_path, monkeypatch):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "loader.py").write_text(
        "def gguf_clip_loader(path):\n"
        "    raise ValueError(\"Unexpected text model architecture type in GGUF file: 'gemma4'\")\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: object() if name == "gguf" else None)
    monkeypatch.setattr(gguf_support, "_load_native", lambda path: ({"w": "native"}, {}, "native"))
    with pytest.raises(RuntimeError, match="could not be loaded"):
        gguf_support.load_clip_state_dict("/x.gguf")


def test_a_text_encoder_gguf_without_the_pack_is_refused_not_dequantized_blind(monkeypatch):
    monkeypatch.setattr(gguf_support, "backend", lambda: "native")
    monkeypatch.setattr(gguf_support, "_load_native", lambda path: ({"w": "native"}, {}, "native"))
    with pytest.raises(RuntimeError, match="ComfyUI-GGUF"):
        gguf_support.load_clip_state_dict("/x.gguf")
    assert gguf_support.load_state_dict("/x.gguf")[0] == {"w": "native"}      # a diffusion model still loads natively


def test_a_pack_that_returns_a_state_dict_and_extras_still_gives_a_state_dict(tmp_path, monkeypatch):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "loader.py").write_text("def gguf_sd_loader(path):\n    return ({'w': 1}, {'arch_str': 'flux'})\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    sd, _options, note = gguf_support.load_state_dict("/x.gguf")
    assert sd == {"w": 1} and "quantized" in note


def test_a_text_encoder_gguf_with_no_backend_says_the_pack_not_pip(monkeypatch):
    monkeypatch.setattr(gguf_support, "backend", lambda: None)
    with pytest.raises(RuntimeError, match="ComfyUI-GGUF"):
        gguf_support.load_clip_state_dict("/x.gguf")


def _pack(tmp_path, body):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "ops.py").write_text("class GGMLOps:\n    pass\n")
    (d / "loader.py").write_text(body)
    return d


def test_a_vision_projector_gguf_the_pack_lets_through_is_refused_as_a_text_encoder(tmp_path, monkeypatch):
    d = _pack(tmp_path, "def gguf_clip_loader(path):\n    return {'v.blk.0.attn_q.weight': 1}\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: None)        # same refusal without gguf
    with pytest.raises(RuntimeError, match="mmproj"):
        gguf_support.load_clip_state_dict("/mmproj.gguf")


def test_a_text_encoder_with_vision_tensors_beside_its_mapped_text_still_loads(tmp_path, monkeypatch):
    d = _pack(tmp_path, "def gguf_clip_loader(path):\n    return {'v.model.layers.0.x': 1, 'model.layers.0.q.weight': 2}\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    sd, _o, _n = gguf_support.load_clip_state_dict("/x.gguf")
    assert "model.layers.0.q.weight" in sd


def test_unmapped_text_names_are_refused_as_a_text_encoder(tmp_path, monkeypatch):
    d = _pack(tmp_path, "def gguf_clip_loader(path):\n    return {'blk.0.attn_q.weight': 1}\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        gguf_support.load_clip_state_dict("/x.gguf")


def test_a_pack_whose_gguf_library_is_missing_says_how_to_fix_it_not_an_import_error(tmp_path, monkeypatch):
    d = _pack(tmp_path, "import gguf_missing_for_this_test\ndef gguf_sd_loader(path):\n    return {}\n")
    monkeypatch.setattr(gguf_support, "_pack_dir", lambda: str(d))
    monkeypatch.delitem(sys.modules, gguf_support._PKG_NAME, raising=False)
    monkeypatch.setattr(gguf_support.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="installed, but the `gguf` package"):
        gguf_support.load_state_dict("/x.gguf")
    with pytest.raises(RuntimeError, match="installed, but the `gguf` package"):
        gguf_support.load_clip_state_dict("/x.gguf")


def test_a_name_that_climbs_out_of_the_model_folder_is_not_found(fake_folders):
    (fake_folders / "secret.gguf").write_bytes(b"x")
    assert gguf_support.gguf_path("text_encoders", "../secret.gguf") is None
    assert gguf_support.gguf_path("text_encoders", "gemma-Q5_K.gguf")


def test_a_pack_switched_off_by_manager_is_not_used(fake_folders):
    pack = fake_folders / "custom_nodes" / "ComfyUI-GGUF.disabled"
    pack.mkdir(parents=True)
    (pack / "ops.py").write_text("")
    (pack / "loader.py").write_text("def gguf_sd_loader(p): pass\n")
    assert gguf_support._pack_dir() is None
    (fake_folders / "custom_nodes" / "ComfyUI-GGUF").mkdir()
    (fake_folders / "custom_nodes" / "ComfyUI-GGUF" / "ops.py").write_text("")
    (fake_folders / "custom_nodes" / "ComfyUI-GGUF" / "loader.py").write_text("def gguf_sd_loader(p): pass\n")
    assert gguf_support._pack_dir().endswith("ComfyUI-GGUF")


def test_native_load_keeps_an_f32_value_too_big_for_fp16_finite(tmp_path):
    gguf = pytest.importorskip("gguf")
    torch = pytest.importorskip("torch")
    import numpy as np
    f = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(f), "llama")
    w.add_tensor("a", np.array([[70000.0, 1.0]], dtype=np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    sd, _o, _n = gguf_support._load_native(str(f))
    assert torch.isfinite(sd["a"].float()).all()



def test_a_symlinked_gguf_inside_the_model_folder_is_found(fake_folders):
    (fake_folders / "elsewhere.gguf").write_bytes(b"x")
    (fake_folders / "text_encoders" / "linked.gguf").symlink_to(fake_folders / "elsewhere.gguf")
    assert gguf_support.gguf_path("text_encoders", "linked.gguf")



def test_symlink_loops_do_not_hang_the_listing_and_dotdot_after_a_link_is_refused(fake_folders):
    enc = fake_folders / "text_encoders"
    (enc / "a").symlink_to(enc)
    (enc / "b").symlink_to(enc)
    assert gguf_support.gguf_names("text_encoders") == ["a/gemma-Q5_K.gguf"] or "gemma-Q5_K.gguf" in gguf_support.gguf_names("text_encoders")
    (fake_folders / "outside").mkdir()
    (fake_folders / "outside" / "inner").mkdir()
    (fake_folders / "outside" / "s.gguf").write_bytes(b"x")
    (enc / "lnk").symlink_to(fake_folders / "outside" / "inner")
    assert gguf_support.gguf_path("text_encoders", "lnk/../s.gguf") is None
