"""GGUF backend selection and file discovery.

Nothing here needs a .gguf file or the gguf package: what is testable is the routing —
which backend gets picked, what happens when none is available, and whether .gguf files
are found at all (core's extension set excludes them, which is the whole reason this
module exists).
"""
import gc
import os
import sys
import types
import weakref

import pytest
import torch

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
    # Both ways out must be named: they have different consequences for VRAM. `gguf` is a
    # FunPack requirement now, so the first is a re-run of them rather than a lone package.
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


def test_pack_refusing_an_architecture_falls_back(tmp_path, monkeypatch):
    """ComfyUI-GGUF raises "Unexpected architecture type in GGUF file: 'minimax_h3'" for
    architectures it has no handling for. That is a fair refusal on its part and a dead end
    for us — but the pack DEPENDS on the gguf package, so the fallback is always there."""
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
                        lambda p: ({"w": 1}, {}, "gguf: dequantized at load"))

    sd, options, note = gguf_support.load_state_dict("/h3.gguf")
    assert sd == {"w": 1}
    # Both halves must be in the note: what refused, and what happened instead.
    assert "minimax_h3" in note
    assert "dequantized at load" in note
    # The quantized ops must NOT be carried over — these tensors are plain.
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
    """The advice in UNAVAILABLE is "re-run FunPack's requirements", which is only true
    while gguf is actually in them."""
    import pathlib
    reqs = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in reqs.splitlines() if ln.strip() and not ln.startswith("#")]
    assert any(ln == "gguf" or ln.startswith("gguf") for ln in lines)


# ── expand once, not every launch ─────────────────────────────────────────────
# The native path has to dequantize every tensor before the model is usable — minutes on a
# video checkpoint, paid again on every launch for a result that is identical every time.


def test_the_cache_sits_beside_the_model(tmp_path):
    src = str(tmp_path / "m.gguf")
    assert gguf_support.cache_path(src) == src + gguf_support.CACHE_SUFFIX


def test_a_cache_older_than_the_model_is_not_trusted(tmp_path):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    dst = tmp_path / ("m.gguf" + gguf_support.CACHE_SUFFIX)
    dst.write_bytes(b"old")
    os.utime(dst, (1, 1))                       # written before the model
    assert not gguf_support._cache_is_current(str(src), str(dst))


def test_a_cache_newer_than_the_model_is_used(tmp_path, monkeypatch):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    dst = tmp_path / ("m.gguf" + gguf_support.CACHE_SUFFIX)
    dst.write_bytes(b"new")
    os.utime(src, (1, 1))
    assert gguf_support._cache_is_current(str(src), str(dst))


def test_a_missing_cache_is_simply_not_current(tmp_path):
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    assert not gguf_support._cache_is_current(str(src), str(tmp_path / "nope"))


def test_the_expansion_is_written_and_reused(tmp_path, monkeypatch):
    """The whole point: the second load must not dequantize anything."""
    import torch
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    calls = []

    def fake_native(path):
        calls.append(path)
        return {"w": torch.zeros(2, 2)}, {}, "gguf: dequantized at load"

    monkeypatch.setattr(gguf_support, "_load_native", fake_native)
    # The stub comfy tree has no real loader; the cache is an ordinary safetensors file.
    import safetensors.torch
    monkeypatch.setattr("comfy.utils.load_torch_file",
                        lambda p, **kw: safetensors.torch.load_file(p), raising=False)
    sd1, _, note1 = gguf_support._load_native_cached(str(src))
    assert calls == [str(src)] and "Written to" in note1
    sd2, _, note2 = gguf_support._load_native_cached(str(src))
    assert calls == [str(src)]                  # never expanded a second time
    assert "dequantized cache" in note2
    assert torch.equal(sd1["w"], sd2["w"])


def test_an_unwritable_cache_is_a_slow_load_not_a_failed_one(tmp_path, monkeypatch):
    import torch
    src = tmp_path / "m.gguf"
    src.write_bytes(b"GGUF")
    monkeypatch.setattr(gguf_support, "_load_native",
                        lambda p: ({"w": torch.zeros(2, 2)}, {}, "expanded"))
    monkeypatch.setattr("safetensors.torch.save_file",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    sd, _, note = gguf_support._load_native_cached(str(src))
    assert "w" in sd and "could not write" in note
    assert not os.path.exists(gguf_support.cache_path(str(src)) + ".part")   # no debris


# ── letting the pack read an architecture it has not been told about ──────────


class _Loader:
    def __init__(self, arches):
        self.IMG_ARCH_LIST = set(arches)


def test_an_unknown_architecture_is_added_so_the_file_loads_quantized(monkeypatch):
    monkeypatch.setattr(gguf_support, "read_architecture", lambda p: "minimax_h3")
    loader = _Loader({"flux", "sd3"})
    note = gguf_support._allow_architectures(loader, "x.gguf")
    assert "minimax_h3" in loader.IMG_ARCH_LIST
    assert "minimax_h3" in note and "FunPack added" in note


def test_a_known_architecture_is_left_alone_and_says_nothing(monkeypatch):
    monkeypatch.setattr(gguf_support, "read_architecture", lambda p: "flux")
    loader = _Loader({"flux"})
    assert gguf_support._allow_architectures(loader, "x.gguf") == ""


def test_an_unreadable_architecture_changes_nothing(monkeypatch):
    monkeypatch.setattr(gguf_support, "read_architecture", lambda p: None)
    loader = _Loader({"flux"})
    assert gguf_support._allow_architectures(loader, "x.gguf") == ""
    assert loader.IMG_ARCH_LIST == {"flux"}


def test_the_text_encoder_list_is_never_touched(monkeypatch):
    """Image and text architectures steer different key renaming in the pack; guessing a
    text arch into the image list is one thing, the reverse is another."""
    monkeypatch.setattr(gguf_support, "read_architecture", lambda p: "minimax_h3")
    loader = _Loader({"flux"})
    loader.TXT_ARCH_LIST = {"t5"}
    gguf_support._allow_architectures(loader, "x.gguf")
    assert loader.TXT_ARCH_LIST == {"t5"}


def test_a_backend_that_returns_something_else_is_refused(tmp_path):
    """The pack's own guard is what we stepped around, so the check it was doing happens
    here: handing a non-state-dict to ComfyUI fails deep in core with a message that makes no
    sense out here. Raising instead drops to the slow path that works."""
    for bad in (None, [], "sd", {}, {1: object()}):
        with pytest.raises(RuntimeError):
            gguf_support._assert_state_dict(bad, "m.gguf")


def test_a_non_tensor_value_is_named(tmp_path):
    with pytest.raises(RuntimeError, match="not a tensor"):
        gguf_support._assert_state_dict({"w": {"nested": 1}}, "m.gguf")


def test_a_real_state_dict_passes():
    import torch
    gguf_support._assert_state_dict({"w": torch.zeros(2)}, "m.gguf")


def _fake_gguf(monkeypatch, tensors):
    """A stand-in for the `gguf` package over `tensors` — (name, shape, values) triples.

    Every tensor is reported as quantized, so `_load_native` takes its dequantize path for
    all of them, and each dequantize hands back a fresh float32 array whose lifetime the
    caller can track.
    """
    import numpy as np

    fake = types.ModuleType("gguf")

    class _QType:
        F32 = "F32"
        F16 = "F16"
        Q8 = "Q8"

    class _Tensor:
        def __init__(self, name, shape, values):
            self.name = name
            # GGUF reports dimensions in the opposite order to torch.
            self.shape = tuple(reversed(shape))
            self.tensor_type = _QType.Q8
            self.data = np.asarray(values, dtype=np.float32)

    class _Reader:
        def __init__(self, path):
            self.tensors = [_Tensor(n, s, v) for n, s, v in tensors]

    produced = []
    live = []

    def _dequantize(data, qtype):
        # Sampled BEFORE this call's own array exists, so it counts what the loader is still
        # holding on to. Peak is the whole point: every array is freed by the time the load
        # returns no matter how it was written.
        live.append(sum(1 for ref in produced if ref() is not None))
        arr = np.asarray(data, dtype=np.float32).copy()
        produced.append(weakref.ref(arr))
        return arr

    quants = types.ModuleType("gguf.quants")
    quants.dequantize = _dequantize
    fake.GGMLQuantizationType = _QType
    fake.GGUFReader = _Reader
    fake.quants = quants
    monkeypatch.setitem(sys.modules, "gguf", fake)
    monkeypatch.setitem(sys.modules, "gguf.quants", quants)
    return produced, live


def test_load_native_returns_half_width_tensors_in_torch_order(monkeypatch):
    monkeypatch.setattr(gguf_support.os, "cpu_count", lambda: 2)   # forces the serial path
    _fake_gguf(monkeypatch, [("a", (2, 3), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])])

    sd, options, note = gguf_support._load_native("/models/m.gguf")

    assert options == {}
    assert list(sd) == ["a"]
    assert tuple(sd["a"].shape) == (2, 3)
    assert sd["a"].dtype is torch.float16
    assert sd["a"][1][2].item() == 6.0
    assert "1 quantized tensors" in note


def test_load_native_does_not_keep_every_expanded_tensor_alive(monkeypatch):
    """The float32 form is four times the finished weight, so holding all of them at once
    peaks at several times the model in host RAM — enough to take a machine down on a video
    checkpoint. Each one must be cast and released as it is produced."""
    monkeypatch.setattr(gguf_support.os, "cpu_count", lambda: 2)   # forces the serial path
    produced, live = _fake_gguf(
        monkeypatch, [(f"t{i}", (4, 4), [[float(i)] * 4] * 4) for i in range(24)])

    gguf_support._load_native("/models/m.gguf")

    assert len(produced) == 24
    # Collecting them all before converting any would climb 0, 1, 2, ... 23.
    assert max(live) <= 2, f"{max(live)} expanded tensors were held at once"
    gc.collect()
    assert [ref for ref in produced if ref() is not None] == []


def test_load_native_rejects_a_tensor_that_contradicts_its_header(monkeypatch):
    monkeypatch.setattr(gguf_support.os, "cpu_count", lambda: 2)
    _fake_gguf(monkeypatch, [("wrong", (5, 5), [1.0, 2.0, 3.0])])

    with pytest.raises(RuntimeError) as excinfo:
        gguf_support._load_native("/models/m.gguf")

    assert "'wrong'" in str(excinfo.value)
    assert "(25)" in str(excinfo.value)
