"""Model family detection: a safetensors header read cold, then handed to
whichever model module claims it. Core never names a family -- these tests
prove that split holds, not any one architecture's signature (that belongs
to modules/models/minimax_h3's own tests).
"""

import json
import struct
import sys
import types
from pathlib import Path

import pytest

from core import probe
from core.contract import ModuleSpec
from core.registry import Registry


def _write_safetensors(path, keys):
    header = {k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 0]} for k in keys}
    header["__metadata__"] = {"format": "pt"}
    body = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(body)) + body)


# --- reading the header -----------------------------------------------------

def test_keys_are_read_without_touching_anything_past_the_header(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["video_patch_proj.weight", "unrelated.bias"])
    keys = probe.read_safetensors_keys(p)
    assert set(keys) == {"video_patch_proj.weight", "unrelated.bias"}


def test_metadata_is_not_a_tensor_name(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["a.weight"])
    assert "__metadata__" not in probe.read_safetensors_keys(p)


def test_a_file_too_short_to_hold_a_length_is_not_a_header(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(b"\x01\x02\x03")
    assert probe.read_safetensors_keys(p) is None


def test_a_wild_length_is_refused_rather_than_allocated(tmp_path):
    p = tmp_path / "m.safetensors"
    # A length bigger than the ceiling, however the bytes got that way --
    # some other file format's first eight bytes, not a real header.
    p.write_bytes(struct.pack("<Q", probe.MAX_HEADER_BYTES + 1) + b"{}")
    assert probe.read_safetensors_keys(p) is None


def test_a_header_that_is_not_json_is_not_a_header(tmp_path):
    p = tmp_path / "m.safetensors"
    body = b"not json"
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    assert probe.read_safetensors_keys(p) is None


def test_a_header_that_is_valid_json_but_not_an_object_is_not_a_header(tmp_path):
    p = tmp_path / "m.safetensors"
    body = json.dumps(["a", "list"]).encode()
    p.write_bytes(struct.pack("<Q", len(body)) + body)
    assert probe.read_safetensors_keys(p) is None


# --- detect() -----------------------------------------------------------

def _registry(*claims):
    """claims: (id, title, keys_it_recognises) triples."""
    reg = Registry()
    for module_id, title, recognises in claims:
        reg.add(ModuleSpec(id=module_id, title=title, mount="",
                           provides={"detect": lambda keyset, r=recognises: r <= keyset}))
    return reg


def test_a_file_no_module_recognises_says_so_rather_than_guessing(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["some.unrelated.weight"])
    out = probe.detect(p, registry=_registry(("m", "M", {"video_patch_proj.weight"})))
    assert out == {"module": None, "title": None, "detected": False,
                   "reason": f"{p.name}: no installed model module recognises this file"}


def test_a_file_a_module_recognises_names_that_module(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["video_patch_proj.weight", "audio_patch_proj.weight"])
    out = probe.detect(p, registry=_registry(("model_minimax_h3", "MiniMax H3",
                                              {"video_patch_proj.weight", "audio_patch_proj.weight"})))
    assert out["module"] == "model_minimax_h3"
    assert out["title"] == "MiniMax H3"
    assert out["detected"] is True
    assert "MiniMax H3" in out["reason"]


def test_a_missing_file_is_not_found_not_undetected(tmp_path):
    out = probe.detect(tmp_path / "nope.safetensors", registry=_registry())
    assert out["detected"] is False
    assert "not found" in out["reason"]


def test_a_non_safetensors_extension_is_refused_without_reading_it(tmp_path):
    p = tmp_path / "m.ckpt"
    p.write_bytes(b"whatever a pickle looks like")
    out = probe.detect(p, registry=_registry())
    assert out["detected"] is False
    assert "only .safetensors" in out["reason"]


def test_the_first_module_that_claims_it_wins_in_a_stable_order(tmp_path):
    """providers() is sorted by module id -- deterministic, not import order."""
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["shared.weight"])
    out = probe.detect(p, registry=_registry(
        ("z_module", "Z", {"shared.weight"}),
        ("a_module", "A", {"shared.weight"}),
    ))
    assert out["module"] == "a_module"


def test_a_detector_that_raises_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "m.safetensors"
    _write_safetensors(p, ["real.weight"])
    reg = Registry()
    def explodes(_keys):
        raise RuntimeError("boom")
    reg.add(ModuleSpec(id="bad", title="Bad", mount="", provides={"detect": explodes}))
    reg.add(ModuleSpec(id="good", title="Good", mount="",
                       provides={"detect": lambda keys: "real.weight" in keys}))
    out = probe.detect(p, registry=reg)
    assert out["module"] == "good"


# --- resolve_diffusion_model -------------------------------------------------
#
# `folder_paths` only exists inside a running ComfyUI, and whether it happens
# to be importable here depends on which OTHER tests already ran and pulled
# ComfyUI's source onto sys.path (core/tests/conftest.py's `comfyui` fixture
# does this session-wide, not just for the test that asked). Asserting on
# ambient absence would make this test's result depend on run order rather
# than on what resolve_diffusion_model() actually does -- so a fake module is
# installed instead of trusting whatever is or isn't already importable.

def _fake_folder_paths(monkeypatch, get_full_path):
    fake = types.SimpleNamespace(get_full_path=get_full_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)


def test_resolve_finds_a_file_in_the_diffusion_models_folder(monkeypatch):
    _fake_folder_paths(monkeypatch, lambda folder, name: (
        "/models/diffusion_models/x.safetensors" if folder == "diffusion_models" else None))
    assert probe.resolve_diffusion_model("x.safetensors") == Path("/models/diffusion_models/x.safetensors")


def test_resolve_falls_back_to_checkpoints(monkeypatch):
    _fake_folder_paths(monkeypatch, lambda folder, name: (
        "/models/checkpoints/x.safetensors" if folder == "checkpoints" else None))
    assert probe.resolve_diffusion_model("x.safetensors") == Path("/models/checkpoints/x.safetensors")


def test_resolve_never_looks_in_a_unet_folder(monkeypatch):
    """No loader in this app reads a model from a "unet" folder --
    modules/loaders/diffusion_model offers "diffusion_models" files,
    modules/loaders/checkpoint offers "checkpoints" ones. Probing "unet" too
    would risk answering for a file no loader here would ever actually load
    under that name."""
    asked = []
    _fake_folder_paths(monkeypatch, lambda folder, name: (asked.append(folder), None)[1])
    probe.resolve_diffusion_model("x.safetensors")
    assert "unet" not in asked


def test_resolve_is_none_when_no_folder_has_it(monkeypatch):
    _fake_folder_paths(monkeypatch, lambda folder, name: None)
    assert probe.resolve_diffusion_model("nope.safetensors") is None


def test_resolve_is_none_when_folder_paths_itself_raises(monkeypatch):
    def boom(folder, name):
        raise KeyError(folder)
    _fake_folder_paths(monkeypatch, boom)
    assert probe.resolve_diffusion_model("x.safetensors") is None
