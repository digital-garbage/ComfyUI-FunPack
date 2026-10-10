"""The loader says what the FILE holds, and when this torch build makes int8 slow."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """The loader test imports a node module, which imports comfy."""


def test_stored_weights_reads_the_tensors_not_the_widget():
    from modules.loaders.common import stored_weights
    mixed = {"a": torch.zeros(64, 64, dtype=torch.int8), "b": torch.zeros(32, 32, dtype=torch.bfloat16),
             "scale": torch.zeros(64, dtype=torch.float32), "tiny": torch.zeros(2, 2, dtype=torch.float32)}
    assert stored_weights(mixed) == "int8+bf16", "1-D scales and a <10% sliver are not the format"
    assert stored_weights({"a": torch.zeros(8, 8, dtype=torch.bfloat16)}) == "bf16"
    assert stored_weights({"n": 1}) is None


@pytest.mark.parametrize("cuda,cap,gpu,slow", [
    ("12.8", (12, 0), True, True),      # the rental that ran int8 2x slower
    ("13.0", (12, 0), True, False),
    ("12.8", (8, 9), True, False),      # not Blackwell: not measured, not claimed
    ("12.8", (12, 0), False, False),
    (None, (12, 0), True, False),       # CPU / MPS build
])
def test_slow_int8_build_names_only_cu12_on_blackwell(monkeypatch, cuda, cap, gpu, slow):
    from modules.loaders import common
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    msg = common.slow_int8_build()
    assert bool(msg) is slow
    if slow:
        assert "--force-reinstall" in msg and "cu130" in msg, "without the flag pip keeps the cu128 build"


def _load(monkeypatch, sd, weight_dtype, slow=None):
    import comfy.sd
    import comfy.utils
    import folder_paths
    from modules.loaders import gguf_support
    from modules.loaders.diffusion_model import nodes
    monkeypatch.setattr(folder_paths, "get_full_path_or_raise", lambda kind, name: "/fake/path")
    monkeypatch.setattr(gguf_support, "has_gguf_magic", lambda path: False)
    monkeypatch.setattr(comfy.utils, "load_torch_file", lambda path, return_metadata: (sd, {}))
    monkeypatch.setattr(comfy.sd, "load_diffusion_model_state_dict", lambda sd, model_options, metadata: object())
    monkeypatch.setattr(nodes, "slow_int8_build", lambda: slow)
    said = []
    monkeypatch.setattr(nodes.log, "info", lambda src, msg: said.append(("info", msg)))
    monkeypatch.setattr(nodes.log, "alert", lambda src, msg: said.append(("alert", msg)))
    nodes.FunPackDiffusionModelLoader.execute(model_name="model.safetensors", weight_dtype=weight_dtype,
                                              compute_dtype="bf16", attention="default")
    return said


def test_an_int8_file_loaded_at_bf16_is_logged_as_int8(monkeypatch):
    said = _load(monkeypatch, {"w": torch.zeros(8, 8, dtype=torch.int8)}, "bf16")
    line = [m for lvl, m in said if lvl == "info" and "loaded as" in m][0]
    assert "weights int8 in file (bf16 asked)" in line


def test_a_plain_file_reads_as_before(monkeypatch):
    said = _load(monkeypatch, {"w": torch.zeros(8, 8, dtype=torch.bfloat16)}, "bf16", slow="slow build")
    line = [m for lvl, m in said if lvl == "info" and "loaded as" in m][0]
    assert "weights bf16," in line
    assert not [m for lvl, m in said if lvl == "alert"], "the slow-build alert is about int8 files only"


def test_an_int8_file_on_a_slow_build_raises_the_alert(monkeypatch):
    said = _load(monkeypatch, {"w": torch.zeros(8, 8, dtype=torch.int8)}, "default", slow="slow build")
    assert ("alert", "slow build") in said
