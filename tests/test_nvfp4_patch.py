"""NVFP4 on-the-fly patching: transform format + scope selection + core-load roundtrip.

Uses the REAL comfy.quant_ops / comfy.ops from the local ComfyUI install (comfy_kitchen's
eager backend quantizes and matmuls NVFP4 on CPU), so these tests prove byte-for-byte
compatibility with the format `comfy.ops._load_quantized_module` consumes - not just our
own invariants. Skipped wholesale if ComfyUI or comfy_kitchen is unavailable.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_COMFY_ROOT = os.environ.get("COMFYUI_ROOT", str(Path.home() / "Documents" / "ComfyUI"))
if _COMFY_ROOT not in sys.path and os.path.isdir(_COMFY_ROOT):
    sys.path.insert(0, _COMFY_ROOT)

quant_ops = pytest.importorskip("comfy.quant_ops")
if not quant_ops._CK_AVAILABLE:
    pytest.skip("comfy_kitchen unavailable", allow_module_level=True)

import comfy.ops  # noqa: E402

from nvfp4_patch import (  # noqa: E402
    QUANTIZE_SCOPES,
    _selected,
    quantize_state_dict_nvfp4,
)

D = 512  # >= _MIN_DIM and a multiple of 16


def _w(out_f=D, in_f=D):
    return torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02


def _fake_ltxav_sd(n_blocks=4, prefix=""):
    """Minimal LTX-AV-shaped state dict: video attn/ff, audio branch, cross-modal
    bridges, plus non-block layers that must never be touched by block scopes."""
    sd = {}
    for i in range(n_blocks):
        b = f"{prefix}transformer_blocks.{i}."
        for mod in ("attn1", "attn2", "audio_attn1", "audio_to_video_attn", "video_to_audio_attn"):
            for lin in ("to_q", "to_k", "to_v", "to_out.0"):
                sd[f"{b}{mod}.{lin}.weight"] = _w()
        sd[f"{b}ff.net.0.proj.weight"] = _w(2 * D, D)
        sd[f"{b}ff.net.2.weight"] = _w(D, 2 * D)
        sd[f"{b}audio_ff.net.0.proj.weight"] = _w(2 * D, D)
        sd[f"{b}scale_shift_table"] = torch.randn(6, D, dtype=torch.bfloat16)  # 2D but not .weight
        sd[f"{b}attn1.norm_q.weight"] = torch.randn(D, dtype=torch.bfloat16)   # 1D
    sd[f"{prefix}patchify_proj.weight"] = _w()
    sd[f"{prefix}proj_out.weight"] = _w()
    sd[f"{prefix}adaln_single.linear.weight"] = _w(6 * D, D)
    return sd


def _quant_layers(out_sd):
    return {k[: -len(".comfy_quant")] for k in out_sd if k.endswith(".comfy_quant")}


def test_video_scope_protects_audio_cross_edges_and_nonblock():
    sd = _fake_ltxav_sd()
    out, targets, keep = quantize_state_dict_nvfp4(sd, "video blocks", 1, 1)
    q = _quant_layers(out)
    assert keep == {0, 3}
    # middle-block video linears quantized
    assert "transformer_blocks.1.attn1.to_q" in q
    assert "transformer_blocks.2.ff.net.2" in q
    # first/last blocks untouched
    assert not any(l.startswith("transformer_blocks.0.") or l.startswith("transformer_blocks.3.") for l in q)
    # audio branch, cross-modal bridges, non-block layers untouched
    assert not any("audio" in l for l in q)
    assert not any(l.split(".")[-2:] == ["patchify_proj"] or "proj_out" in l or "adaln" in l for l in q)
    # untouched keys pass through identically
    assert torch.equal(out["transformer_blocks.0.attn1.to_q.weight"],
                       sd["transformer_blocks.0.attn1.to_q.weight"])
    assert len(targets) == len(q)


def test_scopes_widen_monotonically():
    sd = _fake_ltxav_sd()
    counts = []
    for scope in QUANTIZE_SCOPES:
        _, targets, _ = quantize_state_dict_nvfp4(sd, scope, 1, 1)
        counts.append(len(targets))
    assert counts == sorted(counts) and counts[0] < counts[-1]
    # cross-modal scope adds exactly the two bridges of the two middle blocks (4 linears each)
    assert counts[1] - counts[0] == 2 * 2 * 4


def test_all_2d_scope_still_skips_sensitive_generic_layers():
    sd = _fake_ltxav_sd()
    out, _, _ = quantize_state_dict_nvfp4(sd, "all 2D layers", 0, 0)
    q = _quant_layers(out)
    for name in ("proj_out", "adaln_single.linear", "patchify_proj"):
        assert not any(name in l for l in q)
    assert any("audio_attn1" in l for l in q)  # audio IS included in the widest scopes


def test_serialized_format_matches_core_loader_contract():
    sd = _fake_ltxav_sd(n_blocks=3)
    out, _, _ = quantize_state_dict_nvfp4(sd, "video blocks", 1, 1)
    layer = "transformer_blocks.1.attn1.to_q"
    assert out[f"{layer}.weight"].dtype == torch.uint8            # packed fp4 pairs
    assert out[f"{layer}.weight"].shape[1] == D // 2
    assert out[f"{layer}.weight_scale"].dtype == torch.float8_e4m3fn
    assert out[f"{layer}.weight_scale_2"].dtype == torch.float32
    import json
    conf = json.loads(bytes(out[f"{layer}.comfy_quant"].tolist()))
    assert conf == {"format": "nvfp4"}


def test_prefix_and_refusals():
    sd = _fake_ltxav_sd(prefix="model.diffusion_model.")
    out, targets, _ = quantize_state_dict_nvfp4(sd, "video blocks", 1, 1)
    assert any(k.startswith("model.diffusion_model.") and k.endswith(".comfy_quant") for k in out)
    assert targets
    # already-quantized checkpoints are refused
    with pytest.raises(ValueError):
        quantize_state_dict_nvfp4(out, "video blocks", 1, 1)
    with pytest.raises(ValueError):
        quantize_state_dict_nvfp4({"scaled_fp8": torch.zeros(2)}, "video blocks", 1, 1)


def test_selector_never_matches_bias_or_small():
    sd = {"transformer_blocks.1.attn1.to_q.bias": torch.randn(D, dtype=torch.bfloat16),
          "transformer_blocks.1.attn1.small.weight": _w(64, 64)}
    _, targets, _ = quantize_state_dict_nvfp4(sd, "all blocks", 0, 0)
    assert targets == []
    assert _selected("transformer_blocks.1.attn1.to_q", "video blocks", set())
    assert not _selected("transformer_blocks.1.audio_attn1.to_q", "video blocks", set())


@pytest.mark.parametrize("disabled", [[], ["nvfp4"]])
def test_core_mixed_precision_linear_loads_and_forwards(disabled):
    """The emitted keys load through comfy.ops.mixed_precision_ops Linear and produce
    output close to the bf16 reference - both on the eager NVFP4 matmul path and on
    the dequant-emulation path used by GPUs without FP4 compute."""
    torch.manual_seed(7)
    w = _w()
    sd = {"transformer_blocks.1.attn1.to_q.weight": w,
          "transformer_blocks.1.attn1.to_q.bias": torch.zeros(D, dtype=torch.bfloat16)}
    out, targets, _ = quantize_state_dict_nvfp4(sd, "video blocks", 0, 0)
    assert targets == ["transformer_blocks.1.attn1.to_q.weight"]

    ops = comfy.ops.mixed_precision_ops({}, torch.bfloat16, disabled=disabled)
    lin = ops.Linear(D, D, bias=True, device="cpu", dtype=torch.bfloat16)
    layer_sd = {k.replace("transformer_blocks.1.attn1.to_q.", ""): v for k, v in out.items()}
    missing, unexpected = lin.load_state_dict(layer_sd, strict=False)
    assert not missing and not unexpected
    assert isinstance(lin.weight, quant_ops.QuantizedTensor)
    assert lin.quant_format == "nvfp4"

    x = torch.randn(2, 8, D, dtype=torch.bfloat16) * 0.5
    with torch.no_grad():
        y = lin.forward(x)
        y_ref = torch.nn.functional.linear(x, w)
    assert y.shape == y_ref.shape
    rel = (y.float() - y_ref.float()).abs().mean() / (y_ref.float().abs().mean() + 1e-8)
    assert rel < 0.25, f"relative error too high: {rel:.3f} (disabled={disabled})"
