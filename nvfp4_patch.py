"""FunPack NVFP4 on-the-fly model patching (EXPERIMENTAL).

Blackwell's 5th-gen tensor cores run NVFP4 (e2m1 with per-16-element FP8-E4M3 block
scales + a per-tensor FP32 scale) at roughly 2x FP8 / 4x BF16 GEMM throughput. ComfyUI
core already ships the whole runtime for it (comfy.quant_ops layouts + MixedPrecisionOps
dispatch, dynamic activation quantization, LoRA fallback, dequant emulation on non-FP4
GPUs) - but only for checkpoints that were quantized ahead of time. Stock loaders offer
fp8 at best, and on sm_120 the fp8 path is mostly storage, not tensor-core speedup.

This module closes that gap: it quantizes an existing bf16/fp16 LTX-AV checkpoint to
NVFP4 *at load time* as a pure state-dict transform - selected 2D Linear weights are
re-serialized in core's exact quantized-checkpoint format (weight/weight_scale/
weight_scale_2 + per-layer `comfy_quant` metadata), then handed to the normal
`load_diffusion_model_state_dict` path. Core's own detection routes the model through
MixedPrecisionOps; every layer we did NOT tag simply stays at compute dtype. No custom
forward, no monkeypatching, no new inference code paths to maintain.

Zone-aware precision map (the part generic quantizers can't do): LTX-AV is a joint
audio-video transformer and its zones are not equally robust. By default we quantize
only the video-branch Linears (attn1/attn2/ff) of the middle blocks, keeping at full
precision:
  - the audio branch (audio_attn1/audio_attn2/audio_ff) - joint attention is the part
    that corrupts first (see the self-consistency post-block injection failure),
  - the cross-modal bridges (audio_to_video_attn / video_to_audio_attn),
  - the first/last N blocks (standard quantization practice; they carry the most
    input/output-adjacent signal),
  - everything that is not a plain 2D matmul (norms, ada modulation, embeddings).

Scopes widen from there for experimentation. W4A4 without calibration WILL cost some
fidelity - this is an experimental speed/VRAM lever, off unless you wire the node in.

Requires: ComfyUI with comfy.quant_ops (comfy_kitchen), torch cu13+ on Blackwell
(sm_120+) for real FP4 GEMMs. On anything older the model still loads and runs through
core's dequant emulation (no speedup, same quality) - useful for A/B sanity checks.
"""

from __future__ import annotations

import json
import logging
import re

import torch

QUANTIZE_SCOPES = [
    "video blocks",         # video-branch Linears in middle blocks (default, safest)
    "video + cross-modal",  # + audio_to_video_attn / video_to_audio_attn bridges
    "all blocks",           # every Linear inside transformer_blocks (incl. audio branch)
    "all 2D layers",        # + eligible Linears outside the blocks (most aggressive)
]

# Video-branch submodules inside a BasicAVTransformerBlock (comfy/ldm/lightricks/av_model.py).
_VIDEO_MODULES = ("attn1.", "attn2.", "ff.")
_CROSS_MODULES = ("audio_to_video_attn.", "video_to_audio_attn.")

# Matches "...transformer_blocks.<idx>.<rest>" with any leading prefix
# ("model.diffusion_model." or none - load_diffusion_model_state_dict strips prefixes
# from ALL keys uniformly, so our metadata keys ride along).
_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)\.(.*)$")

# Outside the blocks ("all 2D layers" scope) never touch input/output-adjacent or
# modulation weights even when they are 2D Linears.
_GENERIC_SKIP = ("proj_out", "adaln", "embed", "emb.", "norm", "scale_shift", "patchify")

# Below this, quantization overhead outweighs any GEMM win and relative error is worst.
_MIN_DIM = 512


def _block_ids(sd) -> list[int]:
    ids = set()
    for k in sd:
        m = _BLOCK_RE.search(k)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _selected(key_body: str, scope: str, keep: set[int]) -> bool:
    """Decide whether the Linear at `key_body` (state-dict key minus '.weight') is
    quantized under `scope`, given the high-precision `keep` block set."""
    m = _BLOCK_RE.search(key_body + ".")
    if not m:
        if scope != "all 2D layers":
            return False
        low = key_body.lower()
        return not any(s in low for s in _GENERIC_SKIP)
    idx, rest = int(m.group(1)), m.group(2)
    if idx in keep:
        return False
    if scope in ("all blocks", "all 2D layers"):
        return True
    if any(rest.startswith(p) for p in _VIDEO_MODULES):
        return True
    if scope == "video + cross-modal" and any(rest.startswith(p) for p in _CROSS_MODULES):
        return True
    return False


def _eligible(key: str, w) -> bool:
    return (
        key.endswith(".weight")
        and isinstance(w, torch.Tensor)
        and w.dim() == 2
        and w.shape[0] >= _MIN_DIM
        and w.shape[1] >= _MIN_DIM
        and w.shape[1] % 16 == 0  # NVFP4 blocks run along in_features
    )


def quantize_state_dict_nvfp4(sd, scope="video blocks", keep_first_blocks=2,
                              keep_last_blocks=2, device=None, progress_callback=None):
    """Return (new_sd, quantized_layer_names, kept_block_ids).

    Pure transform: selected 2D weights are replaced by NVFP4 QuantizedTensor
    serialization (weight uint8-packed, weight_scale fp8-e4m3 block scales,
    weight_scale_2 fp32 tensor scale) plus a per-layer `comfy_quant` metadata key -
    byte-for-byte the format `comfy.ops._load_quantized_module` consumes. Unselected
    keys pass through untouched.
    """
    import comfy.quant_ops as quant_ops

    if not quant_ops._CK_AVAILABLE:
        raise RuntimeError(
            "comfy_kitchen is unavailable in this ComfyUI - NVFP4 quantization needs it. "
            "Update ComfyUI / reinstall requirements."
        )
    if scope not in QUANTIZE_SCOPES:
        raise ValueError(f"Unknown quantize scope: {scope}")
    for k in sd:
        if k.endswith(".comfy_quant") or k.endswith("scaled_fp8"):
            raise ValueError(
                "This checkpoint is already quantized (comfy_quant / scaled_fp8 metadata "
                "found). NVFP4 patching needs the original bf16/fp16 weights."
            )

    ids = _block_ids(sd)
    keep: set[int] = set()
    if ids:
        if keep_first_blocks > 0:
            keep |= set(ids[:keep_first_blocks])
        if keep_last_blocks > 0:
            keep |= set(ids[-keep_last_blocks:])

    targets = [k for k, w in sd.items() if _eligible(k, w) and _selected(k[: -len(".weight")], scope, keep)]

    meta = torch.tensor(list(json.dumps({"format": "nvfp4"}).encode("utf-8")), dtype=torch.uint8)
    out = {}
    done = 0
    for k, w in sd.items():
        if k not in targets:
            out[k] = w
            continue
        wq = w
        if wq.dtype not in (torch.float32, torch.bfloat16):
            # fp16 and fp8-stored checkpoints alike: quantize from bf16.
            wq = wq.to(torch.bfloat16)
        if device is not None:
            wq = wq.to(device)
        qt = quant_ops.QuantizedTensor.from_float(wq, "TensorCoreNVFP4Layout", scale="recalculate")
        for kk, vv in qt.state_dict(k).items():
            out[kk] = vv.to("cpu")
        out[k[: -len("weight")] + "comfy_quant"] = meta.clone()
        del qt, wq
        done += 1
        if progress_callback is not None:
            progress_callback(done, len(targets))
        if done % 100 == 0:
            logging.info(f"[FunPack NVFP4] quantized {done}/{len(targets)} layers…")

    return out, targets, keep


def log_nvfp4_diagnostics(model, device=None):
    """Print everything needed to tell WHY NVFP4 is (or is not) fast on this machine.

    'Loaded fine' and 'runs fast' are different claims. The three ways the fast path
    silently degrades, in the order they bite:
      1. comfy_kitchen's CUDA backend is disabled when torch is built for CUDA < 13 -
         quantized matmuls then fall through to the pure-PyTorch eager backend, which
         is CORRECT but no faster than bf16 (the exact "speed didn't change" symptom).
      2. The GPU lacks FP4 tensor cores (pre-Blackwell) - core flags every nvfp4 layer
         _full_precision_mm and dequantizes (visible in the layer census below).
      3. Everything is native but the quantized scope is too small a slice of the
         total step time to move the needle.
    A timed 4096x4096 GEMM probe settles it empirically: on a native FP4 path the
    quantized matmul must beat bf16 clearly; parity or worse means emulation."""
    lines = []
    try:
        import comfy_kitchen as ck
        for name, info in ck.list_backends().items():
            if info.get("available"):
                caps = info.get("capabilities") or []
                lines.append(f"  ck backend '{name}': AVAILABLE, scaled_mm_nvfp4="
                             f"{'yes' if 'scaled_mm_nvfp4' in caps else 'NO'}")
            else:
                lines.append(f"  ck backend '{name}': unavailable ({info.get('unavailable_reason')})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  comfy_kitchen introspection failed: {e}")
    cap = torch.cuda.get_device_capability(device) if torch.cuda.is_available() else None
    lines.append(f"  torch {torch.__version__} | cuda build {torch.version.cuda} | device capability {cap}")
    if torch.version.cuda is not None:
        try:
            if tuple(map(int, str(torch.version.cuda).split(".")))[0] < 13:
                lines.append("  ⚠ torch is built for CUDA < 13 -> comfy_kitchen DISABLES its CUDA "
                             "backend -> nvfp4 matmuls run on the eager (pure PyTorch) backend: "
                             "correct but NO speedup. Install a cu13x torch build to unlock FP4 GEMMs.")
        except Exception:  # noqa: BLE001
            pass

    n_q = n_emul = 0
    try:
        for m in model.model.diffusion_model.modules():
            if getattr(m, "quant_format", None) == "nvfp4":
                n_q += 1
                if getattr(m, "_full_precision_mm", False):
                    n_emul += 1
        lines.append(f"  nvfp4 layers in loaded model: {n_q}"
                     + (f" — {n_emul} flagged FULL-PRECISION EMULATION (no FP4 compute on this GPU)"
                        if n_emul else " — all on the quantized matmul path"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"  layer census failed: {e}")

    if torch.cuda.is_available():
        try:
            import time
            import comfy.quant_ops as quant_ops
            d = torch.device(device) if device is not None else torch.device("cuda")
            w = torch.randn(4096, 4096, dtype=torch.bfloat16, device=d)
            x = torch.randn(1024, 4096, dtype=torch.bfloat16, device=d)
            qw = quant_ops.QuantizedTensor.from_float(w, "TensorCoreNVFP4Layout", scale="recalculate")

            def _timed(fn, n=30):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize(d)
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                torch.cuda.synchronize(d)
                return (time.perf_counter() - t0) / n

            t_bf16 = _timed(lambda: torch.nn.functional.linear(x, w))

            def _q():
                qx = quant_ops.QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout")
                torch.nn.functional.linear(qx, qw)

            t_q = _timed(_q)
            verdict = ("FAST PATH ACTIVE" if t_q < t_bf16 * 0.8
                       else "NO SPEEDUP — emulated or overhead-bound")
            lines.append(f"  GEMM probe 4096x4096 @ bs1024 (incl. dynamic input quant): "
                         f"bf16 {t_bf16 * 1e3:.3f} ms vs nvfp4 {t_q * 1e3:.3f} ms "
                         f"({t_bf16 / max(t_q, 1e-9):.2f}x) -> {verdict}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  GEMM probe failed: {e}")
    else:
        lines.append("  GEMM probe skipped (no CUDA device)")

    print("[FunPack NVFP4] diagnostics:\n" + "\n".join(lines))
    return {"nvfp4_layers": n_q, "emulated_layers": n_emul}


class FunPackNVFP4ModelLoader:
    """EXPERIMENTAL: load a diffusion model with on-the-fly NVFP4 quantization.

    Drop-in replacement for 'Load Diffusion Model' aimed at LTX-AV on Blackwell
    (sm_120+): selected transformer Linears run real FP4 tensor-core GEMMs (~2x fp8
    throughput) and weight VRAM drops to ~1/4 of bf16. Audio branch, cross-modal
    bridges, first/last blocks, and all non-matmul weights stay at full precision
    by default. On GPUs without FP4 compute the model still works via emulation.
    """

    CATEGORY = "FunPack/Experimental"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),
                              {"tooltip": "Original bf16/fp16 checkpoint - NOT a pre-quantized one."}),
                "quantize_scope": (QUANTIZE_SCOPES, {
                    "default": "video blocks",
                    "tooltip": "Which Linears go NVFP4. 'video blocks' keeps the whole audio "
                               "path + cross-modal bridges at full precision (safest). Widen "
                               "only if quality holds.",
                }),
                "keep_first_blocks": ("INT", {"default": 2, "min": 0, "max": 16,
                                              "tooltip": "First N transformer blocks stay full precision."}),
                "keep_last_blocks": ("INT", {"default": 2, "min": 0, "max": 16,
                                             "tooltip": "Last N transformer blocks stay full precision."}),
            }
        }

    def load(self, unet_name, quantize_scope, keep_first_blocks, keep_last_blocks):
        import comfy.model_management as mm
        import comfy.sd
        import comfy.utils
        import folder_paths

        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        # return_metadata is NOT optional here: LTX-AV variants carry their architecture
        # config in the safetensors metadata, and detection without it builds the WRONG
        # model (6 vs 9 ada params, connector dims) -> size-mismatch wall on load.
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)

        if mm.supports_nvfp4_compute(mm.get_torch_device()):
            logging.info("[FunPack NVFP4] FP4 tensor-core compute available - native GEMM path.")
        else:
            logging.warning(
                "[FunPack NVFP4] This GPU has no FP4 compute (needs Blackwell, sm_100+). "
                "The model will load and run through dequant emulation: correct output, "
                "no speedup. Use this only for A/B quality checks."
            )

        device = mm.get_torch_device() if torch.cuda.is_available() else None
        pbar = comfy.utils.ProgressBar(1)

        def _tick(done, total):
            if total:
                pbar.total = total
                pbar.update_absolute(done)

        sd, targets, keep = quantize_state_dict_nvfp4(
            sd, scope=quantize_scope,
            keep_first_blocks=keep_first_blocks, keep_last_blocks=keep_last_blocks,
            device=device, progress_callback=_tick,
        )
        if not targets:
            raise RuntimeError(
                "No layers matched the NVFP4 quantization scope - is this an LTX-AV "
                "diffusion model checkpoint? (Expected transformer_blocks.* keys.)"
            )
        logging.info(
            f"[FunPack NVFP4] quantized {len(targets)} Linear layers "
            f"(scope='{quantize_scope}', full-precision blocks: {sorted(keep) or 'none'})."
        )

        model = comfy.sd.load_diffusion_model_state_dict(sd, metadata=metadata)
        if model is None:
            raise RuntimeError("ComfyUI could not detect the model type after NVFP4 patching.")
        try:
            log_nvfp4_diagnostics(model, device)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"[FunPack NVFP4] diagnostics failed: {e}")
        return (model,)
