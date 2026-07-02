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
        return (model,)
