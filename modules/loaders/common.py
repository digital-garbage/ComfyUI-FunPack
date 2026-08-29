"""Shared loading options.

Not a module: it announces nothing and ships no node. `registry.scan` only walks
two levels of DIRECTORIES, so a plain file at the domain level is a helper the
loaders import and nothing else.

These options are on the loader node rather than in a settings payload, and the
line is worth stating: a loader widget describes the FILE and how it is read,
which the loader cannot be handed by anyone else. Feature knobs -- strengths,
sigma windows, anything a modifier owns -- are settings, and never appear here.
"""

import logging

import torch

# Named exactly as core's UNETLoader names them where they overlap, so a value
# copied from a stock workflow means the same thing here.
WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"]
COMPUTE_DTYPES = ["default", "fp16", "bf16", "fp32"]
VAE_DTYPES = ["bfloat16", "float16", "float32"]

_BY_NAME = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2": getattr(torch, "float8_e5m2", None),
    # fp8_e4m3fn plus comfy's fp8 matmul path; see weight_model_options.
    "fp8_e4m3fn_fast": getattr(torch, "float8_e4m3fn", None),
}


def dtype_of(name):
    """The torch dtype a choice names, or None for 'default'/an unsupported build."""
    return _BY_NAME.get(str(name or "default"))


def weight_model_options(weight_dtype):
    """comfy `model_options` for a weight dtype. The mapping core's UNETLoader uses."""
    options = {}
    dtype = dtype_of(weight_dtype)
    if dtype is not None:
        options["dtype"] = dtype
    if weight_dtype == "fp8_e4m3fn_fast":
        options["fp8_optimizations"] = True
    return options


def attention_choices():
    """Backends this ComfyUI can actually run.

    Read from ComfyUI's own registry rather than a private import, so the list
    offers exactly what the machine has -- sage where SageAttention is installed,
    flash where flash-attn is -- and never lies about the rest. 'default' leaves
    whatever ComfyUI itself selected from its launch flags.
    """
    names = []
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS
        names = sorted(REGISTERED_ATTENTION_FUNCTIONS.keys())
    except Exception:                            # noqa: BLE001 -- older ComfyUI
        pass
    return ["default"] + names


def attention_override(name):
    """A transformer_options override routing every attention call to `name`.

    ComfyUI wraps each implementation with `wrap_attn`, which hands the override
    the original function plus the call's arguments -- so calling the WRAPPED
    replacement would re-enter that machinery. The unwrapped one is what gets
    called.
    """
    if not name or name == "default":
        return None
    try:
        from comfy.ldm.modules.attention import get_attention_function
    except ImportError:
        logging.warning("[FunPack] this ComfyUI has no attention registry; leaving attention alone")
        return None
    chosen = get_attention_function(name, None)
    if chosen is None:
        logging.warning("[FunPack] attention backend %r is not available; leaving attention alone", name)
        return None
    inner = getattr(chosen, "__wrapped__", chosen)

    def override(_func, *args, **kwargs):
        return inner(*args, **kwargs)

    return override


def set_fp16_accumulation(enabled):
    """torch's fp16 accumulation switch, or None where the build has no such knob."""
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return None
    matmul.allow_fp16_accumulation = bool(enabled)
    return bool(enabled)
