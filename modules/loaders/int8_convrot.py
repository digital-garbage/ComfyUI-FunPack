"""Int8 + rotation weights for MiniMax H3, made from an ordinary (bf16/fp16) file at load.

Comfy-Org ships H3 already quantized this way (half the memory, int8 matmuls on the tensor cores). This
does the same to a file that is not: each big Linear's weight is rotated (Hadamard, in groups) so no
weight has a lone huge value, then stored as int8 with one scale per output row -- in exactly the layout
a pre-quantized file has, so ComfyUI's own loader and kernels take it from there. Rotation is what
spares the outliers that broke plain fp8 (NaN audio), but the result is still an approximation of the
file: unvalidated on a GPU.

Only the transformer's attention and MLP projections are touched. The timestep modulation (adaLN, ~40% of
the bytes but tiny rows), embeddings, the output layer and the text refiner stay as they are.
"""

import json
import re

import torch

# blocks.N.attn.{qkv_proj,out_proj} / blocks.N.mlp.{fc1,fc2}, not token_refiner.blocks.*
_BIG = re.compile(r"(?:^|\.)(?<!refiner\.)blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$")
GROUP = 256
_FLOATS = (torch.bfloat16, torch.float16, torch.float32)
_TAG = torch.tensor(list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()), dtype=torch.uint8)


def quantize_state_dict(sd, device=None):
    """Rewrite `sd` in place. -> (layers converted, note). Already-quantized layers and rows that cannot be
    rotated in groups (width not a multiple of 256) are left as they are, and the note says so."""
    from comfy.quant_ops import QuantizedTensor
    done, odd = 0, 0
    for key in [k for k in sd if _BIG.search(k)]:
        w = sd[key]
        layer = key[:-len(".weight")]
        if w.dtype not in _FLOATS or w.ndim != 2 or f"{layer}.comfy_quant" in sd:
            continue
        if w.shape[1] % GROUP:
            odd += 1
            continue
        q = QuantizedTensor.from_float(w.to(device) if device is not None else w, "TensorWiseINT8Layout",
                                       per_channel=True, convrot=True, convrot_groupsize=GROUP, scale="recalculate")
        for name, tensor in q.state_dict(key).items():
            sd[name] = tensor.cpu()
        sd[f"{layer}.comfy_quant"] = _TAG.clone()
        done += 1
    if not done:
        return 0, "int8_convrot: no H3 attention/MLP weights found to convert (not an H3 file, or already quantized) -- loaded as is"
    return done, f"int8_convrot: {done} layers stored as int8" + (f"; {odd} left as they were (width not a multiple of {GROUP})" if odd else "")
