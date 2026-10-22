"""A tiny real native MiniMax-H3, and PTD adapter files in the three layouts, for the PTD tests.

The model builder and file layouts follow ComfyUI-PTD-Compact's tests (Nynxz, Apache-2.0).
"""

import json

import torch
from safetensors.torch import save_file

DIMS = dict(hidden_size=64, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2, attention_head_dim=32,
            ffn_hidden_size=48, text_dim=40, timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=16,
            rope_inv_freq_len=4)
PARTS = {"attn.to_q": ("attn.qkv_proj", 3), "attn.to_k": ("attn.qkv_proj", 3), "attn.to_v": ("attn.qkv_proj", 3),
         "attn.to_out.0": ("attn.out_proj", 1), "ff.net.0.proj": ("mlp.fc1", 1), "ff.net.2": ("mlp.fc2", 1)}
BLOCKS = (("transformer_blocks", "blocks", lambda n: len(n.blocks)),
          ("token_refiner.refiner_blocks", "token_refiner.blocks", lambda n: len(n.token_refiner.blocks)))


def native_model(curve=False, **override):
    """A ModelPatcher around a real MiniMaxH3 (random weights, CPU). curve=True is the pruned layout."""
    import comfy.model_patcher
    import comfy.supported_models
    extra = {"adaln_curve_grid": 9} if curve else {}
    config = comfy.supported_models.MiniMaxH3(dict(image_model="minimax_h3", **{**DIMS, **override}, **extra))
    config.set_inference_dtype(torch.float32, None)
    model = config.get_model({})
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.diffusion_model.parameters():
            p.normal_(0, 0.05)
        dm = model.diffusion_model
        n = dm.rope.inv_freq.numel()
        dm.rope.inv_freq.copy_(1.0 / 10000 ** (torch.arange(n, dtype=torch.float32) / n))
        if curve:
            dm.adaln_t_table.normal_(0, 0.5)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(model, cpu, cpu)


def _int8(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    return (x / s).round().clamp(-127, 127).to(torch.int8), s.squeeze(-1)


def epfl_file(path, native, hops=4, int8=False, zero_a=False, ranks=(3, 4, 5)):
    """An EPFL-layout file; returns {(native module, part, hop): dense delta}."""
    g = torch.Generator().manual_seed(1)
    tensors, dense = {}, {}
    for src, dst, count in BLOCKS:
        for i in range(count(native)):
            for part, (module, div) in PARTS.items():
                layer = native.get_submodule(f"{dst}.{i}.{module}")
                for hop in range(hops):
                    r = ranks[(hop + len(part)) % len(ranks)]
                    A = torch.randn(r, layer.in_features, generator=g) * 0.1 * (0 if zero_a else 1)
                    B = torch.randn(layer.out_features // div, r, generator=g) * 0.1
                    key = f"transformer.{src}.{i}.{part}"
                    if int8:
                        qa, sa = _int8(A)
                        qb, sb = _int8(B)
                        tensors[f"{key}.lora_A.{hop}.weight"], tensors[f"{key}.lora_A.{hop}.weight_scale"] = qa, sa[:, None]
                        tensors[f"{key}.lora_B.{hop}.weight"], tensors[f"{key}.lora_B.{hop}.weight_scale"] = qb, sb[:, None]
                        A, B = qa.float() * sa[:, None], qb.float() * sb[:, None]
                    else:
                        tensors[f"{key}.lora_A.{hop}.weight"], tensors[f"{key}.lora_B.{hop}.weight"] = A, B
                    dense[(f"{dst}.{i}.{module}", part, hop)] = B @ A
    for head, native_head in (("proj_out", "video_out"), ("audio_proj_out", "audio_out")):
        layer = native.final_layer.get_submodule(native_head)
        for hop in range(hops):
            A = torch.randn(layer.out_features, layer.in_features, generator=g) * 0.01 * (0 if zero_a else 1)
            tensors[f"transformer.{head}.main.lora_A.{hop}.weight"] = A
            tensors[f"transformer.{head}.main.lora_B.{hop}.weight"] = torch.eye(layer.out_features)
            dense[(f"final_layer.{native_head}", "head", hop)] = A
    meta = {"ptd_format": "comfyui-native-h3-compact" if int8 else "comfyui-native-h3",
            "ptd_recipe": f"h3_{hops}step"}
    save_file(tensors, str(path), metadata=meta)
    return dense


def pdd_file(path, native, hops=4, steps=32, rank=8, int8=False, zero=False):
    """A Ref2VA-layout file: factors stacked over hops, plus a bank of per-interval heads."""
    g = torch.Generator().manual_seed(1)
    t = {}
    for src, dst, count in BLOCKS:
        for i in range(count(native)):
            for part, (module, div) in PARTS.items():
                layer = native.get_submodule(f"{dst}.{i}.{module}")
                down = torch.randn(hops, rank, layer.in_features, generator=g) * 0.1 * (0 if zero else 1)
                up = torch.randn(hops, layer.out_features // div, rank, generator=g) * 0.1
                key = f"{src}.{i}.{part}"
                if int8:
                    (t[f"{key}.lora_down"], t[f"{key}.lora_down_scale"]) = _int8(down)
                    (t[f"{key}.lora_up"], t[f"{key}.lora_up_scale"]) = _int8(up)
                else:
                    t[f"{key}.lora_down"], t[f"{key}.lora_up"] = down.bfloat16(), up.bfloat16()
    for prefix, head in (("proj_out", native.final_layer.video_out), ("audio_proj_out", native.final_layer.audio_out)):
        w = head.weight.detach().float()
        t[f"{prefix}.weight"] = (w[None] + torch.randn(steps, *w.shape, generator=g) * 0.05).bfloat16()
        t[f"{prefix}.bias"] = (torch.randn(steps, w.shape[0], generator=g) * 0.05).bfloat16()
    meta = {"ptd_num_hops": str(hops), "pdd_num_steps": str(steps), "lora_rank": str(rank), "lora_alpha": f"{rank}.0"}
    if int8:
        meta["ptd_compact"] = json.dumps({"schema": 1})
    save_file(t, str(path), metadata=meta)
