import json

import torch

from .. import int8_convrot as m


def _sd():
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g, dtype=torch.bfloat16)
    return {
        "blocks.0.attn.qkv_proj.weight": r(96, 512), "blocks.0.attn.out_proj.weight": r(32, 512),
        "blocks.0.mlp.fc1.weight": r(64, 512), "blocks.0.mlp.fc2.weight": r(32, 100),     # 100 is not a multiple of 256
        "blocks.0.adaln_proj.linear.weight": r(64, 128), "blocks.0.adaln_proj.linear.bias": r(64),
        "token_refiner.blocks.0.attn.qkv_proj.weight": r(96, 512), "video_patch_proj.weight": r(32, 512),
    }


def test_only_the_big_projections_are_converted():
    sd = _sd()
    keep = {k: v.clone() for k, v in sd.items()}
    n, note = m.quantize_state_dict(sd)
    assert n == 3 and "1 left" in note
    for k in ("blocks.0.attn.qkv_proj", "blocks.0.attn.out_proj", "blocks.0.mlp.fc1"):
        assert sd[k + ".weight"].dtype == torch.int8 and sd[k + ".weight_scale"].shape == (sd[k + ".weight"].shape[0], 1)
        conf = json.loads(sd[k + ".comfy_quant"].numpy().tobytes())
        assert conf == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    for k in ("blocks.0.mlp.fc2.weight", "blocks.0.adaln_proj.linear.weight", "token_refiner.blocks.0.attn.qkv_proj.weight", "video_patch_proj.weight"):
        assert torch.equal(sd[k], keep[k])


def test_second_pass_and_foreign_files_are_left_alone():
    sd = _sd()
    m.quantize_state_dict(sd)
    again = {k: v.clone() for k, v in sd.items()}
    assert m.quantize_state_dict(sd)[0] == 0 and all(torch.equal(sd[k], again[k]) for k in sd)
    n, note = m.quantize_state_dict({"unet.weight": torch.zeros(4, 4)})
    assert n == 0 and "loaded as is" in note


def test_comfy_loads_it_and_the_output_matches():
    import comfy.ops
    sd = {"blocks.0.mlp.fc1.weight": torch.randn(64, 512, dtype=torch.bfloat16)}
    ref = sd["blocks.0.mlp.fc1.weight"].clone()
    m.quantize_state_dict(sd)
    lin = comfy.ops.mixed_precision_ops({"mixed_ops": True}, torch.bfloat16).Linear(512, 64, bias=False)
    lin.load_state_dict({k.split("fc1.")[1]: v for k, v in sd.items()}, strict=False)
    x = torch.randn(3, 512, dtype=torch.bfloat16)
    want = torch.nn.functional.linear(x, ref)
    got = lin(x)
    assert (got.float() - want.float()).abs().mean() / want.float().abs().mean() < 0.03
