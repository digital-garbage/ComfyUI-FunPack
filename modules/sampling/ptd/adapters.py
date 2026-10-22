"""Reading a PTD adapter file into a bank of per-hop experts.

Three families load here:

* EPFL fl2va releases (bf16, or int8 with per-row scales): keys `transformer.<module>.lora_A.<hop>.weight`.
* taobao-content-ai Ref2VA releases (bf16, or int8): factors stacked over hops (`lora_down [H, r, in]`)
  plus a bank of per-interval output heads that is fused per hop.
* Any rank: reduced-rank files simply have smaller factors.

The key mapping, topology checks and head fusion follow ComfyUI-PTD (EPFL VITA Lab) and
ComfyUI-MiniMaxH3-PTD (taobao-content-ai), both Apache-2.0, via ComfyUI-PTD-Compact (Nynxz).
"""

from collections import defaultdict
from dataclasses import dataclass, field
import re

import torch

from . import schedule

HEADS = {"proj_out": "final_layer.video_out", "audio_proj_out": "final_layer.audio_out"}
PARTS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_QKV = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
_DENSE = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
#: Reference tensors some EPFL files carry, to prove the loaded base is the one they were trained on.
BUFFERS = {
    "transformer.rope.inv_freq": "rope.inv_freq",
    "transformer.proj_out.orig_weight": "final_layer.video_out.weight",
    "transformer.proj_out.orig_bias": "final_layer.video_out.bias",
    "transformer.audio_proj_out.orig_weight": "final_layer.audio_out.weight",
    "transformer.audio_proj_out.orig_bias": "final_layer.audio_out.bias",
}
_EPFL_KEY = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<factor>[AB])\.(?P<hop>0|[1-9][0-9]*)\.weight$")
_MODULE = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class Expert:
    a: torch.Tensor                  # [r, in]    int8 or float
    b: torch.Tensor                  # [out, r]   int8 or float
    sa: torch.Tensor | None          # [r]        float32 row scales of a
    sb: torch.Tensor | None          # [out]      float32 row scales of b
    qkv_index: int | None


@dataclass
class Adapter:
    label: str
    hops: int
    grid: tuple                      # hop times t_0 > ... > 0
    audio: tuple                     # per hop, the factor the audio head is scaled by
    bank: dict                       # native module -> {hop: (Expert, ...)}
    heads: list = field(default_factory=list)   # per hop {native head: (weight, bias)} (Ref2VA only)
    int8: bool = False
    buffers: dict = field(default_factory=dict)   # reference tensors of the base (EPFL files)

    @property
    def size(self):
        return sum(t.numel() * t.element_size() for by_hop in self.bank.values()
                   for experts in by_hop.values() for e in experts for t in (e.a, e.b))


def native_target(module):
    """Diffusers module name -> (native module, qkv index)."""
    m = _MODULE.fullmatch(module)
    if m is None:
        raise ValueError(f"Unknown PTD adapter module: {module}")
    base = ("blocks." if m[1] == "transformer_blocks" else "token_refiner.blocks.") + m[2]
    if m[3] in _QKV:
        return f"{base}.attn.qkv_proj", _QKV[m[3]]
    if m[3] not in _DENSE:
        raise ValueError(f"Unknown PTD adapter projection: {module}")
    return f"{base}.{_DENSE[m[3]]}", None


def _add(bank, name, hop, a, b, sa, sb, qkv):
    if name.endswith(".mlp.fc1"):    # Diffusers stores [up, gate]; native H3 consumes [gate, up]
        b = torch.cat(b.chunk(2, dim=0)[::-1], dim=0)
        sb = None if sb is None else torch.cat(sb.chunk(2, dim=0)[::-1], dim=0)
    bank[name][hop].append(Expert(a.contiguous(), b.contiguous(),
                                  None if sa is None else sa.reshape(-1).float(),
                                  None if sb is None else sb.reshape(-1).float(), qkv))


def _epfl_bank(tensors, hops):
    bank = defaultdict(lambda: defaultdict(list))
    for key in tensors:
        if key.endswith("_scale") or key in BUFFERS:
            continue
        m = _EPFL_KEY.fullmatch(key)
        if m is None:
            raise ValueError(f"Not a supported PTD adapter key: {key}")
        if m["factor"] != "A":
            continue
        hop, module = int(m["hop"]), m["module"]
        if hop >= hops:
            raise ValueError(f"Adapter hop {hop} is beyond this file's {hops} hops: {key}")
        b_key = key.replace(".lora_A.", ".lora_B.")
        a, b = tensors[key], tensors[b_key]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"Invalid PTD adapter pair: {key}")
        if module.endswith(".main"):
            name, qkv = HEADS[module[:-len(".main")]], None
        else:
            name, qkv = native_target(module)
        _add(bank, name, hop, a, b, tensors.get(key + "_scale"), tensors.get(b_key + "_scale"), qkv)
    return bank


def _pdd_bank(tensors, hops, scale):
    bank = defaultdict(lambda: defaultdict(list))
    for module in sorted({k.rsplit(".lora_", 1)[0] for k in tensors if ".lora_" in k}):
        name, qkv = native_target(module)
        down, up = tensors[f"{module}.lora_down"], tensors[f"{module}.lora_up"]
        sd, su = tensors.get(f"{module}.lora_down_scale"), tensors.get(f"{module}.lora_up_scale")
        if down.ndim != 3 or up.ndim != 3 or down.shape[0] != hops or up.shape[0] != hops:
            raise ValueError(f"{module}: expected factors stacked over {hops} hops")
        for h in range(hops):
            sb = None if su is None else su[h].float()
            if scale != 1.0:
                sb = torch.full((up.shape[1],), scale) if sb is None else sb * scale
            _add(bank, name, h, down[h], up[h], None if sd is None else sd[h].float(), sb, qkv)
    return bank


def _pdd_heads(tensors, hops, steps):
    """Per hop, the bank of per-interval heads fused with that hop's normalised step sizes."""
    banks = {}
    for prefix, native in HEADS.items():
        if f"{prefix}.weight" not in tensors:
            raise ValueError(f"PTD file is missing the output head bank {prefix}.weight")
        w, b = tensors[f"{prefix}.weight"], tensors.get(f"{prefix}.bias")
        if w.ndim != 3 or w.shape[0] != steps or (b is not None and tuple(b.shape) != (steps, w.shape[1])):
            raise ValueError(f"Output head bank {prefix} must be ({steps}, out, in)")
        banks[native] = (w.float(), None if b is None else b.float())
    heads = []
    for video_plan, audio_plan in schedule.pdd_plans(hops, steps):
        fused = {}
        for name, (w, b) in banks.items():
            plan = video_plan if name.endswith("video_out") else audio_plan
            fused[name] = (torch.einsum("n,noi->oi", plan, w),
                           None if b is None else torch.einsum("n,no->o", plan, b))
        heads.append(fused)
    return heads


def _expected(native, head_experts):
    names = set()
    for prefix, count in (("blocks", len(native.blocks)), ("token_refiner.blocks", len(native.token_refiner.blocks))):
        for i in range(count):
            names |= {f"{prefix}.{i}.attn.qkv_proj", f"{prefix}.{i}.attn.out_proj",
                      f"{prefix}.{i}.mlp.fc1", f"{prefix}.{i}.mlp.fc2"}
    return names | set(HEADS.values()) if head_experts else names


def verify(adapter, native):
    """Raise, in plain words, unless `adapter` fits the loaded H3 model `native` layer for layer."""
    bank, hops = adapter.bank, adapter.hops
    expected = _expected(native, head_experts=not adapter.heads)
    if set(bank) != expected:
        raise ValueError(f"This adapter does not match the loaded model's layers: {len(expected - set(bank))} "
                         f"missing, {len(set(bank) - expected)} unexpected")
    for name, by_hop in bank.items():
        module = native.get_submodule(name)
        if set(by_hop) != set(range(hops)):
            raise ValueError(f"PTD adapter is missing hops for {name}")
        for experts in by_hop.values():
            if len(experts) != (3 if name.endswith("qkv_proj") else 1):
                raise ValueError(f"PTD adapter has the wrong number of factors for {name}")
            for e in experts:
                if e.a.shape[1] != module.in_features or \
                        e.b.shape[0] * (3 if e.qkv_index is not None else 1) != module.out_features:
                    raise ValueError(f"PTD adapter does not fit {name}: it was made for another H3 variant")
    for name in HEADS.values() if adapter.heads else ():
        head = native.get_submodule(name)
        if (head.in_features, head.out_features) != tuple(reversed(adapter.heads[0][name][0].shape)):
            raise ValueError(f"The output head bank does not fit {name}: it was made for another H3 variant")
    for key, target in BUFFERS.items():
        if key not in adapter.buffers:
            continue
        parent, _, attr = target.rpartition(".")
        base = getattr(native.get_submodule(parent), attr)
        if not isinstance(base, torch.Tensor) or base.device.type == "meta" or \
                base.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            continue            # a quantised base cannot be compared; the layer checks above still apply
        value = adapter.buffers[key]
        if base.shape != value.shape or not torch.equal(base.detach().cpu(), value.to("cpu", base.dtype)):
            raise ValueError(f"The loaded MiniMax-H3 base is not the one this adapter was trained on ({target})")


def read(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    for key, value in tensors.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite values in {key}")
    return tensors, meta


def parse(path, label=None):
    """The Adapter in `path`. It knows nothing of any model yet: `verify` it against one before use."""
    tensors, meta = read(path)
    buffers = {k: v for k, v in tensors.items() if k in BUFFERS}
    if "ptd_num_hops" in meta:                                  # Ref2VA release
        hops = int(meta["ptd_num_hops"])
        steps = int(meta.get("pdd_num_steps", schedule.PDD_STEPS))
        if hops < 1 or steps % hops:
            raise ValueError(f"pdd_num_steps={steps} is not divisible by ptd_num_hops={hops}")
        rank = int(meta.get("lora_rank", 128))
        bank = _pdd_bank(tensors, hops, float(meta.get("lora_alpha", rank)) / rank)
        heads = _pdd_heads(tensors, hops, steps)
        grid = schedule.pdd_grid(hops, steps)
        audio = schedule.audio_chords(grid)
    elif meta.get("ptd_recipe") in schedule.RECIPES:             # EPFL release
        grid = schedule.RECIPES[meta["ptd_recipe"]]
        hops = len(grid) - 1
        bank = _epfl_bank(tensors, hops)
        heads = []
        audio = schedule.audio_tangents(grid)
    else:
        raise ValueError("Not a PTD adapter file: it names neither ptd_recipe nor ptd_num_hops")
    bank = {name: {h: tuple(e) for h, e in by_hop.items()} for name, by_hop in bank.items()}
    int8 = any(e.a.dtype == torch.int8 for by_hop in bank.values() for es in by_hop.values() for e in es)
    return Adapter(label or str(path), hops, grid, audio, bank, heads, int8, buffers)
