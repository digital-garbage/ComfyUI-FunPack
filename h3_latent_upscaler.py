"""MiniMax H3's latent upscaler: the 3D-conv resizer, loadable without a node pack.

ComfyUI's own Load Latent Upscale Model knows three architectures (Lightricks'
LatentUpsampler and two Hunyuan SR variants) and this is a fourth, so an H3 upsampler
file could not be loaded at all — which is what blocked `second_pass_op` on H3: the
operation needs an upsampler whose latents are the same width as the model's, and the only
published 24-channel one lives behind its own custom node.

The architecture is dictated by the checkpoint, not chosen here: every module name below
exists to match a key in the published state dict, and the shape of each is read back off
the weights (`detect_config`) rather than configured. Latent normalisation uses the model's
own published per-channel statistics — H3's VAE statistics are a different thing and would
be wrong here.

Reference implementation: LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler (the companion node
for LBH-123-AI/Minimax_h3_latent_Upscaler). Reconstructed rather than vendored — that
repository ships no licence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-channel statistics the upscaler was trained against, from the model's own card. The
# net sees normalised latents and returns normalised latents; skipping this does not
# degrade the output so much as produce a different distribution entirely.
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]

_EMBED_DIM = 64
_GROUPS = 32


def _norm(channels):
    return nn.GroupNorm(_GROUPS, channels)


class _ResBlockEmb3D(nn.Module):
    """Residual block conditioned on the scale factor (FiLM-style shift/scale)."""

    def __init__(self, channels, emb_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _norm(channels), nn.SiLU(), nn.Conv3d(channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels))
        self.out_norm = _norm(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=0.0),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip = (nn.Conv3d(channels, self.out_channels, 1)
                     if self.out_channels != channels else nn.Identity())

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.dim() < h.dim():
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class _TemporalConv(nn.Module):
    """Depthwise conv along time only — what makes this a video upscaler rather than a
    per-frame image one, and why an upscaled clip does not shimmer frame to frame."""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                                padding=(kernel_size // 2, 0, 0), groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class H3LatentResizer3D(nn.Module):
    """Refine, resample, refine. The resample is a plain trilinear interpolation; the
    learned part is what the blocks do on either side of it."""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.in_channels = in_channels
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, _EMBED_DIM), nn.SiLU(), nn.Linear(_EMBED_DIM, _EMBED_DIM))

        def stack(count):
            blocks = nn.ModuleList()
            for index in range(count):
                blocks.append(_ResBlockEmb3D(channels, _EMBED_DIM))
                if temporal_every > 0 and index % temporal_every == 0:
                    blocks.append(_TemporalConv(channels, temporal_kernel))
            return blocks

        # Interleaved in one flat list, so the index a key names depends on this exact
        # construction order — it is part of the format, not a style choice.
        self.in_blocks = stack(in_blocks)
        self.out_blocks = stack(out_blocks)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _run(self, blocks, x, emb):
        for block in blocks:
            if isinstance(block, _ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        return x

    def forward(self, x, scale=None, target_size=None):
        if target_size is not None:
            size = tuple(int(v) for v in target_size)
        elif scale is not None:
            size = tuple(int(round(dim * scale)) for dim in x.shape[-3:])
        else:
            return x
        if size == tuple(x.shape[-3:]):
            return x

        # The block conditioning is scale-1, so an unchanged size embeds as zero.
        embed_in = torch.tensor([[0.0 if scale is None else float(scale) - 1.0]],
                                dtype=x.dtype, device=x.device)
        emb = self.embed(embed_in)

        x = self._run(self.in_blocks, self.conv_in(x), emb)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        x = self._run(self.out_blocks, x, emb)
        return self.conv_out(F.silu(self.norm_out(x)))

    # Marks this as an upscaler that owns its normalisation and takes a scale factor,
    # so the caller does not have to know which architecture it loaded.
    def funpack_latent_upscale(self, x, scale=2.0):
        mean = torch.tensor(LATENTS_MEAN, dtype=x.dtype, device=x.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, dtype=x.dtype, device=x.device).view(1, -1, 1, 1, 1)
        if mean.shape[1] != x.shape[1]:
            raise ValueError(
                f"this upscaler's statistics are {mean.shape[1]}-channel and the latent is "
                f"{x.shape[1]}-channel")
        # Spatial only: H3's video latent is on a 5k+2 time grid and resampling it would
        # leave a frame count the VAE has no defined decode for.
        target = (int(x.shape[2]), int(round(x.shape[3] * scale)), int(round(x.shape[4] * scale)))
        out = self((x - mean) / std, scale=scale, target_size=target)
        return out * std + mean


def is_h3_latent_upscaler(sd):
    """Whether a state dict is this architecture. Checked before ComfyUI's own loader,
    which has no branch for it and fails with a Python error rather than a diagnosis."""
    sd = strip_prefix(sd)
    return "conv_in.weight" in sd and any(k.startswith("in_blocks.") for k in sd) \
        and "post_upsample_res_blocks.0.conv2.bias" not in sd


def strip_prefix(sd):
    """Some checkpoints wrap the net under `upscaler.` (and a `model` top-level key)."""
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(str(k).startswith("upscaler.") for k in sd):
        return {k[len("upscaler."):]: v for k, v in sd.items() if str(k).startswith("upscaler.")}
    return sd


def detect_config(sd):
    """Read the architecture back off the weights — width, depth and temporal layout are
    all recoverable, so there is no config to carry and no version to get wrong."""
    import re

    config = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
              "temporal_every": 2, "temporal_kernel": 5}
    conv_in = sd.get("conv_in.weight")
    if conv_in is not None:
        config["channels"] = int(conv_in.shape[0])
        config["in_channels"] = int(conv_in.shape[1])

    res, temporal = {"in": set(), "out": set()}, False
    for key in sd:
        match = re.match(r"(in|out)_blocks\.(\d+)\.in_layers\.", str(key))
        if match:
            res[match.group(1)].add(int(match.group(2)))
        if str(key).endswith("dwconv.weight"):
            temporal = True
            config["temporal_kernel"] = int(sd[key].shape[2])
    if res["in"]:
        config["in_blocks"] = len(res["in"])
    if res["out"]:
        config["out_blocks"] = len(res["out"])
    if not temporal:
        config["temporal_every"] = 0
    return config


def from_state_dict(sd, dtype=None):
    """Build the resizer this state dict describes and load it."""
    sd = strip_prefix(sd)
    if any(".q.weight" in str(k) or ".proj_out.weight" in str(k) for k in sd):
        raise ValueError(
            "this H3 latent upscaler has attention blocks, which this loader does not build "
            "— use the companion custom node for that checkpoint")
    model = H3LatentResizer3D(**detect_config(sd))
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model
