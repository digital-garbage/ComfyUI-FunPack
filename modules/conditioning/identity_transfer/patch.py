"""Putting a reference face INTO an LTX model's token sequence.

The reference is appended as separate tokens after the video's own (never blended into a real frame),
sharing the target's frame-0 coordinate grid, given a clean timestep (they are an image, not noise),
tagged with a source-phase RoPE rotation so the model can tell them from the real frame 0, and cut
off again before the output is unpatchified -- exactly what the identity LoRAs were trained on.

Four methods of the LTX model are replaced, one per stage of its own `_forward`:
`_process_input` (append) -> `_prepare_timestep` (clean timestep for them) ->
`_prepare_positional_embeddings` (the source-phase tag) -> `_process_output` (cut them off).
They are object patches, so they exist only while the model is loaded for sampling, and every one is
tagged so a model that goes round again sheds them. What the stages share (how many tokens were
appended) lives in this install's own closure, never on the shared model.
"""

import copy

import torch

from ..._core import log, patching
from .rope import rotate_overlap_freqs

_PREFIX = "diffusion_model."


def _say(what, exc):
    # Said once per install, never silently: an untagged reference reads as the real first frame
    # (the clip opens on the reference image), and a half-applied patch looks like a working one.
    log.once(f"identity_transfer:{what}", log.ALERT, "FunPack Identity Transfer",
             f"{what} failed ({type(exc).__name__}: {exc}) -- identity conditioning is NOT applied "
             f"correctly this run. This usually means a ComfyUI update changed the LTX model internals.")


def install(patcher, ref_latent, seg_value, key):
    """Hang the four stages on `patcher`'s model. Raises when this is not an LTX model."""
    names = ("_process_input", "_prepare_timestep", "_prepare_positional_embeddings", "_process_output")
    original = {}
    for name in names:
        fn = patcher.get_model_object(_PREFIX + name)
        if not callable(fn):
            raise RuntimeError(f"this model has no {name}: identity transfer needs an LTX audio+video model")
        original[name] = fn
    ltxv = patcher.get_model_object("diffusion_model")
    state = {"ref_len": 0, "target_len": None}

    def process_input(x, keyframe_idxs, denoise_mask, **kw):
        out = original["_process_input"](x, keyframe_idxs, denoise_mask, **kw)
        state["ref_len"] = 0
        try:
            from comfy.ldm.lightricks.model import latent_to_pixel_coords
            xx, pix, add = out
            is_av = isinstance(xx, (list, tuple))
            vx = xx[0] if is_av else xx
            vco = pix[0] if is_av else pix
            rt, rlc = ltxv.patchifier.patchify(ref_latent.to(dtype=vx.dtype, device=vx.device))
            rpc = latent_to_pixel_coords(latent_coords=rlc, scale_factors=ltxv.vae_scale_factors,
                                         causal_fix=ltxv.causal_temporal_positioning)
            rt = ltxv.patchify_proj(rt)
            if rt.shape[0] != vx.shape[0]:
                rt = rt.expand(vx.shape[0], -1, -1)
            if rpc.shape[0] != vco.shape[0]:
                rpc = rpc.expand(vco.shape[0], *([-1] * (rpc.dim() - 1)))
            state["target_len"] = vx.shape[1]
            vx = torch.cat([vx, rt], dim=1)
            vco = torch.cat([vco, rpc.to(vco)], dim=2)
            state["ref_len"] = rt.shape[1]
            if is_av:
                xx, pix = [vx, xx[1]], [vco, pix[1]]
            else:
                xx, pix = vx, vco
            return xx, pix, add
        except Exception as exc:                    # noqa: BLE001
            _say("appending the reference tokens", exc)
            state["ref_len"] = 0
            return out

    def prepare_timestep(timestep, batch_size, hidden_dtype, **kwargs):
        ref_len = state["ref_len"]
        if ref_len:
            target_len = state["target_len"]
            if timestep.dim() <= 1 and target_len is not None:
                timestep = timestep.view(-1, 1).expand(batch_size, target_len).contiguous()
            if timestep.dim() >= 2:
                clean = torch.zeros(batch_size, ref_len, *timestep.shape[2:],
                                    device=timestep.device, dtype=timestep.dtype)
                timestep = torch.cat([timestep, clean], dim=1)
        return original["_prepare_timestep"](timestep, batch_size, hidden_dtype, **kwargs)

    def prepare_pe(pixel_coords, frame_rate, x_dtype):
        pe = original["_prepare_positional_embeddings"](pixel_coords, frame_rate, x_dtype)
        ref_len = state["ref_len"]
        if not ref_len or not seg_value:
            return pe
        try:
            # AV models return [(v_pe, cross_video), (a_pe, cross_audio)]; only the video one has the tokens.
            if isinstance(pe, list) and pe and isinstance(pe[0], (list, tuple)) and isinstance(pe[0][0], (list, tuple)):
                return [(rotate_overlap_freqs(pe[0][0], ref_len, seg_value), pe[0][1]), pe[1]]
            return rotate_overlap_freqs(pe, ref_len, seg_value)
        except Exception as exc:                    # noqa: BLE001
            _say("tagging the reference tokens' position", exc)
            return pe

    def process_output(x, embedded_timestep, keyframe_idxs, **kw):
        ref_len = state["ref_len"]
        if ref_len:
            try:
                from comfy.ldm.lightricks.av_model import CompressedTimestep
                if isinstance(x, (list, tuple)):
                    x = [x[0][:, :x[0].shape[1] - ref_len], *x[1:]]
                    parts = list(embedded_timestep) if isinstance(embedded_timestep, (list, tuple)) else [embedded_timestep]
                    video_et = parts[0]
                    if isinstance(video_et, CompressedTimestep):
                        per_frame = max(1, getattr(video_et, "patches_per_frame", 1) or 1)
                        cut = max(1, ref_len // per_frame)
                        shorter = copy.copy(video_et)
                        shorter.data = video_et.data[:, : video_et.num_frames - cut].contiguous()
                        shorter.num_frames = video_et.num_frames - cut
                        parts[0] = shorter
                    elif hasattr(video_et, "shape") and video_et.dim() >= 2 and video_et.shape[1] > 1:
                        parts[0] = video_et[:, : video_et.shape[1] - ref_len]
                    embedded_timestep = parts
                else:
                    x = x[:, :x.shape[1] - ref_len]
                    if hasattr(embedded_timestep, "shape") and embedded_timestep.dim() >= 2 and embedded_timestep.shape[1] > 1:
                        embedded_timestep = embedded_timestep[:, : embedded_timestep.shape[1] - ref_len]
            except Exception as exc:                # noqa: BLE001
                _say("cutting the reference tokens off", exc)
        return original["_process_output"](x, embedded_timestep, keyframe_idxs, **kw)

    for name, fn in (("_process_input", process_input), ("_prepare_timestep", prepare_timestep),
                     ("_prepare_positional_embeddings", prepare_pe), ("_process_output", process_output)):
        patcher.add_object_patch(_PREFIX + name, patching.tag(fn, key))
