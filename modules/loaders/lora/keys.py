"""Which of a LoRA's weights land in this model, and which cannot.

ComfyUI converts the naming dialects (lora_up / lora_A / PEFT); it does not unwrap a file whose
keys sit under a training wrapper. v4 measured it: an H3 adapter keyed `base_model.model.dit.*`
matched 0 of 532 keys and did nothing, silently. Every known wrapper is tried and the one matching
the most weights wins. A pair that matches by name but not by shape is dropped: ComfyUI reshapes
any delta with the right element count, so a wrong-shaped one merges scrambled (all-NaN mid-render),
and a full-width LoRA on a curve-form H3 checkpoint churned ~25 GB failing per key.
"""

import comfy.lora
import comfy.lora_convert

# (prefix, replacement). Replacing, not only stripping: ComfyUI's key map has `diffusion_model.<path>`
# and no bare form, so `base_model.model.dit.blocks.0...` must become `diffusion_model.blocks.0...`.
REKEY = (
    ("transformer.", ""), ("diffusion_model.", ""), ("model.diffusion_model.", ""),
    ("base_model.model.", ""), ("base_model.model.dit.", ""), ("base_model.model.diffusion_model.", ""),
    ("lora_model.", ""), ("net.", ""),
    ("base_model.model.dit.", "diffusion_model."), ("base_model.model.", "diffusion_model."),
    ("dit.", "diffusion_model."), ("model.dit.", "diffusion_model."), ("transformer.", "diffusion_model."),
)


def _rekey(lora, prefix, replacement):
    if not any(k.startswith(prefix) for k in lora):
        return None
    return {(replacement + k[len(prefix):] if k.startswith(prefix) else k): v for k, v in lora.items()}


def _misfit(sd, key, adapter):
    """True when a plain LoRA pair cannot make the weight it names (both dimensions, not the count)."""
    w, target = getattr(adapter, "weights", None), sd.get(key)
    if not w or len(w) < 2 or target is None or target.dim() < 2:
        return False
    if (len(w) > 3 and w[3] is not None) or (len(w) > 5 and w[5] is not None):   # locon mid / padded target: not modelled
        return False
    try:
        got = (int(w[0].flatten(start_dim=1).shape[0]), int(w[1].flatten(start_dim=1).shape[-1]))
    except Exception:  # noqa: BLE001 -- not a plain pair (LoKr, LoHa, diff): left to ComfyUI
        return False
    return got != (int(target.shape[0]), int(target.shape[1:].numel()))


def match(model, clip, lora):
    """-> (patches, how, dropped): the best keying's patches, which keying won, and how many were dropped for shape."""
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    if clip is not None:
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    converted = comfy.lora_convert.convert_lora(lora)
    best, how = comfy.lora.load_lora(converted, key_map, log_missing=False), "as-is"
    for prefix, replacement in REKEY:
        variant = _rekey(converted, prefix, replacement)
        if variant is not None:
            got = comfy.lora.load_lora(variant, key_map, log_missing=False)
            if len(got) > len(best):
                best, how = got, f"{prefix} -> {replacement or '(none)'}"
    sd = model.model.state_dict()
    bad = [k for k, a in best.items() if _misfit(sd, k, a)]
    for k in bad:
        best.pop(k)
    return best, how, len(bad)
