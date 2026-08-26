"""FunPack's own model loaders.

The point is not to replace ComfyUI's loaders — it is to make the FunPack pipeline
configurable by picking files and nothing else. One loader per model kind, every option
that actually matters on the node (quantization, compute dtype, attention backend), and
list inputs where a family needs more than one file.

Attention backends come from ComfyUI's own registry rather than a private sageattention
import, so this node offers exactly what the machine can run — sage/sage3 when
SageAttention is installed, flash when flash-attn is, xformers when enabled — and never
lies about the rest.
"""
import logging

import comfy.sd
import comfy.utils
import folder_paths
import torch

try:
    from .widgets import field, list_widget, parse_rows
    from . import gguf_support, sla_attention
except ImportError:  # standalone tests import the modules directly
    from widgets import field, list_widget, parse_rows
    import gguf_support
    import sla_attention

def model_file_choices():
    """Diffusion model files, `.gguf` included.

    Core's extension set has no `.gguf`, so those files are on disk and invisible to every
    picker. Appended rather than merged in sorted order, so an existing pipeline's saved
    choice keeps its position in the list and nothing a user already picked moves.
    """
    return list(folder_paths.get_filename_list("diffusion_models")) + \
        gguf_support.gguf_names("diffusion_models")


def encoder_file_choices():
    """Text encoder files, `.gguf` included. Same reasoning as model_file_choices()."""
    return list(folder_paths.get_filename_list("text_encoders")) + \
        gguf_support.gguf_names("text_encoders")


WEIGHT_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16", "fp32"]
COMPUTE_DTYPES = ["default", "fp16", "bf16", "fp32"]
VAE_DTYPES = ["default", "fp16", "bf16", "fp32"]

_DTYPE_BY_NAME = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2": getattr(torch, "float8_e5m2", None),
    # fp8_e4m3fn_fast is fp8_e4m3fn plus comfy's fp8 matmul path; see weight_model_options.
    "fp8_e4m3fn_fast": getattr(torch, "float8_e4m3fn", None),
}


def dtype_of(name):
    """The torch dtype a dtype choice names, or None for 'default'/unsupported builds."""
    return _DTYPE_BY_NAME.get(str(name or "default"))


def weight_model_options(weight_dtype):
    """comfy `model_options` for a weight dtype choice. Same mapping core's UNETLoader uses."""
    options = {}
    dtype = dtype_of(weight_dtype)
    if dtype is not None:
        options["dtype"] = dtype
    if weight_dtype == "fp8_e4m3fn_fast":
        options["fp8_optimizations"] = True
    return options


def raven_lora_choices():
    """LoRA files, with "None" first — the causal lane is opt-in per model."""
    return ["None"] + list(folder_paths.get_filename_list("loras"))


def attention_choices():
    """Attention backends this ComfyUI can actually run, newest-registry first.

    'default' leaves the model on whatever ComfyUI itself selected (the --use-sage-attention
    / --use-flash-attention launch flags), which is why it stays the default here.

    SLA is deliberately NOT in this list: it is not a rival backend but a layer above one.
    It handles only H3's long packed self-attention and hands everything else — the text
    refiner, masked calls, the trailing dense steps — to whichever backend is chosen here,
    so the two compose instead of displacing each other. Its switch is `sla`.
    """
    names = []
    try:
        from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS
        names = sorted(REGISTERED_ATTENTION_FUNCTIONS.keys())
    except Exception:  # noqa: BLE001 - older ComfyUI without the registry
        pass
    return ["default"] + names


def attention_override(name):
    """A transformer_options override that routes every attention call to `name`.

    ComfyUI wraps each attention implementation with `wrap_attn`, which hands the override
    the original function plus the call's arguments. Calling the wrapped replacement would
    re-enter that machinery, so the unwrapped function is what gets called.
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
    """torch's fp16 accumulation switch. Returns what it ended up as, or None if unsupported."""
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return None
    matmul.allow_fp16_accumulation = bool(enabled)
    return bool(enabled)


def _load_with_raven_lora(model_name, lora_name, weight_dtype, model_options):
    """Load an H3 checkpoint as a chunk-causal DiT with the RAVEN LoRA attached.

    Delegated to the RAVEN package rather than reimplemented, for a reason that is not
    convenience: the adapter is an FP32 activation residual registered as parameters of the
    base Linear leaves, and it has to be attached to the raw model BEFORE the ModelPatcher is
    built. ``ModelPatcher.model_size()`` memoises, so an adapter attached afterwards is
    invisible to ComfyUI's memory ledger and never moved by partial CPU offload — the model
    would look smaller than it is and the residual would sit on the wrong device.

    Returns (model, note). Raises with an actionable message rather than falling back to a
    plain load: silently returning a model WITHOUT the adapter would leave the causal lane
    reading an attention pattern nothing was trained for, and look like a quality problem.
    """
    try:
        from .raven_causal import locate_raven
    except ImportError:
        from raven_causal import locate_raven
    module, reason = locate_raven()
    if module is None:
        raise RuntimeError(
            f"ERROR: a RAVEN LoRA is selected ({lora_name}) but the causal model class is "
            f"unavailable — {reason} Set raven_lora to None to load this model normally.")
    from raven_streaming import loader as raven_loader
    from raven_streaming.causal_model import RavenCausalMiniMaxH3Model

    model = raven_loader.load_raven_diffusion_model(
        model_name, lora_name,
        weight_dtype=weight_dtype,
        model_options=dict(model_options),
        unet_model_cls=RavenCausalMiniMaxH3Model,
    )
    return model, (f"RAVEN: chunk-causal DiT + {lora_name} at strength 1.0 (a LoRA on the "
                   f"ordinary H3 checkpoint, not a separate model). The Chain Sampler's "
                   f"'Remembering chunks' mode can now run; without that mode on, this model "
                   f"samples exactly as usual.")


class FunPackDiffusionModelLoader:
    """Loads a diffusion model with quantization and an attention backend chosen per node."""

    CATEGORY = "FunPack/Loaders"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "status")
    FUNCTION = "load_model"
    DESCRIPTION = ("Diffusion model loader with weight/compute dtype and a per-model attention "
                   "backend (SageAttention, FlashAttention, xformers — whichever are installed).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (model_file_choices(), {
                    "tooltip": "The diffusion model file, from ComfyUI/models/diffusion_models. "
                               ".gguf files are listed too (see the status output for which "
                               "GGUF backend loaded it)."}),
                "weight_dtype": (WEIGHT_DTYPES, {
                    "default": "default",
                    "tooltip": "How weights are stored in VRAM. fp8_e4m3fn roughly halves the "
                               "model's memory; _fast additionally runs fp8 matmuls, which is "
                               "faster on Ada and newer and slightly less precise. 'default' "
                               "keeps whatever the file already is."}),
                "compute_dtype": (COMPUTE_DTYPES, {
                    "default": "default",
                    "tooltip": "The dtype the model computes in. Setting this stops per-weight "
                               "casting, so an fp8 model computes in bf16 without re-casting "
                               "every layer."}),
                "attention": (attention_choices(), {
                    "default": "default",
                    "tooltip": "Attention backend for THIS model. 'default' uses whatever "
                               "ComfyUI was launched with. Only backends installed on this "
                               "machine are listed. Composes with `sla`: sparse attention "
                               "handles H3's long packed sequence, this backend handles "
                               "everything else."}),
            },
            "optional": {
                "raven_lora": (raven_lora_choices(), {
                    "default": "None",
                    "tooltip": "MiniMax H3 only. Loads this model as a CHUNK-CAUSAL DiT with a "
                               "RAVEN streaming LoRA attached, which is what the Chain "
                               "Sampler's 'Remembering chunks' mode needs. The clip is then "
                               "generated in time chunks that each remember the ones before "
                               "them, so a long clip stays one continuous shot instead of "
                               "separate scenes stitched together. The LoRA is not optional "
                               "for that mode: the chunked attention pattern is what it was "
                               "trained to read, and the base H3 weights have never seen a "
                               "key/value cache. Needs the ComfyUI-MiniMax-H3-RAVEN-Streaming "
                               "pack installed for the causal model class. This is a LoRA on "
                               "the ordinary H3 checkpoint — there is no separate RAVEN model "
                               "file. Strength is fixed at 1.0: the adapter is what teaches "
                               "the chunked attention pattern, so 'partly on' is not a useful "
                               "state. 'None' loads normally. "
                               "Use the full non-pruned checkpoint: the pruned/adaln-curve cut "
                               "has no time_embedder for the adapter to attach to."}),
                "fp16_accumulation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Accumulate fp16 matmuls in fp16. Faster on recent NVIDIA cards; "
                               "needs a torch build that supports it, ignored otherwise."}),
                "sla": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Block-sparse attention for MiniMax H3 — the inference path "
                               "lightx2v's SLA turbo LoRA was distilled against, which is why "
                               "that LoRA gives no speedup on its own. Roughly 3.7x the "
                               "attention throughput at 768p/15s. It runs ALONGSIDE the "
                               "attention backend above rather than replacing it: SLA takes "
                               "H3's long packed self-attention, the chosen backend (sage3, "
                               "int8, flash…) takes the text refiner, masked calls and any "
                               "trailing dense steps. Skipped, with the reason, on anything "
                               "that is not MiniMax H3 or on a machine without Triton."}),
                # SLA settings. Every one is validated at its default (see sla_attention);
                # they are here because a knob whose value was measured is still a knob.
                "sla_sparsity": ("FLOAT", {
                    "default": sla_attention.SLA_DEFAULTS["sparsity_ratio"],
                    "min": 0.0, "max": 0.95, "step": 0.05,
                    "advanced": True,
                    "tooltip": "Fraction of key blocks skipped, when attention is sla_h3. 0.90 "
                               "is validated; 0.85 is lightx2v's own value and ~15% slower. "
                               "Break-even is around 0.60 — below that the kernel is SLOWER "
                               "than dense, so a low value is a loss, not a safe fallback. "
                               "Speech artefacts on H3 come from 4-step distillation, not from "
                               "sparsity: use 6 steps rather than lowering this."}),
                "sla_block_size": (["64", "128"], {
                    "default": str(sla_attention.SLA_DEFAULTS["block_size"]),
                    "advanced": True,
                    "tooltip": "How many sequence tokens share one key selection. H3 packs "
                               "audio at 80 rows per second, so a 128-row block forces 1.6s of "
                               "speech down one attention pattern while the same rows are 3% of "
                               "a video frame. Total attention work is identical either way — "
                               "only the routing granularity changes. Use 128 only when the "
                               "audio does not matter."}),
                "sla_protect_audio": ("BOOLEAN", {
                    "default": sla_attention.SLA_DEFAULTS["protect_audio"],
                    "advanced": True,
                    "tooltip": "Always attend the [text | cond | audio] prefix, whatever top-k "
                               "picks. Audio is ~1% of the packed sequence, so plain top-k "
                               "regularly drops all of it and the soundtrack degrades while the "
                               "video still looks fine. Costs about 7%."}),
                "sla_min_seq_len": ("INT", {
                    "default": sla_attention.SLA_DEFAULTS["min_seq_len"],
                    "min": 0, "max": 1000000, "step": 1024,
                    "advanced": True,
                    "tooltip": "Sequences shorter than this stay dense. Guards the short text "
                               "refiner, which must never be sparsified, and low-resolution "
                               "runs where block selection costs more than it saves."}),
                "sla_dense_last_steps": ("INT", {
                    "default": sla_attention.SLA_DEFAULTS["dense_last_steps"],
                    "min": 0, "max": 8,
                    "advanced": True,
                    "tooltip": "Run the last N sampling steps at full attention. 0 matches "
                               "lightx2v; 1 was tested and did not help, for +20% time."}),
            },
        }

    def load_model(self, model_name, weight_dtype, compute_dtype, attention,
                   fp16_accumulation=False, sla_sparsity=None, sla_block_size=None,
                   sla_protect_audio=None, sla_min_seq_len=None, sla_dense_last_steps=None,
                   sla=False, raven_lora="None"):
        notes = [f"FunPack Diffusion Model Loader | {model_name}"]

        accum = set_fp16_accumulation(fp16_accumulation)
        if fp16_accumulation and accum is None:
            notes.append("fp16_accumulation: unsupported by this torch build, ignored")
        elif accum is not None:
            notes.append(f"fp16_accumulation: {accum}")

        model_options = weight_model_options(weight_dtype)
        notes.append(f"weight dtype: {weight_dtype}")

        path = None
        misnamed = False
        if gguf_support.is_gguf(model_name):
            path = gguf_support.gguf_path("diffusion_models", model_name)
            if not path:
                raise RuntimeError(f"ERROR: {model_name} is no longer where it was listed from.")
        else:
            path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            # The CONTENT decides, not the extension. A .gguf renamed to .safetensors used to
            # reach the safetensors parser and fail with a UTF-8 decode error from reading a
            # binary header as JSON — true, and no help at all.
            misnamed = gguf_support.has_gguf_magic(path)
        if path is not None and (gguf_support.is_gguf(model_name) or misnamed):
            if misnamed:
                notes.append(f"{model_name} is named .safetensors but is a GGUF container — "
                             f"loaded as GGUF")
            state_dict, gguf_options, gguf_note = gguf_support.load_state_dict(path)
            # The quantized path needs its own torch operations, and they must not be lost to
            # the dtype options merged above — a GGUF loaded with stock ops would try to matmul
            # block-quantized storage.
            model_options = {**model_options, **gguf_options}
            metadata = None
            notes.append(gguf_note)
        else:
            state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        if str(raven_lora) not in ("", "None"):
            model, raven_note = _load_with_raven_lora(
                model_name, raven_lora, weight_dtype, model_options)
            notes.append(raven_note)
        else:
            model = comfy.sd.load_diffusion_model_state_dict(
                state_dict, model_options=model_options, metadata=metadata)
        if model is None:
            raise RuntimeError(
                f"ERROR: could not detect a diffusion model in {model_name}. "
                "Loading a text encoder or VAE file here is the usual cause.")

        dtype = dtype_of(compute_dtype)
        if dtype is not None and hasattr(model, "set_model_compute_dtype"):
            model.set_model_compute_dtype(dtype)
            model.force_cast_weights = False
            notes.append(f"compute dtype: {compute_dtype}")
        elif dtype is not None:
            notes.append("compute dtype: unsupported by this ComfyUI, ignored")

        override = attention_override(attention)

        # SLA wraps the chosen backend rather than replacing it: there is one override
        # slot, so the backend goes in as SLA's dense fall-through. When SLA does not
        # take (not H3, no Triton, switched off) the backend is installed on its own —
        # asking for sparse attention must never cost the backend you picked.
        installed = False
        if sla:
            model, note, installed = sla_attention.install_sla(
                model,
                sparsity_ratio=sla_sparsity, block_size=sla_block_size,
                min_seq_len=sla_min_seq_len, dense_last_steps=sla_dense_last_steps,
                protect_audio=sla_protect_audio,
                dense_fn=override, dense_label=attention)
            notes.append(note)
        if not installed:
            if override is not None:
                model.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = override
                notes.append(f"attention: {attention}")
            else:
                notes.append("attention: default (as launched)")
        else:
            notes.append(f"attention: {attention} (dense calls)")

        return (model, "\n".join(notes))


class FunPackCLIPLoader:
    """Loads one or more text encoder files into a single CLIP."""

    CATEGORY = "FunPack/Loaders"
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "status")
    FUNCTION = "load_clip"
    DESCRIPTION = ("Text encoder loader. Add one slot for a single-file encoder (LTX-2.5's "
                   "Gemma4), or two for an encoder plus its connector (LTX-2.3's Gemma3).")

    @classmethod
    def clip_types(cls):
        try:
            return [t.name.lower() for t in comfy.sd.CLIPType]
        except Exception:  # noqa: BLE001
            return ["ltxv"]

    @classmethod
    def INPUT_TYPES(cls):
        encoders = encoder_file_choices()
        types = cls.clip_types()
        return {
            "required": {
                "clip_list": list_widget(
                    "text encoder",
                    [field("clip_name", "combo", label="file", choices=encoders)],
                    add_label="+ Add slot",
                    tooltip="Text encoder files, in load order. One slot for LTX-2.5 (Gemma4); "
                            "two for LTX-2.3 (Gemma3 + its connector). .gguf files are listed "
                            "too, and may be mixed with .safetensors ones."),
                "type": (types, {
                    "default": "ltxv" if "ltxv" in types else (types[0] if types else "ltxv"),
                    "tooltip": "Which model family the encoder is for. LTX-2 (all point "
                               "releases) is 'ltxv'; MiniMax H3 is 'minimax'."}),
            },
            "optional": {
                "device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "'cpu' keeps the encoder off the GPU — slower prompts, more VRAM "
                               "left for the diffusion model."}),
            },
        }

    def load_clip(self, clip_list, type, device="default"):
        rows = parse_rows(clip_list, [field("clip_name", "combo")], key="clip_name")
        names = [row["clip_name"] for row in rows]
        if not names:
            raise RuntimeError("FunPack CLIP Loader: add at least one text encoder file.")

        clip_type = getattr(comfy.sd.CLIPType, str(type).upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        gguf_notes = []
        # Resolved once: a slot counts as GGUF by extension OR by container magic, so a
        # renamed file behaves the same here as in the diffusion loader.
        def _gguf_slot(n):
            if gguf_support.is_gguf(n):
                return gguf_support.gguf_path("text_encoders", n)
            try:
                p = folder_paths.get_full_path_or_raise("text_encoders", n)
            except Exception:  # noqa: BLE001 — the normal path reports a missing file
                return None
            return p if gguf_support.has_gguf_magic(p) else None

        gguf_slots = {n: _gguf_slot(n) for n in names}
        if any(gguf_slots.values()):
            # A .gguf encoder cannot go through load_clip(), which reads files itself. Every
            # slot becomes a state dict instead, so a GGUF and a .safetensors connector can
            # sit in the same list — which is the normal LTX-2.3 shape.
            state_dicts = []
            for n in names:
                gpath = gguf_slots.get(n)
                if gpath:
                    if not gguf_support.is_gguf(n):
                        gguf_notes.append(f"{n} is named .safetensors but is a GGUF container")
                    sd, gopts, gnote = gguf_support.load_clip_state_dict(gpath)
                    state_dicts.append(sd)
                    model_options = {**model_options, **gopts}
                    gguf_notes.append(f"{n}: {gnote}")
                else:
                    state_dicts.append(comfy.utils.load_torch_file(
                        folder_paths.get_full_path_or_raise("text_encoders", n)))
            clip = comfy.sd.load_text_encoder_state_dicts(
                state_dicts,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )
        else:
            paths = [folder_paths.get_full_path_or_raise("text_encoders", n) for n in names]
            clip = comfy.sd.load_clip(
                ckpt_paths=paths,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )
        status = "FunPack CLIP Loader | type={} device={}\n{}".format(
            type, device, "\n".join(f"  {i + 1}. {n}" for i, n in enumerate(names)))
        for gn in gguf_notes:
            status += f"\n  {gn}"
        return (clip, status)


class FunPackVAELoader:
    """Loads one VAE. Add a second instance for the audio VAE."""

    CATEGORY = "FunPack/Loaders"
    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("VAE", "status")
    FUNCTION = "load_vae"
    DESCRIPTION = ("VAE loader with an explicit dtype. LTX-2 and MiniMax H3 both ship a video "
                   "VAE and an audio VAE — use one of these per VAE.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "The VAE file, from ComfyUI/models/vae."}),
                "dtype": (VAE_DTYPES, {
                    "default": "default",
                    "tooltip": "MiniMax H3's VAEs must run in bf16 — fp16 produces garbage and "
                               "fp32 decodes slowly. 'default' lets ComfyUI choose."}),
            },
        }

    def load_vae(self, vae_name, dtype="default"):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=state_dict, dtype=dtype_of(dtype), metadata=metadata)
        check = getattr(vae, "throw_exception_if_invalid", None)
        if callable(check):
            check()
        return (vae, f"FunPack VAE Loader | {vae_name} dtype={dtype}")
