"""The diffusion model loader.

Picking a file and how to read it. Every option here changes what the loaded
model IS, which is why they are node widgets and not settings: nobody downstream
can supply them, and a saved workflow has to record them.
"""

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..._core import log
from .. import gguf_support, sla_attention
from ..common import (COMPUTE_DTYPES, WEIGHT_DTYPES, attention_choices,
                      attention_override, dtype_of, set_fp16_accumulation,
                      weight_model_options)


def _model_file_choices():
    """Diffusion model files, `.gguf` included.

    Core's extension set has no `.gguf`, so those files are on disk and
    invisible to every picker. Appended rather than merged in sorted order, so
    an existing pipeline's saved choice keeps its position in the list and
    nothing a user already picked moves.
    """
    return list(folder_paths.get_filename_list("diffusion_models")) + \
        gguf_support.gguf_names("diffusion_models")


class FunPackDiffusionModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackDiffusionModelLoader",
            display_name="FunPack Diffusion Model Loader",
            category="FunPack/Loaders",
            description="Load a diffusion model, choosing precision and attention backend.",
            inputs=[
                io.Combo.Input("model_name", options=_model_file_choices()),
                io.Combo.Input("weight_dtype", options=WEIGHT_DTYPES, default="default",
                               tooltip="How the weights are stored. 'default' keeps whatever "
                                       "the file already is."),
                io.Combo.Input("compute_dtype", options=COMPUTE_DTYPES, default="default",
                               tooltip="What the maths runs in, independently of storage."),
                io.Combo.Input("attention", options=attention_choices(), default="default",
                               tooltip="Attention backend. Only what this machine can run is "
                                       "listed; 'default' keeps ComfyUI's launch choice."),
                io.Boolean.Input("fp16_accumulation", default=False, optional=True,
                                 tooltip="Faster fp16 matmuls where the torch build supports it."),
                io.Boolean.Input("sla", default=False, optional=True,
                                 tooltip="Block-sparse attention for MiniMax H3 -- the "
                                         "inference path lightx2v's SLA turbo LoRA was "
                                         "distilled against, which is why that LoRA gives no "
                                         "speedup on its own. Roughly 3.7x the attention "
                                         "throughput at 768p/15s. Runs ALONGSIDE the attention "
                                         "backend above rather than replacing it: SLA takes "
                                         "H3's long packed self-attention, the chosen backend "
                                         "takes the text refiner, masked calls and any trailing "
                                         "dense steps. Skipped, with the reason, on anything "
                                         "that is not MiniMax H3 or on a machine without Triton."),
                # SLA settings. Every one is validated at its default (see sla_attention);
                # they are here because a knob whose value was measured is still a knob.
                io.Float.Input("sla_sparsity", default=sla_attention.SLA_DEFAULTS["sparsity_ratio"],
                              min=0.0, max=0.95, step=0.05, optional=True,
                              tooltip="Fraction of key blocks skipped, when attention is "
                                      "sla_h3. 0.90 is validated; 0.85 is lightx2v's own value "
                                      "and ~15% slower. Break-even is around 0.60 -- below that "
                                      "the kernel is SLOWER than dense, so a low value is a "
                                      "loss, not a safe fallback."),
                io.Combo.Input("sla_block_size",
                               options=["64", "128"],
                               default=str(sla_attention.SLA_DEFAULTS["block_size"]),
                               optional=True,
                               tooltip="How many sequence tokens share one key selection. H3 "
                                       "packs audio at 80 rows per second, so a 128-row block "
                                       "forces 1.6s of speech down one attention pattern while "
                                       "the same rows are 3% of a video frame. Use 128 only "
                                       "when the audio does not matter."),
                io.Boolean.Input("sla_protect_audio",
                                 default=sla_attention.SLA_DEFAULTS["protect_audio"], optional=True,
                                 tooltip="Always attend the [text | cond | audio] prefix, "
                                         "whatever top-k picks. Audio is ~1% of the packed "
                                         "sequence, so plain top-k regularly drops all of it "
                                         "and the soundtrack degrades while the video still "
                                         "looks fine. Costs about 7%."),
                io.Int.Input("sla_min_seq_len",
                            default=sla_attention.SLA_DEFAULTS["min_seq_len"],
                            min=0, max=1000000, step=1024, optional=True,
                            tooltip="Sequences shorter than this stay dense. Guards the short "
                                    "text refiner, which must never be sparsified, and "
                                    "low-resolution runs where block selection costs more "
                                    "than it saves."),
                io.Int.Input("sla_dense_last_steps",
                            default=sla_attention.SLA_DEFAULTS["dense_last_steps"],
                            min=0, max=8, optional=True,
                            tooltip="Run the last N sampling steps at full attention. 0 "
                                    "matches lightx2v; 1 was tested and did not help, for "
                                    "+20% time."),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str, compute_dtype: str,
                attention: str, fp16_accumulation: bool = False, sla: bool = False,
                sla_sparsity: float = None, sla_block_size: str = None,
                sla_protect_audio: bool = None, sla_min_seq_len: int = None,
                sla_dense_last_steps: int = None) -> io.NodeOutput:
        notes = [f"FunPack Diffusion Model Loader | {model_name}"]

        accumulation = set_fp16_accumulation(fp16_accumulation)
        if fp16_accumulation and accumulation is None:
            notes.append("fp16_accumulation: unsupported by this torch build, ignored")
        elif accumulation is not None:
            notes.append(f"fp16_accumulation: {accumulation}")

        model_options = weight_model_options(weight_dtype)
        notes.append(f"weight dtype: {weight_dtype}")

        if gguf_support.is_gguf(model_name):
            path = gguf_support.gguf_path("diffusion_models", model_name)
            if not path:
                raise RuntimeError(f"{model_name} is no longer where it was listed from.")
            misnamed = False
        else:
            path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
            # The CONTENT decides, not the extension. A .gguf renamed to
            # .safetensors used to reach the safetensors parser and fail with
            # a UTF-8 decode error from reading a binary header as JSON --
            # true, and no help at all.
            misnamed = gguf_support.has_gguf_magic(path)
        if gguf_support.is_gguf(model_name) or misnamed:
            if misnamed:
                notes.append(f"{model_name} is named .safetensors but is a GGUF "
                             f"container -- loaded as GGUF")
            state_dict, gguf_options, gguf_note = gguf_support.load_state_dict(path)
            # The quantized path needs its own torch operations, and they must
            # not be lost to the dtype options above -- a GGUF loaded with
            # stock ops would try to matmul block-quantized storage.
            model_options = {**model_options, **gguf_options}
            metadata = None
            notes.append(gguf_note)
        else:
            state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        model = comfy.sd.load_diffusion_model_state_dict(
            state_dict, model_options=model_options, metadata=metadata)
        if model is None:
            # comfy returns None rather than raising, and a None reaching the
            # sampler fails much further away than the mistake.
            raise RuntimeError(
                f"Could not detect a diffusion model in {model_name}. Loading a text "
                f"encoder or a VAE file here is the usual cause.")

        dtype = dtype_of(compute_dtype)
        if dtype is not None and hasattr(model, "set_model_compute_dtype"):
            # Do NOT clear force_cast_weights afterwards. set_model_compute_dtype
            # sets it deliberately, and it is what casts each layer's WEIGHTS to
            # match. Without it the model still casts its INPUT to the requested
            # dtype (_apply_model reads manual_cast_dtype), so the first Linear
            # gets a bf16 activation against fp32 weights and sampling dies with
            # "mat1 and mat2 must have the same dtype". v4 cleared it and shipped
            # that way; it survives on constrained hardware only because the
            # low-VRAM path forces the cast back on regardless.
            model.set_model_compute_dtype(dtype)
            notes.append(f"compute dtype: {compute_dtype}")
        elif dtype is not None:
            notes.append("compute dtype: unsupported by this ComfyUI, ignored")

        override = attention_override(attention)

        # SLA wraps the chosen backend rather than replacing it: there is one
        # override slot, so the backend goes in as SLA's dense fall-through.
        # When SLA does not take (not H3, no Triton, switched off) the backend
        # is installed on its own -- asking for sparse attention must never
        # cost the backend that was picked.
        installed = False
        if sla:
            model, sla_note, installed = sla_attention.install_sla(
                model,
                sparsity_ratio=sla_sparsity, block_size=sla_block_size,
                min_seq_len=sla_min_seq_len, dense_last_steps=sla_dense_last_steps,
                protect_audio=sla_protect_audio,
                dense_fn=override, dense_label=attention)
            notes.append(sla_note)
        if not installed:
            if override is not None:
                model.model_options.setdefault("transformer_options", {})[
                    "optimized_attention_override"] = override
                notes.append(f"attention: {attention}")
            else:
                notes.append("attention: default (as launched)")
        else:
            notes.append(f"attention: {attention} (dense calls)")

        # Never reach into the model's shape to describe it: a log line that
        # assumes structure can fail the load it was only meant to narrate.
        kind = type(getattr(model, "model", model)).__name__
        applied_attention = "sla_h3" if installed else (
            attention if override is not None else "default (as launched)")
        log.info("FunPack Diffusion Model Loader",
                 f"{model_name} loaded as {kind}, weights "
                 f"{weight_dtype}, compute {compute_dtype}, attention "
                 f"{applied_attention}")
        return io.NodeOutput(model, "\n".join(notes))
