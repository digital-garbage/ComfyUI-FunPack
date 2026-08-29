"""The diffusion model loader.

Picking a file and how to read it. Every option here changes what the loaded
model IS, which is why they are node widgets and not settings: nobody downstream
can supply them, and a saved workflow has to record them.
"""

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io

from ..common import (COMPUTE_DTYPES, WEIGHT_DTYPES, attention_choices,
                      attention_override, dtype_of, set_fp16_accumulation,
                      weight_model_options)


class FunPackDiffusionModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackDiffusionModelLoader",
            display_name="FunPack Diffusion Model Loader",
            category="FunPack/Loaders",
            description="Load a diffusion model, choosing precision and attention backend.",
            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("diffusion_models")),
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
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, weight_dtype: str, compute_dtype: str,
                attention: str, fp16_accumulation: bool = False) -> io.NodeOutput:
        notes = [f"FunPack Diffusion Model Loader | {model_name}"]

        accumulation = set_fp16_accumulation(fp16_accumulation)
        if fp16_accumulation and accumulation is None:
            notes.append("fp16_accumulation: unsupported by this torch build, ignored")
        elif accumulation is not None:
            notes.append(f"fp16_accumulation: {accumulation}")

        model_options = weight_model_options(weight_dtype)
        notes.append(f"weight dtype: {weight_dtype}")

        path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
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
        if override is not None:
            model.model_options.setdefault("transformer_options", {})[
                "optimized_attention_override"] = override
            notes.append(f"attention: {attention}")
        else:
            notes.append("attention: default (as launched)")

        return io.NodeOutput(model, "\n".join(notes))
