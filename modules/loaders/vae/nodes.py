"""Loading a VAE, with an explicit dtype and a report of what happened.

Two differences from core's VAELoader, both carried over from v4 where they were
proven: the dtype is chosen rather than guessed, and a second output narrates
what actually loaded. Silent success that produced the wrong thing is the failure
mode these exist to remove.
"""

import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from ..._core import log

# Chosen, not inferred. bf16 is the default because fp16 is a known failure on
# the AV VAEs, and fp32 decodes slowly for no visible gain.
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def vae_files():
    # Empty is a legitimate state -- a fresh install has no models. Returning an
    # empty list lets the node exist and say so, rather than failing to register.
    return folder_paths.get_filename_list("vae")


class FunPackVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            # Frozen once shipped: this string is stored as `class_type` in every
            # saved workflow, so changing it silently breaks them.
            node_id="FunPackVAELoader",
            display_name="FunPack VAE Loader",
            category="FunPack/Loaders",
            description="Load a VAE with an explicit compute dtype.",
            inputs=[
                io.Combo.Input("vae_name", options=vae_files()),
                io.Combo.Input("dtype", options=list(DTYPES), default="bfloat16"),
            ],
            outputs=[
                io.Vae.Output(display_name="vae"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, vae_name: str, dtype: str) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        state_dict = comfy.utils.load_torch_file(path)
        vae = comfy.sd.VAE(sd=state_dict, dtype=DTYPES[dtype])
        vae.throw_exception_if_invalid()
        log.info("FunPack VAE Loader", f"{vae_name} loaded as {dtype}")
        return io.NodeOutput(vae, f"{vae_name} loaded as {dtype}")
