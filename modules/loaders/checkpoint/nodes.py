"""One file holding a model, its text encoder and its VAE.

The split loaders are right for the families FunPack targets -- LTX and H3 ship
the diffusion model, the encoders and the VAEs separately, and pairing them is
the user's choice. The SD family does not: one checkpoint carries all three, and
without this there is no way to load one at all.

Both exist, and neither is the "real" one. A pipeline using three slots can be
replaced by one using this, which is the point of slots being replaceable.
"""

import comfy.sd
import folder_paths
from comfy_api.latest import io

from ..._core import log


class FunPackCheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackCheckpointLoader",
            display_name="FunPack Checkpoint Loader",
            category="FunPack/Loaders",
            description="Load a model, text encoder and VAE from one checkpoint file.",
            inputs=[
                io.Combo.Input("ckpt_name", options=folder_paths.get_filename_list("checkpoints")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Vae.Output(display_name="vae"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, ckpt_name: str) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae, *_ = comfy.sd.load_checkpoint_guess_config(
            path, output_vae=True, output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"))

        # Each part is reported, because a checkpoint missing its VAE or its
        # encoder loads happily and then fails somewhere else entirely.
        missing = [name for name, part in (("clip", clip), ("vae", vae)) if part is None]
        if missing:
            log.warning("FunPack Checkpoint Loader",
                        f"{ckpt_name} has no {' or '.join(missing)}; those outputs are empty "
                        f"and whatever reads them will fail")
        kind = type(getattr(model, "model", model)).__name__
        log.info("FunPack Checkpoint Loader", f"{ckpt_name} loaded as {kind}")

        return io.NodeOutput(model, clip, vae,
                             f"{ckpt_name} loaded as {kind}"
                             + (f" (no {', '.join(missing)})" if missing else ""))
