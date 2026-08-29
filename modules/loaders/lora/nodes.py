"""The LoRA loader.

One LoRA per node, chained for more. v4 carried N LoRAs in a single node behind a
JSON list widget and a `FlexibleOptionalInputType` -- a dict subclass that lied to
ComfyUI's validator so arbitrary keys would pass. Chaining needs neither: it is
what ComfyUI's own LoraLoader does, every strength is a real socket, and the graph
shows how many LoRAs are actually applied instead of hiding the count in a string.
"""

import comfy.sd
import comfy.utils
import folder_paths
from comfy_api.latest import io


class FunPackLoraLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackLoraLoader",
            display_name="FunPack LoRA Loader",
            category="FunPack/Loaders",
            description="Apply one LoRA to a model, and optionally its text encoder. "
                        "Chain these for more than one.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Clip.Input("clip", optional=True,
                              tooltip="Leave unwired for a model-only LoRA."),
                io.Float.Input("strength_clip", default=1.0, min=-100.0, max=100.0, step=0.01,
                               optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name: str, strength_model: float,
                clip=None, strength_clip: float = 1.0) -> io.NodeOutput:
        # Doing nothing is a valid request, but it must return the SAME objects
        # rather than clones: a clone here would silently drop patches a later
        # loader in the chain applied to the original.
        if strength_model == 0 and (clip is None or strength_clip == 0):
            return io.NodeOutput(model, clip, f"{lora_name}: both strengths are 0, not applied")

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(path, safe_load=True)
        patched_model, patched_clip = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip)

        applied = [f"model={strength_model}"]
        applied.append(f"clip={strength_clip}" if clip is not None else "clip=not wired")
        return io.NodeOutput(patched_model, patched_clip,
                             f"FunPack LoRA Loader | {lora_name} ({', '.join(applied)})")
