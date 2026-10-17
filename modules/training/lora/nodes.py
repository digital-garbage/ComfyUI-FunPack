"""The training node: one node that trains a LoRA from a saved training set on the models the pipeline loaded."""

from comfy_api.latest import io

from ..._core import addons
from . import train


class FunPackTrainLora(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackTrainLora",
            display_name="FunPack Train LoRA",
            category="FunPack/Training",
            description="Train a LoRA from a saved training set, on ComfyUI's own trainer. Saves it to the loras folder.",
            is_output_node=True,
            inputs=[
                io.Model.Input("model", optional=True, tooltip="A model already loaded to train on. Leave empty and give model_name to have the trainer load the file itself, once."),
                io.Clip.Input("clip", tooltip="The text encoder that turns the captions into conditioning."),
                io.Vae.Input("vae", tooltip="The VAE that turns pictures and clips into latents."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Only for models that learn sound with the picture (H3): turns the clips' sound into latents."),
                io.String.Input("set_name", default="", tooltip="The training set to learn from."),
                io.String.Input("name", default="funpack_lora", tooltip="File name of the finished LoRA (no extension). An earlier file is never overwritten."),
                io.Combo.Input("use", options=list(train.USES), default="normal",
                               tooltip="normal: liked results and your own media. avoid: only the disliked results, for a LoRA you apply at negative strength. all: everything."),
                io.Int.Input("steps", default=500, min=1, max=100000, tooltip="How many training steps."),
                io.Float.Input("learning_rate", default=0.0005, min=0.0000001, max=1.0, step=0.0000001, tooltip="How big each learning step is. Lower is safer and slower."),
                io.Int.Input("rank", default=16, min=1, max=128, tooltip="How much the LoRA can learn. Higher holds more detail and makes a bigger file."),
                io.Combo.Input("algorithm", options=list(train.ALGORITHMS), default="LoRA", tooltip="The adapter kind. LoRA is the common one."),
                io.Combo.Input("optimizer", options=["AdamW", "Adam", "SGD", "RMSprop"], default="AdamW", tooltip="How the weights are updated."),
                io.Combo.Input("loss", options=["MSE", "L1", "Huber", "SmoothL1"], default="MSE", tooltip="How the error is measured."),
                io.Int.Input("batch_size", default=1, min=1, max=64, tooltip="Samples per step. Models that train one sample at a time (H3) lower it to 1."),
                io.Int.Input("grad_accumulation", default=1, min=1, max=256, tooltip="Steps added up before the weights change: a bigger batch for less memory."),
                io.Int.Input("resolution", default=512, min=64, max=2048, step=32, tooltip="Training size in pixels per side. Each item is fitted to about this area, keeping its shape."),
                io.Boolean.Input("aspect_buckets", default=False, tooltip="Keep each picture's own shape (a few size groups) instead of one shape for all. Not used for H3."),
                io.Int.Input("frames", default=33, min=1, max=1025, tooltip="Frames taken from each clip, for video models. H3 rounds to its own grid (17 x n + 5)."),
                io.Float.Input("fps", default=0.0, min=0.0, max=120.0, step=0.5, tooltip="Frames per second to read the clips at. 0 uses each clip's own rate (H3: 24)."),
                io.Combo.Input("start", options=["start", "middle"], default="start", tooltip="Where in a clip to take the frames from."),
                io.String.Input("trigger", default="", tooltip="The prefix: text put at the start of every caption, for example a word to call the LoRA up later."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, tooltip="Seed for the noise and the LoRA's starting weights."),
                io.Combo.Input("training_dtype", options=["bf16", "fp32", "none"], default="bf16", tooltip="Precision used while training. bf16 is right for nearly everything."),
                io.Combo.Input("lora_dtype", options=["bf16", "fp32"], default="bf16", tooltip="Precision the LoRA is stored in."),
                io.Boolean.Input("gradient_checkpointing", default=True, tooltip="Trades speed for a lot less memory."),
                io.Boolean.Input("offloading", default=False, tooltip="Keeps model weights in system memory while training: slower, for cards that cannot hold the model."),
                io.Boolean.Input("bypass", default=False, tooltip="Apply the LoRA by hooks instead of changing weights. For quantized models."),
                io.String.Input("model_name", default="", tooltip="The diffusion model file to train on, as the loader lists it. The trainer loads it itself, once, with no loader node in front."),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"], default="default", tooltip="How the model file's weights are stored when it is loaded here."),
                io.Combo.Input("compute_dtype", options=["default", "fp16", "bf16", "fp32"], default="default", tooltip="What the maths runs in when the model is loaded here."),
                io.String.Input("postfix", default="", tooltip="Text put at the end of every caption."),
            ],
            outputs=[io.String.Output(display_name="lora")],
        )

    @classmethod
    def execute(cls, clip, vae, set_name, name, use, steps, learning_rate, rank, algorithm, optimizer, loss, batch_size,
                grad_accumulation, resolution, aspect_buckets, frames, fps, start, trigger, seed, training_dtype, lora_dtype,
                gradient_checkpointing, offloading, bypass, model=None, audio_vae=None, model_name="", weight_dtype="default", compute_dtype="default", postfix="") -> io.NodeOutput:
        if not addons.enabled("lora_training"):
            raise RuntimeError("LoRA training is switched off. Turn it on in Settings > Modules.")
        cfg = {"name": name, "use": use, "steps": steps, "learning_rate": learning_rate, "rank": rank, "algorithm": algorithm,
               "optimizer": optimizer, "loss": loss, "batch_size": batch_size, "grad_accumulation": grad_accumulation,
               "resolution": resolution, "aspect_buckets": aspect_buckets, "frames": frames, "fps": fps, "start": start,
               "trigger": trigger, "seed": seed, "training_dtype": training_dtype, "lora_dtype": lora_dtype,
               "gradient_checkpointing": gradient_checkpointing, "offloading": offloading, "bypass": bypass,
               "model_name": model_name, "weight_dtype": weight_dtype, "compute_dtype": compute_dtype, "postfix": postfix}
        return io.NodeOutput(train.run(model, clip, vae, audio_vae, set_name, cfg))
