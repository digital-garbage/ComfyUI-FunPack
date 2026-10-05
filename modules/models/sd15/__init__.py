"""Stable Diffusion 1.5 support: a small model the whole pipeline can be run end to end on a Mac.

It is an image model, so its pipeline holds the one picture for the clip's length: the editor gets a
clip the length it asked for, made the same way every other pipeline makes one. The latent and the
decode need nothing of their own (a plain 4-channel /8 latent, derived by core from the model).

What this teaches the system is only what the file's header says before it loads: that it is SD1.x
(a CLIP-L text encoder under `cond_stage_model.transformer`, which SD2's OpenCLIP and SDXL's
`conditioner` do not have), and the facts core would read off the loaded model anyway, so every
video-only module hides the moment the file is picked.
"""

ID = "model_sd15"
TITLE = "Stable Diffusion 1.5"
STAGE = "load"
CATEGORY = "system"
STATUS = "experimental"

_SIGNATURE_KEYS = ("model.diffusion_model.input_blocks.0.0.weight",
                   "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")


def detect(keys) -> bool:
    keyset = set(keys)
    return all(key in keyset for key in _SIGNATURE_KEYS)


def probe_traits(keys) -> list:
    """What core's `universal()` reads off a loaded SD1.5: a 2-axis latent, noise prediction."""
    return ["spatial_latent", "predict_eps"] if detect(keys) else []


def text_to_image():
    video = {"at": "project.video", "input": "fps", "label": "FPS", "drives": "fps"}
    return [
        {"id": "model", "group": "Loaders", "node": "FunPackCheckpointLoader", "inputs": {"ckpt_name": ""}},
        {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"}],
         "inputs": {"clip": ["model", 1], "text": ""}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["model", 1], "text": ""}},
        {"id": "latent", "group": "Preparation", "node": "FunPackEmptyLatent",
         "roles": [{"at": "project.video", "input": "width", "label": "Width"},
                   {"at": "project.video", "input": "height", "label": "Height"}],
         "inputs": {"model": ["model", 0], "width": 512, "height": 512, "length": 1, "batch_size": 1}},
        {"id": "settings", "group": "Preparation", "node": "FunPackModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunPackLoadModifiers",
         "inputs": {"model": ["model", 0], "settings": ["settings", 0]}},
        {"id": "sampler", "group": "Sampling", "node": "FunPackSampler",
         "roles": [{"at": "generation.sampling", "input": "steps", "label": "Steps"},
                   {"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.sampling", "input": "scheduler", "label": "Scheduler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {"model": ["modifiers", 0], "positive": ["positive", 0], "negative": ["negative", 0],
                    "latent": ["latent", 0], "settings": ["settings", 0],
                    "seed": 0, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0}},
        {"id": "decode", "group": "Render", "node": "FunPackDecode",
         "inputs": {"samples": ["sampler", 0], "vae": ["model", 2], "model": ["model", 0]}},
        {"id": "hold", "group": "Render", "node": "RepeatImageBatch",
         "roles": [{"at": "project.video", "input": "amount", "label": "Length", "drives": "frames"}],
         "inputs": {"image": ["decode", 0], "amount": 1}},
        {"id": "video", "group": "Render", "node": "CreateVideo", "roles": [video],
         "inputs": {"images": ["hold", 0], "fps": 24.0}},
        {"id": "save", "group": "Render", "node": "SaveVideo",
         "inputs": {"video": ["video", 0], "filename_prefix": "FunPack", "format": "auto"}},
    ]


def presets():
    return [{"id": "sd15_text_to_image", "title": "Stable Diffusion 1.5 · Text to Image (held as a clip)",
             "slots": text_to_image()}]


PROVIDES = {"detect": detect, "probe_traits": probe_traits, "pipeline_presets": presets}
