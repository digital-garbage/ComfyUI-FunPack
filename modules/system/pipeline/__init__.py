"""The default pipeline, as data.

Every slot here can be pointed at a different node or taken out entirely, and
that is only true because this is a list rather than a function. v4's equivalent
was a `CORE` dict inside the builder, so "remove this node" had nothing to
delete and "use a different node" had nothing to change -- the pipeline was code,
and code is not a thing a user can edit.

The rule for what belongs here: enough to generate with nothing else wired. A
run with no overrides at all has to produce a picture, because that is what makes
the UI an override layer rather than a requirement.

Nothing about these slots is privileged. Core does not know this list exists; it
is handed one and builds it.
"""

ID = "pipeline"
TITLE = "Pipeline"
STAGE = "load"
STATUS = "proven"

# Slot ids are stable: they are how an override says which slot it means, and
# they end up as node ids in the queued prompt.
DEFAULT = [
    {"id": "model", "node": "FunPackDiffusionModelLoader", "inputs": {
        "weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
    {"id": "clip", "node": "FunPackCLIPLoader", "inputs": {"type": "stable_diffusion"}},
    {"id": "vae", "node": "FunPackVAELoader", "inputs": {"dtype": "bfloat16"}},

    {"id": "positive", "node": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": ""}},
    {"id": "negative", "node": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": ""}},

    {"id": "latent", "node": "FunPackEmptyLatent", "inputs": {
        "model": ["model", 0], "width": 512, "height": 512, "length": 1, "batch_size": 1}},

    {"id": "settings", "node": "FunPackModifierSettings", "inputs": {"settings": "{}"}},
    {"id": "modifiers", "node": "FunPackLoadModifiers", "inputs": {
        "model": ["model", 0], "settings": ["settings", 0]}},

    {"id": "sampler", "node": "FunPackSampler", "inputs": {
        "model": ["modifiers", 0], "positive": ["positive", 0], "negative": ["negative", 0],
        "latent": ["latent", 0], "settings": ["settings", 0],
        "seed": 0, "steps": 20, "cfg": 7.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},

    {"id": "decode", "node": "FunPackDecode", "inputs": {
        "samples": ["sampler", 0], "vae": ["vae", 0], "model": ["model", 0]}},

    {"id": "save", "node": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "FunPack"}},
]


def default():
    """A fresh copy: a caller that edits the pipeline must not edit everyone's."""
    return [dict(slot, inputs=dict(slot.get("inputs") or {})) for slot in DEFAULT]


PROVIDES = {"default_pipeline": default}
