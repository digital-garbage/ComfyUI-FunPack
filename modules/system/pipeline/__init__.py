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

`group` is the pipeline's own idea of how it reads to a person -- the cards the
app shows are these names, in the order they first appear. It travels on the
slot rather than living in the app, so a pipeline that replaced every slot keeps
its own arrangement, and a group is created by a slot claiming a name nobody
used yet rather than by registering it anywhere.
"""

ID = "pipeline"
TITLE = "Pipeline"
STAGE = "load"
STATUS = "proven"

# Slot ids are stable: they are how an override says which slot it means, and
# they end up as node ids in the queued prompt.
DEFAULT = [
    {"id": "model", "group": "Loaders", "node": "FunPackDiffusionModelLoader", "inputs": {
        "weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
    {"id": "clip", "group": "Loaders", "node": "FunPackCLIPLoader", "inputs": {"type": "stable_diffusion"}},
    {"id": "vae", "group": "Loaders", "node": "FunPackVAELoader", "inputs": {"dtype": "bfloat16"}},

    {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": ""}},
    {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode", "inputs": {
        "clip": ["clip", 0], "text": ""}},

    {"id": "latent", "group": "Preparation", "node": "FunPackEmptyLatent", "inputs": {
        "model": ["model", 0], "width": 512, "height": 512, "length": 1, "batch_size": 1}},

    {"id": "settings", "group": "Preparation", "node": "FunPackModifierSettings", "inputs": {"settings": "{}"}},
    {"id": "modifiers", "group": "Preparation", "node": "FunPackLoadModifiers", "inputs": {
        "model": ["model", 0], "settings": ["settings", 0]}},

    {"id": "sampler", "group": "Sampling", "node": "FunPackSampler", "inputs": {
        "model": ["modifiers", 0], "positive": ["positive", 0], "negative": ["negative", 0],
        "latent": ["latent", 0], "settings": ["settings", 0],
        "seed": 0, "steps": 20, "cfg": 7.0,
        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},

    {"id": "decode", "group": "Render", "node": "FunPackDecode", "inputs": {
        "samples": ["sampler", 0], "vae": ["vae", 0], "model": ["model", 0]}},

    {"id": "save", "group": "Render", "node": "SaveImage", "inputs": {
        "images": ["decode", 0], "filename_prefix": "FunPack"}},
]


def default():
    """A fresh copy: a caller that edits the pipeline must not edit everyone's."""
    return [dict(slot, inputs=dict(slot.get("inputs") or {})) for slot in DEFAULT]


PROVIDES = {"default_pipeline": default}
