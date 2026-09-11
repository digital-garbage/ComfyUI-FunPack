"""The MiniMax H3 · Reference to Video pipeline, as a loadable preset.

Not the app's one true default -- `modules/system/pipeline` already owns that,
and a second module silently claiming `default_pipeline` would make which one
actually starts depend on load order. This is offered as a PRESET instead: a
slot list a person picks from the pipeline window, same shape as the default,
loaded into the very same editable graph (`core/graph.py`'s wire/replace/remove
apply to it exactly as they do to anything else).

The chain mirrors what this app's own H3 user actually runs by hand today:
loaders, MiniMaxH3SigmaShift patching the model's flow shifts, an
ImageTransformKJ node turning a dropped-in picture into canvas-multiple
width/height ints -- but the picture supplies only its ASPECT RATIO, not the
resolution. The actual pixel budget is target_width*target_height, role-
mounted at "project.video" exactly like the default pipeline's own
width/height: a project setting, typed once, not implied by whatever a phone
camera happened to produce. "total_pixels" keep_proportion is what makes that
true -- it fits the image's own aspect ratio into that budget's area, rather
than using the image's raw size. Those two resulting ints feed
MiniMaxH3ReferenceToVideo's width/height, and Reference-to-Video's own
conditioning+latent feeds a plain sampler at CFG 1 (H3 wants no negative
guidance) into FunPackDecode and a video/audio mux.

The image and each reference are FunPackLoadMedia nodes: fed by a `media_id`
this app sets from what is picked in the Media bin, at "assets.*" roles that
mount no on-screen field of their own (nothing here `offer()`s an `assets.*`
region) -- the same "a role naming a place nobody offers is simply not shown"
rule as everywhere else, used here on purpose so boot.js can still find these
slots BY ROLE (never by this preset's own slot ids) without the app rendering
a raw text box asking someone to type a media id by hand.
"""

#: How many reference images this preset wires a slot for. R2V natively takes
#: up to 9; four is the common case and, like the CLIP loader's four-encoder
#: ceiling, raising it later only adds optional slots -- backwards compatible
#: with anything already saved.
MAX_REFERENCES = 4


def h3_reference_to_video():
    slots = [
        {"id": "model", "group": "Loaders", "node": "FunPackDiffusionModelLoader", "inputs": {
            "weight_dtype": "default", "compute_dtype": "default", "attention": "default"}},
        {"id": "clip", "group": "Loaders", "node": "FunPackCLIPLoader",
         "inputs": {"type": "minimax"}},
        {"id": "vae", "group": "Loaders", "node": "FunPackVAELoader",
         "inputs": {"dtype": "bfloat16"}},
        {"id": "audio_vae", "group": "Loaders", "node": "FunPackVAELoader",
         "inputs": {"dtype": "bfloat16"}},

        {"id": "shift", "group": "Sampling", "node": "MiniMaxH3SigmaShift",
         "roles": [{"at": "generation.model", "input": "shift_video", "label": "Video shift"},
                   {"at": "generation.model", "input": "shift_audio", "label": "Audio shift"}],
         "inputs": {"model": ["model", 0], "shift_video": 12.0, "shift_audio": 3.0}},

        {"id": "source_image", "group": "Reference media", "node": "FunPackLoadMedia",
         # Unhosted: see the module docstring. boot.js finds this slot by this
         # role's name and sets `media_id` from the scene's own picked image,
         # the same way "generation.prompt" is found by name and set from the
         # scene's own text -- neither is addressed by this preset's slot id.
         "roles": [{"at": "assets.source_image", "input": "media_id"}],
         "inputs": {"media_id": ""}},
        {"id": "image_transform", "group": "Reference media", "node": "ImageTransformKJ",
         # target_width/height are the PIXEL BUDGET, set in Project settings
         # like any other project resolution -- not the dropped picture's own
         # size. "total_pixels" is what makes that budget win: it takes
         # target_width*target_height as an area and fits the image's own
         # aspect ratio into it, so the picture supplies the SHAPE and the
         # project supplies the SCALE. A 12-megapixel phone photo dropped in
         # must not become the generation's actual resolution.
         "roles": [{"at": "project.video", "input": "target_width", "label": "Width"},
                   {"at": "project.video", "input": "target_height", "label": "Height"}],
         "inputs": {
            "image": ["source_image", 0], "target_width": 1344, "target_height": 768,
            "upscale_method": "lanczos",
            "keep_proportion": {"keep_proportion": "total_pixels"},
            "divisible_by": 32,
            "extra_padding": {"extra_padding": "disabled"},
            "invert_crop": {"invert_crop": "disabled"},
            "bboxes": ""}},

        {"id": "r2v", "group": "Preparation", "node": "MiniMaxH3ReferenceToVideo",
         "roles": [{"at": "generation.prompt", "input": "prompt", "label": "Prompt"},
                   {"at": "project.video", "input": "length", "label": "Length"}],
         "inputs": {
             "clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["audio_vae", 0],
             "prompt": "", "width": ["image_transform", 4], "height": ["image_transform", 5],
             "length": 124, "ref_image_size": "match"}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         # H3 samples at CFG 1, where a negative prompt does nothing -- wired
         # only because FunPackSampler requires SOME conditioning here.
         "inputs": {"clip": ["clip", 0], "text": ""}},

        {"id": "settings", "group": "Preparation", "node": "FunPackModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunPackLoadModifiers", "inputs": {
            "model": ["shift", 0], "settings": ["settings", 0]}},

        {"id": "sampler", "group": "Sampling", "node": "FunPackSampler",
         "roles": [{"at": "generation.sampling", "input": "steps", "label": "Steps"},
                   {"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.sampling", "input": "scheduler", "label": "Scheduler"}],
         "inputs": {
             "model": ["modifiers", 0], "positive": ["r2v", 0], "negative": ["negative", 0],
             "latent": ["r2v", 1], "settings": ["settings", 0],
             "seed": 0, "steps": 20, "cfg": 1.0,
             "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},

        {"id": "decode", "group": "Render", "node": "FunPackDecode", "inputs": {
            "samples": ["sampler", 0], "vae": ["vae", 0], "model": ["shift", 0],
            "audio_vae": ["audio_vae", 0]}},
        {"id": "video", "group": "Render", "node": "CreateVideo", "inputs": {
            "images": ["decode", 0], "fps": 24.0, "audio": ["decode", 1]}},
        {"id": "save", "group": "Render", "node": "SaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": "FunPack",
            "format": {"format": "auto"}}},
    ]

    for n in range(1, MAX_REFERENCES + 1):
        slots.append({
            "id": f"ref_source_{n}", "group": "Reference media", "node": "FunPackLoadMedia",
            # `wireTo` is not read by core or by graph.py -- it is this
            # preset's own data, read by boot.js at queue time (see
            # queueInputs()'s handling of "assets.reference_N" there) to know
            # where a reference's IMAGE output should land WITHOUT boot.js
            # having to know "r2v" or "ref_images" itself. Not wired here in
            # the STATIC slot list -- see the module docstring above: an
            # unwired optional autogrow slot is what "this reference is not
            # in use" actually looks like to MiniMaxH3ReferenceToVideo,
            # absent from its `ref_images` dict entirely, not an empty image
            # degrading its output. Only a scene that actually has this many
            # references gets this wire added, per-run.
            "roles": [{"at": f"assets.reference_{n}", "input": "media_id",
                       "wireTo": {"slot": "r2v", "input": f"ref_images.ref_image_{n - 1}"}}],
            "inputs": {"media_id": ""}})

    return slots


def presets():
    return [{"id": "minimax_h3_reference_to_video",
             "title": "MiniMax H3 · Reference to Video",
             "slots": h3_reference_to_video()}]


PROVIDES = {"pipeline_presets": presets}
