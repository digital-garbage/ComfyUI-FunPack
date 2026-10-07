"""LTX 2.3 pipelines, as loadable presets.

Built from ComfyUI's own LTX nodes plus FunPack's, the chain the stock templates use: the
checkpoint carries the model and the picture VAE, the audio VAE and the text encoder are loaded
from the same file, conditioning carries the frame rate, and the sampler runs the distilled
schedule written out as numbers (`project_sigmas`: the official anchors, never replaced).

The checkpoint is picked three times (model, audio VAE, text encoder): they are three nodes that
each read a file, and the first two are the same file.

Every `frame_rate` and `fps` is a role at `project.video`, so the project's FPS is the one number
the conditioning, the sound's length and the saved video agree on.
"""

#: The distilled 8-step schedule, from the official ComfyUI LTX 2.x templates.
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def _base(image_to_video: bool, as_guide: bool = False):
    video = {"at": "project.video", "input": "frame_rate", "label": "FPS", "drives": "fps"}
    latent_source = ["latent", 0]
    model_source, positive_source, negative_source = ["modifiers", 0], ["conditioning", 0], ["conditioning", 1]
    slots = [
        {"id": "model", "group": "Loaders", "node": "FunPackCheckpointLoader", "inputs": {"ckpt_name": ""}},
        {"id": "audio_vae", "group": "Loaders", "node": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ""}},
        {"id": "clip", "group": "Loaders", "node": "LTXAVTextEncoderLoader",
         "inputs": {"text_encoder": "", "ckpt_name": "", "device": "default"}},

        {"id": "positive", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "generation.prompt", "input": "text", "label": "Prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "negative", "group": "Preparation", "node": "CLIPTextEncode",
         "roles": [{"at": "project.negative", "input": "text", "label": "Negative prompt"}],
         "inputs": {"clip": ["clip", 0], "text": ""}},
        {"id": "conditioning", "group": "Preparation", "node": "LTXVConditioning",
         "roles": [video],
         "inputs": {"positive": ["positive", 0], "negative": ["negative", 0], "frame_rate": 25.0}},

        {"id": "latent", "group": "Preparation", "node": "FunPackEmptyLatent",
         "roles": [{"at": "project.video", "input": "width", "label": "Width"},
                   {"at": "project.video", "input": "height", "label": "Height"},
                   {"at": "project.video", "input": "length", "label": "Length", "drives": "frames"},
                   video],
         "inputs": {"model": ["model", 0], "audio_vae": ["audio_vae", 0],
                    "width": 768, "height": 512, "length": 121, "frame_rate": 25.0, "batch_size": 1}},
    ]
    if image_to_video:
        slots += [
            {"id": "source_image", "group": "Reference media", "node": "FunPackLoadMedia",
             "roles": [{"at": "assets.source_image", "input": "media_id"}],
             "inputs": {"media_id": ""}},
            {"id": "split", "group": "Preparation", "node": "LTXVSeparateAVLatent",
             "inputs": {"av_latent": ["latent", 0]}},
        ]
        if as_guide:
            # The picture steers through guide attention at frame 0 and is NOT the clip's first frame:
            # the latent stays empty, so the opening is free to move. Strength is the slot's own number.
            slots += [
                {"id": "anchor", "group": "Preparation", "node": "LTXVAddGuide",
                 "inputs": {"positive": ["conditioning", 0], "negative": ["conditioning", 1], "vae": ["model", 2],
                            "latent": ["split", 0], "image": ["source_image", 0], "frame_idx": 0, "strength": 0.35}},
                {"id": "join", "group": "Preparation", "node": "LTXVConcatAVLatent",
                 "inputs": {"video_latent": ["anchor", 2], "audio_latent": ["split", 1]}},
            ]
            guided = (["anchor", 0], ["anchor", 1])
        else:
            slots += [
                {"id": "anchor", "group": "Preparation", "node": "LTXVImgToVideoInplace",
                 "inputs": {"vae": ["model", 2], "image": ["source_image", 0], "latent": ["split", 0],
                            "strength": 1.0, "bypass": False}},
                {"id": "join", "group": "Preparation", "node": "LTXVConcatAVLatent",
                 "inputs": {"video_latent": ["anchor", 0], "audio_latent": ["split", 1]}},
            ]
            guided = (["conditioning", 0], ["conditioning", 1])
        latent_source = ["join", 0]
        positive_source, negative_source = guided
    slots += [
        {"id": "settings", "group": "Preparation", "node": "FunPackModifierSettings", "inputs": {"settings": "{}"}},
        {"id": "modifiers", "group": "Preparation", "node": "FunPackLoadModifiers",
         "inputs": {"model": ["model", 0], "settings": ["settings", 0]}},
    ]
    if image_to_video:
        slots += [
            # A reference face carried into the clip. The picture is the scene's first reference
            # (assets.reference_1): with none picked this node passes everything through untouched.
            {"id": "identity", "group": "Preparation", "node": "FunPackIdentityTransfer",
             "inputs": {"model": ["modifiers", 0], "vae": ["model", 2], "positive": positive_source,
                        "negative": negative_source, "latent": latent_source, "source_id": 2.0,
                        "phase_scale": 1.0, "id_strength": 1.0, "arcface_mode": "auto_adjust"}},
            {"id": "identity_source", "group": "Reference media", "node": "FunPackLoadMedia",
             "roles": [{"at": "assets.reference_1", "input": "media_id",
                        "wireTo": {"slot": "identity", "input": "image"}}],
             "inputs": {"media_id": ""}},
        ]
        model_source, positive_source, negative_source = ["identity", 0], ["identity", 1], ["identity", 2]
    slots += [
        {"id": "sigmas", "group": "Sampling", "node": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
        {"id": "sampler", "group": "Sampling", "node": "FunPackSampler",
         "roles": [{"at": "generation.sampling", "input": "sampler_name", "label": "Sampler"},
                   {"at": "generation.seed", "input": "seed", "label": "Seed"}],
         "inputs": {"model": model_source, "positive": positive_source, "negative": negative_source,
                    "latent": latent_source, "settings": ["settings", 0], "sigmas": ["sigmas", 0],
                    "seed": 0, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0}},

        *_uncrop(as_guide),
        {"id": "decode", "group": "Render", "node": "FunPackDecode", "inputs": {
            "samples": ["rejoin", 0] if as_guide else ["sampler", 0], "vae": ["model", 2], "model": ["model", 0], "audio_vae": ["audio_vae", 0]}},
        {"id": "video", "group": "Render", "node": "CreateVideo",
         "roles": [{"at": "project.video", "input": "fps", "label": "FPS", "drives": "fps"}],
         "inputs": {"images": ["decode", 0], "fps": 25.0, "audio": ["decode", 1]}},
        {"id": "save", "group": "Render", "node": "FunPackSaveVideo", "inputs": {
            "video": ["video", 0], "filename_prefix": "FunPack"}},
    ]
    return slots


def _uncrop(as_guide):
    """A guide sits in the latent as extra frames: take them off again before the picture is decoded."""
    if not as_guide:
        return []
    return [
        {"id": "unjoin", "group": "Render", "node": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["sampler", 0]}},
        {"id": "crop", "group": "Render", "node": "LTXVCropGuides",
         "inputs": {"positive": ["identity", 1], "negative": ["identity", 2], "latent": ["unjoin", 0]}},
        {"id": "rejoin", "group": "Render", "node": "LTXVConcatAVLatent",
         "inputs": {"video_latent": ["crop", 2], "audio_latent": ["unjoin", 1]}},
    ]


def presets():
    return [{"id": "ltx23_text_to_video", "title": "LTX 2.3 · Text to Video", "slots": _base(False)},
            {"id": "ltx23_image_to_video", "title": "LTX 2.3 · Image to Video", "slots": _base(True)},
            {"id": "ltx23_anchor_guide", "title": "LTX 2.3 · Anchor as guide (image steers, opening stays free)",
             "slots": _base(True, True)}]


PROVIDES = {"pipeline_presets": presets}
