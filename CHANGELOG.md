# Changelog

## [5.0.0] "Cunning Cranberry" - unreleased

A rewrite from scratch, not an update of 4.x. Same idea (a studio for video inside ComfyUI that learns from your ratings), new architecture: every feature is a module that announces itself, so a feature cannot be forgotten in the UI, a wire cannot be missed, and a failing feature module is left out while the run carries on. 4.x projects are not imported; see "Not carried over".

### What it is

- **A browser studio at `/funpack/`**: timeline with one clip per scene, Composer (story in one box, cut word, templates, `$variables`, shortcuts), properties, media bin, preview, takes, rating, Export, Render, project wizard and guided tour.
- **Models & Pipeline**: the pipeline as cards. Pick files, swap or add nodes (any installed custom node), wire one node's output into another's input, import a ComfyUI workflow. The server refuses what would not build and says why.
- **Modules** (Settings ▸ Modules): each feature switches on or off on its own. A module that fails during a run is left out of it, the log says why, and it stays off until its code changes or you turn it back on.
- **Learning from ratings**: a Taste key remembers ratings; the guidance, steering and prompt modules that use them start acting once they have enough ratings (a few liked clips; some also need disliked ones) and say so when they need more.

### Models

- **MiniMax H3**: the model v5 is built and tested around. Presets for reference to video; works text-only (no source picture). A second-pass preset exists in the code but is not offered until it has been verified.
- **Stable Diffusion 1.5**: experimental.
- **LTX 2.x**: **UNTESTED.** Never run on a real LTX model, and nobody plans to test it soon. Presets and the module are marked UNTESTED, and LTX runs log the same notice. You are free to try it; please report problems in Issues.

### Features

- **Proven**: prompt markup, empty latent, second pass (the H3 preset for it is held back), loaders (diffusion model, CLIP, VAE, LoRA, checkpoint, media), cut opening, decode, save video, anchor de-staticking (ALG), randomizer, DynaShift, first-step seed search, modifiers, sampler, detail.
- **Experimental**: shot camera, continuity, identity transfer, negative erase, prompt enhancer, Render, taste modules (slider, guidance, prompt nudge, attention, REINS, early guidance), shot memory, decisiveness, late-branch guidance, camera move, context windows, block influence, block repeat, phrase probe, shadow negative, momentum, velocity bias, video detail, temporal style, massive-activation steering, video-to-audio coupling.
- **Prompt enhancer**: a language model rewrites each finished prompt. Has its own encoder pick (Composer ▸ Enhance ▸ Encoder), because H3's own encoder cannot write prose. Verified with Gemma 4.
- **Auto Montage**: builds a new cut at the end of the timeline from your rendered clips; originals stay untouched, and the pieces are never regenerated or rated.
- **Upscale** a finished render with an upscale model (`.safetensors`) or NVIDIA RTX Super Resolution.
- **Add-ons** (spaCy for automatic camera and focus, nvidia-vfx) install on the next start after you switch them on.

### Not carried over

- 4.x projects and saved settings are not imported: start new projects in 5. ComfyUI workflow files can still be imported in Models & Pipeline.
- Features that never proved out stay out: plateau step-cache, block steering, NVFP4 loader.
- The unwired Engine Appearance module (the theme lives in Settings ▸ Appearance). Settings saved while it existed are ignored, not refused.

### Known limits

- **LTX 2.x is untested** (see Models).
- **`.gguf` text encoders** work only for architectures ComfyUI-GGUF understands; others are refused with a message. Gemma 4 `.gguf` does not work; a loader for it may come later.
- **H3 start frame**: in the H3 preset the scene picture sets the video's shape only; a start frame is not part of it.
- **H3 second pass** is held back: not offered until verified.
- Turning some modules on switches the model compiler off for the run, which costs sampling time.

### Docs

- [How to start generating with FunPack](docs/how-to-start.md).
