# Changelog

## [5.0.0] "Cunning Cranberry" - unreleased

A rewrite from scratch, not an update of 4.x. Same idea (a studio for video inside ComfyUI that learns from your ratings), new architecture: every feature is a module that announces itself, so a feature cannot be forgotten in the UI, a wire cannot be missed, and one failing module never stops the run. 4.x projects are not imported; see "Not carried over".

### What it is

- **A browser studio at `/funpack/`**: timeline with one clip per scene, Composer (story in one box, cut word, templates, `$variables`, shortcuts), properties, media bin, preview, takes, rating, Export, Render, project wizard and guided tour.
- **Models & Pipeline**: the pipeline as cards. Pick files, swap or add nodes (any installed custom node), wire one node's output into another's input, import a ComfyUI workflow. The server refuses what would not build and says why.
- **Modules** (Settings ▸ Modules): each feature switches on or off on its own and shows whether it is proven or experimental. A module that fails during a run is switched off for the rest of it, and the log says why.
- **Learning from ratings**: a Taste key remembers ratings; the guidance, steering and prompt modules that use them start acting after a few liked clips and say so when they need more.

### Models

- **MiniMax H3**: the model v5 is built and tested around. Presets for reference to video; works text-only (no source picture); second-pass preset (needs a compatible H3 latent upscaler file, not verified).
- **Stable Diffusion 1.5**: experimental.
- **LTX 2.x**: **UNTESTED.** Never run on a real LTX model, and nobody plans to test it soon. Presets are marked UNTESTED and every LTX run logs the same notice. You are free to try it; please report problems in Issues.

### Features

- **Proven**: prompt markup, empty latent, second pass, loaders (diffusion model, CLIP, VAE, LoRA, checkpoint, media), cut opening, decode, save video, anchor de-staticking (ALG), randomizer, DynaShift, first-step seed search, modifiers, sampler, detail.
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
- **`.gguf` files** load only for architectures ComfyUI-GGUF understands. Gemma 4 `.gguf` does not work; a loader for it may come later.
- **H3 start frame**: the scene picture sets the video's shape only; image-to-video from it is not implemented.
- **H3 second pass** is not verified.
- Turning some modules on switches the model compiler off for the run, which costs sampling time.

### Docs

- [How to start generating with FunPack](docs/how-to-start.md).
