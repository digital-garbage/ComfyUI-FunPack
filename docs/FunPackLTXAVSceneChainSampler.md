# FunPack LTXAV Scene Chain Sampler

`FunPack LTXAV Scene Chain Sampler` samples multi-entry scene conditioning as one smooth continuation run.

Use it with `FunPack Studio` or `FunPack Video Refiner V2` when `split_by_transitions` is enabled. The refiner returns one positive conditioning entry per detected scene, and this sampler uses each entry for one sequential chunk.

## Inputs

- `model`: LTXV or LTXAV model.
- `vae`: VAE with the LTX time scale metadata.
- `positive`: Scene conditioning list. One entry becomes one generated scene.
- `negative`: Negative conditioning shared by every scene.
- `sampler`: ComfyUI sampler object, for example from Studio's sampler output.
- `sigmas`: Sigma schedule for each chunk.
- `seed`: Base seed. Scene N uses `seed + N`.
- `latent_template`: One scene-sized latent template. Plain video and nested AV latents are supported.
- `num_frames_per_scene`: Pixel frame count represented by `latent_template`.
- `frame_overlap`: Pixel frames to preserve and blend between scene chunks.
- `cfg`: Internal CFG value.
- `max_scenes`: Maximum scene entries to consume, capped at 8.

## Behavior

The first scene samples from a fresh copy of `latent_template`.

Each following scene copies the previous output tail into the start of the next chunk, masks that overlap so it is preserved during denoising, samples the new frames with that scene's conditioning, then blends the overlap in latent space.

For nested LTXAV latents, video and audio tensors are continued together. Audio overlap is derived from the audio/video latent length ratio.

## Notes

Multi-entry conditioning from `split_by_transitions` is meant for this sampler. Connecting it to a normal sampler can mix scene conditionings together instead of routing one scene per chunk.
