# FunPack LTXAV Scene Chain Sampler

`FunPack LTXAV Scene Chain Sampler` samples multi-entry scene conditioning as one smooth continuation run.

Use it with `FunPack Studio` or `FunPack Video Refiner V2` when `split_by_transitions` is enabled. The refiner returns one positive conditioning entry per detected scene, and this sampler uses each entry for one sequential chunk.

Important: this sampler is resource heavy. Long chains can produce very large final latents. You may run out of memory during VAE Decode even if every sampling chunk completed successfully.

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
- `max_scenes`: Maximum scene entries to consume. Default is `8`, but it can be raised for longer chains.
- `carry_i2v_guides`: Reuses protected frames from `latent_template`'s `noise_mask` in each continuation chunk after the overlap. Leave enabled for i2v/keyframe consistency.

## Behavior

The first scene samples from a fresh copy of `latent_template`.

Each following scene copies the previous output tail into the start of the next chunk, masks that overlap so it is preserved during denoising, samples the new frames with that scene's conditioning, then blends the overlap in latent space.

When `carry_i2v_guides` is enabled, protected source frames from the incoming `latent_template` are carried into later chunks after the preserved overlap. This keeps LTXV's native i2v/keyframe conditioning alive across scene chunks without needing a separate identity latent.

For nested LTXAV latents, video and audio tensors are continued together. Audio overlap is derived from the audio/video latent length ratio.

## Notes

Multi-entry conditioning from `split_by_transitions` is meant for this sampler. Connecting it to a normal sampler can mix scene conditionings together instead of routing one scene per chunk.

Scene order is first in, first out. Written labels like `scene ten`, `scene -999999`, or `scene minus infinity` are treated as transition text only. They do not assign scene numbers.

For character consistency, keep the character or subject description before the first transition. Refiner V2 and Studio prepend that prefix to every detected scene conditioning.

VAE Decode memory use grows with the final stitched latent length. If decode fails with OOM, reduce `max_scenes`, lower `num_frames_per_scene`, decode shorter chains, or use a lower-memory decode path if your workflow provides one.
