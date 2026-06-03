# FunPack LTXAV Scene Chain Sampler

`FunPack LTXAV Scene Chain Sampler` samples multi-entry scene conditioning as one smooth continuation run.

Use it with `FunPack Studio` or `FunPack Video Refiner V2` when `split_by_transitions` is enabled. The refiner returns one positive conditioning entry per detected scene, and this sampler uses each entry for one sequential chunk.

You can also use it without Studio. Encode each scene separately, then combine those conditioning outputs with ComfyUI's `ConditioningCombine`. The sampler treats each conditioning entry in the combined list as one scene.

Important: this sampler is resource heavy. Long chains can produce very large final latents. You may run out of memory during VAE Decode even if every sampling chunk completed successfully.

## Inputs

- `model`: LTXV or LTXAV model.
- `vae`: VAE with the LTX time scale metadata.
- `positive`: Scene conditioning list. One entry becomes one generated scene.
- `negative`: Negative conditioning shared by every scene.
- `sampler`: ComfyUI sampler object, for example from Studio's sampler output.
- `sigmas`: Sigma schedule for each chunk.
- `seed`: Base seed. Scene N uses `seed + N` unless scene seed metadata is present.
- `latent_template`: One scene-sized latent template. Plain video and nested AV latents are supported.
- `num_frames_per_scene`: Pixel frame count represented by `latent_template`.
- `frame_overlap`: Pixel frames to preserve and blend between scene chunks. Set to 0 to disable overlap entirely. **Warning:** combining `frame_overlap=0` with `carry_i2v_guides=True` is confirmed to produce bad results — use only for testing.
- `cfg`: Internal CFG value.
- `max_scenes`: Maximum scene entries to consume. Default is `8`, but it can be raised for longer chains.
- `use_same_seed`: When off, each scene uses `funpack_scene_seed` metadata from Studio/Refiner split mode, falling back to `seed + scene_index`. When on, every scene uses the first provided scene seed or the base `seed`.
- `carry_i2v_guides`: Experimental. Appends protected frames from `latent_template`'s `noise_mask` as hidden LTX guide tokens in each continuation chunk. Default is off. **Warning:** using this with `frame_overlap=0` is confirmed to produce bad results — enable only for testing.
- `mid_scene_guide`: Experimental. Appends the middle frame of the previous scene as a guide for the current scene via LTX guide attention, helping maintain character positioning and static-element layout across scenes. Default is off. (Replaced the older `self_consistency` feature, which corrupted audio through joint attention.)
- `mid_scene_guide_strength`: Guide-attention strength for the mid-scene anchor. `0.25` is the minimum — below that audio degrades and character appearance drifts; above `~0.35` it causes spatial conflicts when scene composition shifts. Capped at `0.5`.
- `embed_guidance`: Applies the Refiner's learned liked-quality conditioning direction at *every* denoising step, not just once before sampling. Requires `refinement_key_input` and enough liked generations to have formed a direction. Adds ~20–30% inference overhead. Default is off.
- `embed_guidance_strength`: Per-step nudge strength toward the liked direction. Keep small — it is applied at every step so it compounds; `0.01`–`0.03` is typical.
- `transition_duration`: Extra pixel frames of fade beyond the blend zone on each side of a scene boundary. `0` disables all transition effects.
- `decode_tile_size` *(optional)*: Tile size for VAE decode (`0` = no tiling). Set to e.g. `512` if decode OOMs.
- `refinement_key_input` *(optional)*: Connect to the same refinement key as your V2 Refiner. When wired, the sampler writes `carry_i2v_guides`, `frame_overlap`, and the scene count into the refinement state so the Refiner can reason about what changed between rated runs, and it enables the value-function-driven `embed_guidance` direction.

## Outputs

`latent`, `images`, `status`, `scene_count`, `scene_report`, `scene_boundaries`. The `scene_report` and `scene_boundaries` describe how the chain was split and stitched; the boundaries can be used downstream to locate scene cuts in the decoded video.

## Behavior

The first scene samples from a fresh copy of `latent_template`.

Each following scene copies the previous output tail into the start of the next chunk, masks that overlap so it is preserved during denoising, samples the new frames with that scene's conditioning, then blends the overlap in latent space.

When Studio or Refiner V2 split mode provides scene seed metadata, the sampler reports and uses those exact seeds. This lets successful seed memory replay a known-good scene seed set while keeping the public Studio `seed` output as a single integer.

When `carry_i2v_guides` is enabled, protected source frames from the incoming `latent_template` are appended as guide tokens with `keyframe_idxs` and `guide_attention_entries`, then cropped away after sampling. This follows the same broad idea as LTX guide/IC-LoRA conditioning: the reference is extra context, not a visible frame inserted into the generated timeline. Keep it off unless you are testing that behavior deliberately. **Combining `carry_i2v_guides=True` with `frame_overlap=0` is confirmed to produce bad results and is not recommended for normal use.**

For nested LTXAV latents, video and audio tensors are continued together. Audio overlap is derived from the audio/video latent length ratio.

### Batch Training

This sampler is also the engine for **Batch Training**. When Studio packs a batch of variants into the conditioning, the sampler runs the chain `N` times with everything frozen except the per-variant noise seed, producing `N` directly-comparable videos for rating. Distinct per-variant seeds are used in both the split-scene and single-scene paths (identical seeds would otherwise produce identical videos), and the node's own rating is ignored while a batch is in progress. See `FunPack Studio` for the rating panel and submit flow.

## Notes

Multi-entry conditioning from `split_by_transitions` is meant for this sampler. Connecting it to a normal sampler can mix scene conditionings together instead of routing one scene per chunk.

For manual workflows, use `ConditioningCombine`, not `ConditioningConcat`, when you want multiple scenes. `ConditioningCombine` preserves separate conditioning entries. `ConditioningConcat` merges token tensors into a single conditioning entry, so the sampler correctly sees it as one scene because no scene boundary remains.

Scene order is first in, first out. Written labels like `scene ten`, `scene -999999`, or `scene minus infinity` are treated as transition text only. They do not assign scene numbers.

For character consistency, keep the character or subject description before the first transition. Refiner V2 and Studio prepend that prefix to every detected scene conditioning.

VAE Decode memory use grows with the final stitched latent length. If decode fails with OOM, reduce `max_scenes`, lower `num_frames_per_scene`, decode shorter chains, or use a lower-memory decode path if your workflow provides one.
