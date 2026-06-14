# Changelog

## [Unreleased]

### Added

**Multi-refinement-key support in the Movie Editor.** Refinement keys are now a first-class, per-shortcut training signal. (1) The Engine settings → FunPack Studio card has a **Refinement key** field, so a project can set/wire its own key (previously the key was fixed and unreachable even in full-control mode); it feeds the FunPackRefinementKeyLoader for Studio / Chain Sampler / SaveRefinementLatent. (2) The Shortcuts editor gains a **"Use non-default refinement key"** checkbox + key-name field — firing that shortcut means its key is being trained. (3) **Multi-key per scene:** a scene whose prompt fires several shortcuts bound to different keys now has its conditioning steered by *each* key and **averaged/merged** into one (one key ⇒ substitute the default; none ⇒ default). Rating that scene trains **every** participating key. A key counts for a scene only if one of its shortcuts fired in that scene's text (anchor-bound keys count for every scene); attribution falls back to the safe union if the prompt was rewritten or the split diverges.

**Per-scene refinement-key preview.** The timeline preview now shows, on each scene, the refinement key(s) it will actually steer with before you generate — explicit keys (with `(avg)` when more than one), or the project default key (greyed, `(default)`) when no shortcut key fired in that scene. The preview reuses the exact generation-time resolver (`resolve_scene_refinement_keys`), so what you see matches what runs, including the safe-union fallback.

**Session Reset now wipes every key a run trains, not just the default.** Because per-scene multi-key learning trains *every* refinement key whose shortcut fired in the prompt, a Studio Session Reset now clears all of those keys too (project/default key + each non-default key activated by a shortcut), so no stale per-key state survives a reset (`FunPackVideoRefinerV2._v2_reset_prompt_keys`). The Movie Editor's "Reset Studio session" now confirms first, listing exactly which keys will be wiped: *"This action will reset Studio learning for keys: default, key1, key2… To avoid resetting a non-default key, remove the shortcut that activates it from the prompt. Proceed?"* — disarming a mis-armed reset does not re-prompt.

### Fixed

**Editor timeline dropped a scene when a transition trigger was also a shortcut.** If a scene-cut marker (e.g. `qcut`, `cut`) was *also* a shortcut — common now that cut markers can carry refinement keys (e.g. `qcut` → "cuts" key) — the lossless splitter expanded the trigger word away *before* detecting transitions, so the split was lost or misplaced. Classic symptom: "scene 2 removed from the timeline, scene 1 shows scene 2's prompt" until reload. `split_timeline_verbatim` now also scans the original (verbatim) text for triggers, catching trigger-shortcuts at their true position; genuine shortcut-driven splits (trigger only appears after expansion) and plain triggers are unchanged, and the result still round-trips losslessly.

### Added

**FunPack Cutting Room (Movie Editor)** — a full browser-based montage editor served from ComfyUI at `/funpack/movie`. Build multi-scene projects on a real NLE timeline (ruler, proportional clips, trim, split, drag-to-space pauses, crossfades, per-clip effects), preview the whole sequence in a program monitor with seamless scrubbing, and generate or stitch final video without leaving the UI. Includes a Media bin (upload, filter, sort, rename, export), Characters bible with per-scene assign, global prompt editor with lossless split markers, prompt Shortcuts library, Engine settings (FunPack Studio + LTXAV Scene Chain Sampler panels), Models & Pipeline wiring (built-in LTX pipeline or imported ComfyUI workflow), overlay tracks (text/image compositing with WYSIWYG preview), separated and inserted audio lanes, per-scene Refiner ratings, refinement key import/export, ComfyUI log viewer, git-based FunPack update UI, interactive welcome tour with sandbox demo, and first-run pipeline dependency installer (ComfyUI-Manager + LTXVideo / Video Helper Suite / KJNodes packs).

**Interactive Guessing** — a new Batch Training mode. Instead of varying the seed, it freezes *everything* (including the noise seed) and sweeps the **conditioning's spread** (its "sigma") on video channels along a **linear ramp** across the N rungs — up to amplify toward overbake, down to dampen — so you can see exactly where it breaks. Each rung records its factor. When you rate the ladder, it learns your **safe steering ceiling** for that key and **auto-caps future steering**: a soft pre-limit on absolute/relative steer strength plus a hard clamp on the output conditioning's spread (the guarantee). Audio is never touched (reuses the LTXAV video/audio channel split). The Batch Training tab gains **Mode** (Regular / Interactive Guessing), **Direction**, **Range**, and a **Learning** toggle — Learning off (either mode) means pure generation that teaches nothing. Combined with the new IMAGES output (all batch videos come out the Chain Sampler's IMAGES port), you can now generate N slight variations and just watch them without rating.

**FunPack Normalizing Sampler** — a new `SAMPLER` (selectable in Studio alongside Hybrid Euler 2S / Distilled Flow / KSampler) built for distilled few-step LTXAV at CFG=1. Video-only latent normalization counteracts overbaking / oversaturation / colour-drift; audio stays on plain euler. Node `FunPackNormalizingSampler` + Studio sampler panel.

### Changed

The Chain Sampler's **IMAGES output now returns every batch video** concatenated, not just the last one — for both regular batches and Interactive Guessing.

**Latent normalization stacks on the Hybrid and Distilled samplers too.** The same video-only anti-overbake normalization from the Normalizing sampler is now an opt-in `normalize_strength` / `normalize_start_sigma` on the Hybrid Euler 2S (CONST/RF path) and Distilled Flow samplers. Default off; audio is never touched.

**Distilled Flow `AB2 ramp` — graduated 2nd order.** New opt-in toggle that ramps the AB2 contribution linearly from 0→1 across the schedule. No effect at `order=1`. Exposed on the Distilled Flow sampler node and in Studio's sampler panel.

**FunPack development direction (3.0+).** New work focuses on the Cutting Room frontend and its pipeline integration. Pre-3.0 ComfyUI graph nodes remain supported but will receive **bugfixes only** — no major UI reworks on legacy node popups unless a fix requires it.

### Fixed

**Audio corruption in the FunPack Hybrid Euler 2S and Distilled Flow samplers on LTXAV.** AB2 and Heun are now confined to the video stream; audio rides plain 1st-order euler. Distilled Flow's optional `s_noise` is likewise video-only.

Movie Editor: timeline pause UX (drag clips apart to create pauses, drag flush to remove), preview minibar crash on gap segments, pipeline dependency installer with ComfyUI-Manager bootstrap and cancel, overlay/audio lane stability, multi-scene chain preview offsets, and numerous NLE polish fixes across the dev cycle.

## [2.7.8] - 2026-06-04

### Added

Made all conditioning steering **audio-safe on LTXAV**. LTXAV conditions video and audio from two separate text cross-attentions that the model carves out of one conditioning tensor by splitting its channel dim (`comfy/ldm/lightricks/av_model.py` `_prepare_context`: `torch.split(context, [v_context_dim, a_context_dim], -1)` — leading channels → video `attn2`, trailing → audio `audio_attn2`). Previously every steer (relative/absolute pull, value-function ascent + search, embed_guidance, the attn2 direction patch, and manual Conditioning Adjust) shifted the *whole* tensor, corrupting the audio's own text conditioning and degrading audio. Now a shared `protect_audio_channels` confines every edit to the video channel-slice and restores the audio slice from the unsteered conditioning — effectively "modified conditioning for video, original for audio", with no model patching or extra forward pass. The split is auto-detected from the channel width (7680→3840, 6144→4096) and logs the detected layout once; single-stream LTXV (unrecognised width) is a clean no-op, so video-only models are unaffected. Note: this protects the audio's *direct* text conditioning (the dominant lever); the per-block cross-modal `video_to_audio_attn` can still carry a weaker second-order influence from heavily-steered video.

Added an **Absolute / Relative steer mode** to Refiner V2 and Studio. *Relative* (the default, unchanged behaviour) is per-prompt: it learns and applies the best conditioning for one specific prompt. *Absolute* is prompt-agnostic — it accumulates a single global "taste" prior across **every** rated generation and pulls conditioning toward it regardless of the prompt. In Absolute mode a rating means "this has (or lacks) details I like **in general**", not prompt adherence: a Perfect means "I love this, give me more of it everywhere", and a low rating means "this is missing the details I like in general". The pull layers two engines — a pooled liked/disliked direction (the learned, automated analogue of the manual Conditioning Adjust phrase shift) and the keyless value function on top — and is applied at the conditioning output (`_v2_finalize_conditioning`). *Both* keeps the per-prompt fit and layers the global prior under it; pure *Absolute* bypasses per-prompt memory entirely. The global store learns from rated runs even with no refinement key wired (it is keyless by design). The Scene Chain sampler's `embed_guidance` gains an `embed_guidance_source` (`relative` / `absolute`) so the per-step nudge can draw from the global taste direction too. Studio exposes `Steer mode` + `Absolute strength` in the refiner panel.

Ported **velocity bias + reactive rescue** to the `FunPack Distilled Flow` sampler (`sample_funpack_distilled_flow`), reusing the same capture/memory/rescue machinery as the Hybrid Euler-2S and LTXAV/RF samplers. The sampler now exposes `velocity_bias_mode`, `velocity_bias_strength`, `velocity_bias_source`, `velocity_refinement_key`, `rescue_mode`, `rescue_threshold`, and `rescue_strength`, and these are wired into Studio's **Distilled Flow** sampler config panel (with the same blank/`default` velocity-key → wired-refinement-key fallback the Hybrid panel uses). Steering is applied magnitude-preservingly and, on packed LTXAV latents, confined to the video stream (audio-safe). Note: few-step distilled schedules may only land on one or two velocity targets, so apply/rescue fire less often than on an 8-step run; they no-op cleanly when no target matches. With everything off the sampler is byte-identical to the previous deterministic ODE.

## [2.7.7] - 2026-06-03

### Added

Added **Batch Training** — a controlled-batch RLHF workflow built around the principle that the cleanest learning signal comes from rating several generations that differ in exactly one thing. When a batch is active, the Scene Chain sampler runs the chain `N` times with everything frozen except the noise seed, producing `N` directly-comparable videos. Studio then shows a rating panel where every variant is scored, and on submit the value function trains on the batch's shared (frozen) conditioning with each variant's reward — same conditioning, N rewards, which is the variance-reduced comparison signal. Spans the full pipeline: the engine (controlled N-run mode on the sampler), a Studio variant producer that packs the variants into conditioning, the `/funpack/batch` server routes and rating window, and the value-function intake on submit. Phase 3b adds deeper learning from batch ratings via axis and direction memory. Batches live in ComfyUI's temp directory and are wiped on restart.

Added a **reactive in-flight rescue** system to the Hybrid Euler-2S sampler. During sampling the trajectory is compared against conditioning-clustered memory of past good and bad runs; when a step drifts toward a known-bad trajectory it is nudged back. Rescue is rating-gated (separate good/bad trajectory banks, learned only from rated runs), prompt-aware (clustered by conditioning so the right memory is consulted), runs in both sampler phases, and persists its trajectory banks to disk. The full feature set was also ported to the LTXAV rectified-flow (CONST-model) path via a sampler mirror, so velocity capture and rescue now work for LTXV/LTXAV, not just the discrete-step models.

Added **Monte Carlo conditioning search** after the value-function gradient ascent, plus a VF final gate, so the conditioning chosen for generation is the best of a sampled neighbourhood rather than the raw ascent endpoint.

Added a Studio **Variability macro** and an **active-feature readout** for the Hybrid sampler, summarising which steering features (velocity bias, rescue, embed guidance, value guidance) are live for the current configuration.

### Changed

Unlocked `velocity_bias_strength` from a `0.35` cap to `3.0` for deliberate creative action injection. Velocity bias is an artistic / action-injection control (and carries an emergent scene-cut prior — cuts survive the trajectory mean and get reintroduced unprompted); it is not a consistency tool.

Refiner V2 now **always trains the value function**; `value_guidance` only gates whether the learned reward is *applied* during sampling, and defaults on. This means rated runs keep teaching the value function even when guidance is off.

Decoupled `eta_final` decay from the quality boundary — it now anchors to schedule progress instead, so ancestral-noise decay tracks the sampling timeline rather than a quality threshold.

### Fixed

Audio-safe LTXAV sampling: ancestral noise and trajectory steering are now applied to the video latent only, never the audio latent, preventing the joint-attention audio corruption that steering on the combined tensor caused.

Velocity-bias anti-softening: the bias is now applied magnitude-preservingly with sigma-decay and a quality-sharpness term, and sourced from the nearest trajectory cluster, so it injects motion without washing out detail.

Audit fixes: corrected a velocity commit key mismatch, sparse rescue targets, and a redundant aspect-bucket computation. Batch Training fixes: distinct per-variant seeds in both the split-scene and single-scene paths (identical seeds were producing identical videos), correct activation alongside `split_by_transitions`, the node rating is ignored while a batch is in progress, and the in-node panel was de-duplicated from the Studio Refiner tab.

## [2.7.6] - 2026-05-30

### Added

Added **pre-generation conditioning ascent** driven by the value function. Before sampling, the positive conditioning is moved by gradient ascent toward higher predicted reward, in both the single and scene-split Refiner V2 output paths. Displacement is capped to prevent reward hacking (the value function will otherwise push conditioning into degenerate high-score regions). The value function can be exported and imported from Studio and is cleared on session reset.

Added **VF-driven conditioning shaping**: confidence scaling and gradient-aligned phrase boosting in conditioning memory, attention-weight accumulation, and VF-driven temperature, so the learned reward influences which phrases and attention weights are emphasised.

Added **concept-in-context conditioning guidance** and **concept-pair bad-direction repulsion** — conditioning is nudged toward liked concepts in context and away from concept pairs learned to be bad.

Added a **motion floor** to embed guidance: when temporal variance falls below a threshold the guidance auto-boosts it, fighting static/frozen output. Activation is logged with step, sigma, and variance ratio.

Reworked the rating UI into a **rating picker popup**. Added a `Wrong` action and an explicit quality rating, a `Nailed it` rating (prompt-adherence positive, weaker than `Perfect`), and replaced the standalone `Loved it` rating with a per-option **heart modifier** (an axis-blind quality endorsement layered on top of any rating). The heart is disabled for quality-degraded ratings and for `Awful`.

### Changed

Disabled `perfect_freeze` — prompt changes made after a `Perfect` rating are now respected instead of the conditioning being frozen.

### Removed

Removed the experimental latent value function (VGG-Flow-style per-step latent steering). It was added in this cycle but pulled: the LTX token format does not expose the latent in a form the per-step steering could act on.

### Fixed

Fixed an `UnboundLocalError` (`eff_key` not yet assigned when the value function loads), value-function loading needing `inference_mode(False)`, several rating-picker bugs (stale outside-click listener re-opening the picker, active state not updating on option change, clicks landing on a canvas button widget rather than a DOM element), and `Save Refinement Latent` not executing.

## [2.7.5] - 2026-05-29

### Added

Added **embed guidance** — a per-step nudge of the conditioning toward the learned "liked-quality" direction during sampling. Near-free overhead; requires a wired `refinement_key_input` with enough liked generations to have formed a direction. Exposed as `embed_guidance` / `embed_guidance_strength` on the Scene Chain sampler.

Added an **online value function** for reward-guided sampling — a small MLP trained on rated generations that predicts reward and can steer sampling toward it. Required several fixes to train and run gradients inside ComfyUI's `inference_mode` execution context.

Added **mid-scene guide** (`mid_scene_guide` / `mid_scene_guide_strength`), replacing the broken `self_consistency` feature. It uses the LTX guide-attention mechanism rather than post-block hidden-state injection (which corrupted audio through joint attention). At ~`0.25`–`0.3` strength it preserves static-element layout across a scene; capped at `0.5`.

Added a **vision-conditioning toggle** to the Studio popup, and moved reference-image conditioning into Studio, where the source image already lives.

### Changed

Removed the predefined transition-phrase list — scene splitting is now fully driven by user-defined and auto-detected transitions.

### Removed

Removed the **per-scene vision re-encoding and `clip` input** from the Scene Chain sampler, and reverted the per-scene reference-image encoding in Studio (both added in 2.7.4). They were unstable in practice.

Removed `self_consistency` (corrupted audio via joint attention — superseded by `mid_scene_guide`), the `i2i_scene_cut` feature, and dead guide-keyframe / guide-conditioning code paths.

### Fixed

Fixed `i2i_strength` inversion (higher now means stronger reference influence) and replaced the 1-frame i2i anchor with 2-frame i2v anchor generation for hard cuts.

## [2.7.4] - 2026-05-28

### Added

Added **K/V in-context conditioning** for LTXAV identity blocks during i2v generation. Reference hidden states are captured at the start of each scene's denoising pass and prepended as extra attention tokens to the identity-formation blocks ([14, 20, 21, 30, 33]). This forces the model to attend to the reference character's appearance during every self-attention step in those blocks. Result: strong character consistency across scene cuts, view changes, and orientation changes with no LoRAs required.

Added **Gemma3 vision prompting**. When `source_image` is connected to Studio and the CLIP was loaded with a Gemma3-12B checkpoint via `DualCLIPLoader`, Studio automatically encodes the reference image through the built-in SigLIP vision encoder and feeds the resulting vision tokens into the text conditioning. This means the model conditions on both the prompt text and the actual pixel content of the reference frame. No extra node is required — `DualCLIPLoader` already loads the vision weights when present in the checkpoint.

Added **per-scene vision re-encoding** to `FunPack LTXAV Scene Chain Sampler`. Connect an optional `clip` input. After each scene is sampled, the next scene's conditioning is re-encoded using the previous scene's decoded last frame as vision context. This gives identical scene texts genuinely different conditioning based on runtime-generated content, so the model knows what state it came from when building the next scene.

Added **duplicate scene text differentiation**. When two or more scenes share identical text, the second and subsequent occurrences are encoded with a `"Returning to an earlier scene: "` prefix. The original text is preserved in metadata for logging. This breaks the shared conditioning cache entry so the model receives distinct input for each occurrence.

### Improved

Reworked `frame_overlap=0` soft continuation: the previous scene's last **4 frames** are now prepended with **mask 0.4** (partial denoising) instead of 1 frame at mask 0.0 (fully pinned). The model receives temporal context from the previous scene while retaining enough denoising freedom to commit to orientation and pose changes directed by the text prompt.

### Fixed

Fixed anchor text punctuation when joining scene segments. If the character description ends with `.`, `!`, `?`, or `,`, a plain space is used as the separator instead of injecting a redundant `, `. Previously a description ending with a full stop produced `"description., scene text"`.

### Removed

Removed `FunPackGemmaVision` node. It was redundant: `DualCLIPLoader` with a Gemma3-12B checkpoint already loads `vision_model` and `multi_modal_projector` weights through the normal `load_state_dict` path, so the manual weight injection the node performed was a second load of the same file. The vision capability check now detects the attributes directly instead of relying on an injected flag.

## [2.7.3] - 2026-05-27

### Fixed

Fixed `carry_i2v_guides` soft continuation when `frame_overlap=0`: the anchor mask is now fully pinned (0.0) to prevent denoising from disturbing guide tokens.

### Notes

Using `frame_overlap=0` together with `carry_i2v_guides=True` is confirmed to produce bad results and is not recommended. Both parameters now warn about this in their tooltips and in the sampler documentation. Use this combination only for deliberate testing.

## [2.7.2] - 2026-05-22

### Added

Added **Shortcuts** system. Activation phrases in the prompt are replaced with full cinematic descriptions before encoding. Multiple replacement options are randomly picked per seed. Empty replacement removes the matched phrase entirely. Longer phrases always win over shorter overlapping triggers. Managed in the new Studio Shortcuts tab with Add/Save/Delete/Import/Export.

Added **Transitions** system. User-defined transition phrases extend the built-in split list. Custom entries support a placement override - `start` (transition opens the new segment), `end` (transition closes the previous segment), or `silent` (split happens but the phrase is stripped from output entirely). Managed in the new Studio Transitions tab.

Added global **Transition placement** setting in Studio Refiner tab (`start` / `end` / `silent`), with per-entry override on each custom transition entry.

### Fixed

Removed all single-word temporal markers (`next`, `suddenly`, `later`, `finally`, etc.) from the built-in transition phrase list. They caused false splits on normal prose.

Fixed `_GENERIC_SCENE_LABEL_PATTERN` matching "scene proceeds", "scene features", "scene shows" and similar noun-verb constructions as scene labels.

Fixed dangling trailing transition segments (prompt ending with "...cuts to the") being kept as near-empty scenes.

Fixed stray comma artifacts after article words when transition phrase merging occurs.

Fixed custom transition triggers ending with punctuation (e.g. "Scene cut.") not being detected due to a misplaced word boundary.

## [2.7.1] - 2026-05-21

### Added

Added successful seed memory for `FunPack Studio`. When the `seed` output is connected, `Perfect` and `Loved it` ratings store the previous run's sampler seed under the active refinement key. Future runs occasionally reuse concept-matched successful seeds while keeping normal fresh seeds as the default path.

Added per-scene seed metadata for prompt-split mode. Studio and Refiner V2 keep the public `seed` socket as a single integer, while each detected scene conditioning entry can carry its own `funpack_scene_seed` for the Scene Chain sampler.

Added `use_same_seed` to `FunPack LTXAV Scene Chain Sampler`. When enabled, every scene uses the first provided scene seed or the base seed. When disabled, each scene uses scene seed metadata or falls back to `seed + scene_index`.

### Changed

Made Scene Chain i2v guide carry opt-in. The default path no longer appends protected i2v frames as hidden guide tokens, preserving cleaner scene cuts unless the experimental option is deliberately enabled.

### Fixed

Fixed compact i2v guide masks failing to concatenate with full spatial Scene Chain masks by broadcasting guide masks to the chunk mask shape.

Updated README release notes and added the Intent section.

## [2.7.0] - 2026-05-21

### Added

Added `FunPack LTXAV Scene Chain Sampler` for split-scene LTXV/LTXAV continuation in one ComfyUI run. It consumes multi-entry positive conditioning from `FunPack Studio` or `FunPack Video Refiner V2`, samples one scene chunk per conditioning entry, increments the seed per scene, preserves overlap from the previous chunk, and blends/appends chunks in latent space.

Added support for plain LTXV video latents and nested LTXAV video/audio latents in the Scene Chain sampler. For nested AV latents, video and audio tensors are continued together, with audio overlap derived from the video/audio latent length ratio.

Added broad order-only scene splitting for `split_by_transitions`. Scene labels such as `scene ten`, `scene -999999`, and `scene minus infinity` are transition cues, but their written labels never affect scene numbering or order.

Expanded transition phrase detection with scene progression, camera shift, zoom, final shot, and final transition phrases.

### Changed

`split_by_transitions=True` now returns one conditioning entry per detected scene through the existing `modified_positive` output. No new Refiner V2 or Studio output sockets were added.

The text before the first transition is treated as a shared character/global anchor and prepended to every detected scene conditioning. This is intended to improve character consistency across generated chunks.

Removed the hard 8-scene cap from Refiner V2 split output and Scene Chain sampler execution. `max_scenes` still defaults to `8`, but users can raise it for longer chains.

Standalone `then` is no longer a transition trigger. More specific phrases such as `and then`, camera transitions, scene labels, and explicit cut/transition language still split scenes.

### Fixed

Fixed Studio's Scene Builder mode dropdown not refreshing the active tab immediately after selecting a new mode.

### Warning

`FunPack LTXAV Scene Chain Sampler` is resource heavy. Long chains create large final latents and may run out of memory during VAE Decode even when sampling succeeds. Start with short scenes and a modest `max_scenes`, then increase carefully.

## [2.6.0] - 2026-05-16

### Added

Added `FunPack Studio` - a single node that replaces the typical chain of Refinement Key Loader, Scene Builder, Apply LoRA Weights, LoRA Loader, Video Refiner V2, and Conditioning Adjust with a tabbed popup editor. All settings are managed inside the popup; only the rating widget and Open Studio button are visible on the node face.

Studio inputs (in order): model, clip, advisor_clip, positive_conditioning, negative_conditioning, clip_vision_output, source_image, lora_stack, positive_prompt, negative_prompt, user_intent_prompt, feedback_prompt, refinement_key_input.

Studio outputs (in order): model (LoRAs applied + attn2 direction patch), modified_positive, negative (encoded from negative_prompt or passed through from negative_conditioning), seed (for wiring to sampler), high_pass_sampler, high_pass_sigmas, low_pass_sampler, low_pass_sigmas, loss_graph, status, training_info, encoded_prompts.

Studio popup tabs:
- **Session**: refinement key management and Scene Builder mode selector (Pass-through / Manual / Auto / Learning).
- **Scene**: scene preset load and save, phrase bank from session memory, positive prompt composer.
- **Refiner**: all Refiner V2 settings including negative prompt field, feedback, and intent override. Shows a banner and disables the intent field when Scene Builder is active.
- **Advisor**: enable/configure an internal HuggingFace CausalLM advisor. Uses the same model cache as the standalone Advisor LLM node.
- **LoRA**: full LoRA pipeline - session weight suggestions are read first, then LoRAs are applied to model and CLIP, then the direction patch is applied on top. Supports model type (ltx2/wan) and per-block settings.
- **Sampler**: configure Hybrid Euler 2S, Distilled Flow, or any KSampler for the high-pass and low-pass outputs independently. Sigma schedules entered as comma-separated floats.
- **Adjustments**: phrase-level conditioning adjustments with session phrase bank.

Three text inputs (refinement_key, feedback_prompt, user_intent_prompt) have override toggles: when off, connected inputs win; when on, popup values win.

The popup remembers its last active tab per node via localStorage. All field changes auto-save to the node widget after 600ms, so settings survive page refresh without requiring Close to be clicked.

Added `negative_prompt` encoding to Studio: when no pre-encoded `negative_conditioning` is connected, Studio encodes the `negative_prompt` text via CLIP internally, removing the need for an external CLIPTextEncode node.

Added `/funpack/available_loras` and `/funpack/phrase_memory` backend endpoints used by Studio's LoRA picker and phrase banks.

Added `FunPackConditioningAdjust` standalone node for phrase-level conditioning adjustments. Encodes each phrase via CLIP, computes a unit-norm direction from the base conditioning, and applies it at user-set strength. Positive pushes toward the phrase, negative pushes away. Popup editor with session phrase bank.

Added `seed` output to `FunPack Video Refiner V2` (via optional `_seed` parameter) so Studio can generate a seed and expose it as an output for wiring to samplers.

### Fixed

Fixed `FunPackAdvisorLLM` tokenize failing with `AttributeError` when `apply_chat_template` returns a `BatchEncoding` instead of a plain tensor in newer transformers versions. Explicit `hasattr(result, "input_ids")` check now handles both return types.

Fixed advisor generation producing no output for Qwen3 and other chain-of-thought models: thinking tokens (`<think>...</think>`) were not being stripped because they are special token IDs that disappear with `skip_special_tokens=True`. Switched to decoding with `skip_special_tokens=False` then stripping thinking blocks with regex. Truncated thinking blocks (no closing tag, token budget exhausted) are also stripped.

Fixed `FunPackAdvisorLLM` attention mask warning and erratic output (echoed prompts, missing spaces) caused by `pad_token == eos_token` without an explicit mask. Now passes `torch.ones_like(input_ids)` as attention mask to all generate calls.

Fixed `FunPackConditioningAdjust` adjustments not applying for LTX/Gemma3 conditioning: the node was reading `pooled_output` which is `None` for T5-based encoders. Now uses `conditioning.mean(dim=(0,1))` on the sequence tensor, matching how V2 handles conditioning internally.

Fixed Refiner V2 advisor diagnostic not being generated in Full mode: the LLM analysis prompt was asking for session-wide pattern recognition when no history existed yet. Now adapts - uses a simple per-run analysis on early sessions and session pattern analysis when history exists. Added a rule-based fallback so diagnostic history always accumulates even when the LLM produces empty output.

Fixed `perfect_repair_phrases` and `_v2_emphasized_prompt` injecting phrases regardless of the `prompt_repair` toggle. Both are now gated behind `prompt_repair=False`.

Removed the Perfect-rating advisor gate. The advisor previously skipped both analysis and repair when rating was Perfect and no text feedback was provided. Perfect is not a ceiling - the advisor now runs normally for Perfect ratings.

## [2.5.3] - 2026-05-16

### Fixed

Fixed `FunPackAdvisorLLM` wrapper not triggering advisor generation. Qwen3 and other chain-of-thought models emit `<think>...</think>` blocks that were not stripped, causing the parsed repaired prompt to contain reasoning text. This made body-similarity validation reject the result as "too far from intent." Fixed by stripping thinking blocks in `decode`. Also added `enable_thinking` kwarg support in `tokenize` (with TypeError fallback for models that don't support it) and expanded `max_new_tokens` by 2048 when thinking mode is active so the reasoning budget does not crowd out the actual response. Fixed `_v2_text_semantic_similarity` returning 0.0 for generation-only clips that have no `encode_from_tokens_scheduled` - these now return 1.0 (skip the semantic gate) instead of causing spurious rejections. Fixed `FunPackAdvisorLLM` missing from the standalone import block in `__init__.py`, which broke the test suite.

Removed Perfect-rating advisor gate. The advisor was silently skipping both the analysis pass and the repair pass whenever the rating was Perfect and no text feedback was provided, even when the user had an active advisor mode. Perfect is not a ceiling - if the user provides `feedback_prompt`, it must be honored regardless of rating. The only remaining guard is the `allow_prompt_change` check (Learning mode).

## [2.5.1] - 2026-05-16

### Added

Added `FunPackAdvisorLLM` node. Loads any HuggingFace CausalLM (including sharded checkpoints) as an advisor for Refiner V2. Connect the output to `advisor_clip`. Model is cached after first load so subsequent runs do not reload. Also fully compatible with the built-in `TextGenerate` and `TextGenerateLTX2Prompt` nodes - supports `skip_template`, `min_p`, `presence_penalty`, and progressive fallback for unsupported generation parameters.

Added `_v2_direction_readout` to training_info Adaptation section. Shows each direction memory slot in plain language: run count, magnitude, whether it is in direction mode or lerp fallback, and the role each axis is playing this run.

### Changed

Advisor prompt format rewritten to natural language. Both the repair and analysis user messages now read as plain enhancement requests rather than structured field-value pairs. Works with enhancement-type models (Sulphur, Qwen prompt enhancers) as well as instruction-following models. System prompt reduced to one sentence.

Direction-based conditioning now uses `max_new_tokens` instead of `max_length` in the `FunPackAdvisorLLM` wrapper so prompt length does not eat into the generation budget.

Model patch status expanded to show which direction slots are active with run counts and which phrase texts are being emphasized in cross-attention.

Adaptation status block rewritten to multi-line readable format showing strength, reward trend, streak, per-slot mode (direction vs lerp fallback), and axis adjustments applied this run.

### Fixed

Fixed `_v2_generate_advisor_text` returning `None` when layer 1 tokenization succeeded - `generate`/`decode`/`return` were inside the `except TypeError` block so they only ran when layer 1 failed.

Fixed session reset not clearing `intent_expansion_memory`, `session_source_mean_count`, `liked_dir`, and `bad_dir` - these fields were missing from `_v2_empty_state` and survived reset via `setdefault`.

Fixed advisor repetition loops: `repetition_penalty` raised from 1.05 to 1.3, added `no_repeat_ngram_size=5`, temperature raised from 0.5 to 0.7.

Fixed system prompt bleeding into advisor output by splitting prompts into `(system, user)` tuples and applying the model's native chat template. Three-layer fallback: native `system_prompt` kwarg, manual `apply_chat_template` via BFS, flat string with completion anchor.

Added persistent cross-run encode cache (`_V2_PERSISTENT_ENCODE_CACHE`, 4096 entry cap) so phrase encodings are not recomputed every run when CLIP and text are unchanged.

## [2.5.0] - 2026-05-15

### Added

Added a CLIP text-generation advisor to `FunPack Video Refiner V2`. The advisor runs two sequential passes: an analysis pass that identifies what specifically needs to change in the suggested prompt, followed by a repair pass that applies those findings. Both passes see the current suggested prompt, user intent, previous prompt, memory suggestions, and the full feedback history.

Added `advisor_clip` input to use a separate generative CLIP/Gemma model for the advisor while the main `clip` continues handling encoding and similarity checks.

Added `advisor_mode` dropdown: `Off`, `Only diagnostics`, `Only prompt`, `Full`. In `Full` mode both analysis and repair passes run. In `Only diagnostics` mode only the analysis pass runs and its finding is stored in feedback history for the next run. In `Only prompt` mode only the repair pass runs silently.

Added `feedback_prompt` optional input. When connected, the user's natural-language description of what was wrong is placed first in both advisor passes, with the system instructed to follow it exactly and override all other repair logic.

Added persistent `advisor_feedback_history` stored in V2 session state. Up to ten past feedback entries accumulate across runs, each labelled with the corresponding rating: `Missing action: he was supposed to hold her hand not her head`. Advisor-generated diagnostics from `Only diagnostics` runs are stored as `Advisor note:` entries so they carry forward into subsequent `Full` runs.

Added `Prompt only` execution mode to `FunPack Video Refiner V2`. All prompt shaping runs as normal but conditioning vectors are passed through unchanged. Learning still applies. Useful when conditioning adaptation should be paused while prompt refinement continues.

Added `prompt_repair` boolean input to `FunPack Video Refiner V2` (default on). Turning it off disables the rule-based phrase injection from phrase memory and passes no repair candidates to the advisor. Useful early in a session before enough context has been built.

Added `encoded_prompts` STRING output to `FunPack Video Refiner V2`. When the advisor ran and produced a suggestion, the output includes up to four labelled sections: `Positive prompt` (what was encoded), `Advisor suggestion (applied/rejected)` (what the advisor generated), `Advisor analysis` (the diagnostic text from the analysis pass), and `Pre-advisor prompt` (the prompt before the advisor rewrote it).

Added `eta_final` parameter to `FunPack Hybrid Euler 2S Sampler`. When set below `eta`, ancestral noise strength decays linearly toward this value as sigma approaches the quality phase boundary, smoothing the transition into deterministic refinement. Default `1.0` preserves existing behaviour.

### Changed

Replaced the Refiner V2 advisor system prompt with a structured repair format. The advisor now receives four explicit variables — `ORIGINAL_USER_INTENT`, `LAST_PROMPT`, `RATING`, and `OPTIONAL_NOTE` — and is instructed to rewrite `LAST_PROMPT` to fix the specific failure described by the rating. Memory suggestions, feedback history, and analysis context are folded into `OPTIONAL_NOTE`. The repair pass for `Only prompt` mode outputs a plain prompt string with no labels, matching the instruction to output only the final text.

Removed `negative_prompt` input and `modified_negative` conditioning output from `FunPack Video Refiner V2`. Negative conditioning has no effect at CFG=1.0 with NAG guidance and added a redundant AI generation call in every mode. Both the rule-based negative repair and the negative advisor pass are removed.

Increased advisor token budget: repair pass 800 → 1600 tokens, analysis pass fixed at 1200 tokens (was `repair // 2 = 400`). The analysis limit is now independent of the repair limit so it does not shrink if the repair budget changes.

`FunPack Hybrid Euler 2S Sampler` early phase now uses an order-2 ancestral denoised extrapolation (Adams-Bashforth 2-step) in addition to the existing Euler-A update. The previous step's denoised estimate is used to extrapolate a better score direction at zero extra model-call cost. The state resets after any motion pulse.

`FunPack Hybrid Euler 2S Sampler` quality phase now uses a progressive `correction_blend`: the first half of quality steps use a single-eval Euler ODE pass; the second half use the configured 2S correction. This reduces model calls in the quality phase while concentrating the expensive correction where sigma is lowest and it has the most impact.

### Fixed

Fixed `Only prompt` advisor mode running two AI generation calls per invocation (one for positive repair, one for the now-removed negative advisor), causing each run to take twice as long.

Fixed `encoded_prompts` always showing only `Positive prompt:` regardless of advisor activity. The final return path was calling `_v2_encoded_prompts_output` without the advisor keyword arguments.

Fixed Refiner V2 advisor generation: `do_sample` was `False`, forcing greedy decoding and silently ignoring temperature, top_k, top_p, and all sampling parameters. The model was always producing its highest-probability default output regardless of instructions or feedback. Changed to `do_sample=True` with temperature 0.5.

Fixed advisor validation silently rejecting valid feedback-driven repairs: the intent-distance check and protected-category checks now bypass when `feedback_prompt` is connected, allowing the advisor to implement what the user explicitly requested.

Fixed `_v2_find_perfect_example_for_intent` accessing a field (`loved_delta_sources`) that was never written. It now correctly reads from `perfect_anchors` and `loved_variants`.

Fixed `_v2_update_streaks` updating conditioning strength signals (`avg_reward_ema`, `good_streak`, `bad_streak`) in `Prompt only` mode, which contaminated conditioning adaptation for subsequent `Refine` runs. Rating and axis labels still update for repair continuity.

Fixed advisor rating label on first run or session reset: was forwarding the user's rating widget value even when there was no previous output to apply it to. Now passes `"No previous output (first run or session reset)"` when `has_previous_run` is false.

## [2.4.2] - 2026-05-15

### Added

Added `Learning` mode to `FunPack Video Refiner V2`. It still observes prompts, conditioning, ratings, phrase memory, and diagnostics, but passes positive and negative prompt conditioning through without prompt repair, Lucky composition, wildcard cleanup, or conditioning-vector adaptation.

### Fixed

Fixed `FunPack Scene Builder` mode handling so the live Mode widget stays independent from the selected saved scene, including queue-time `Learning` and `Auto` behavior.

Fixed `FunPack Scene Builder` rich prompt editing so the caret can move past a final inline phrase chip with the mouse or right arrow key.

## [2.4.1] - 2026-05-14

### Fixed

Improved `FunPack Scene Builder` database rows so long words and phrases show their full text as a hover hint, and double-click editing opens a wider multiline field with explicit OK/Cancel buttons.

## [2.4.0] - 2026-05-14

### Added

Added `FunPack Scene Builder`, a scene preset node that replaces `FunPack Template Manager`. It collects universal prompt phrase memory, lets users manually assign positive/negative scene phrases, passes the current LoRA stack through unchanged, and can auto-apply a saved scene from an intent prompt match.

Simplified `FunPack Scene Builder` so prompt and intent text are connection-only inputs, removed model-mode and per-block controls, and outputs only scene prompt data plus the pass-through FunPack LoRA stack instead of conditioning.

Added `Learning` mode to `FunPack Scene Builder`; it records connected prompt phrases into the selected refinement key's scene memory while passing positive prompt, negative prompt, and LoRA stack through unchanged. Refiner reset clears conditioning-delta learning while preserving the refinement key's Scene Builder memory.

Redesigned `FunPack Scene Builder` as a compact button-driven node with centered editor menus for scene name, mode, aliases, Positive Prompt, Negative Prompt, and Database controls. First use asks for a scene name before editing, connected prompts now teach useful words as well as phrase chunks, the editor refreshes the selected refinement-key database before opening, prompt editors highlight already-used chips, database words can be double-clicked for inline editing, and wildcard random choice is now a checkbox for adjacent entries instead of a text group.

Added searchable LoRA picking to `FunPack Apply LoRA Weights`. The compact row UI remains the primary workflow, and saved workflows still serialize through the existing `lora_list` JSON value.

Added optional `clip_vision_output`, `source_image`, and `negative_prompt` inputs to `FunPack Video Refiner V2`.

Added a final `modified_negative` conditioning output to `FunPack Video Refiner V2`. When negative repair has prompt text to encode and `CLIP` is connected, the node returns repaired negative conditioning; otherwise it returns an empty conditioning list.

Added advisory V2 vision context storage for source image dimensions, aspect ratio bucket, image fingerprint, CLIP Vision tensor summaries, and changed-image detection. Vision context is diagnostic only and is not blended into positive conditioning.

Added experimental early velocity bias capture/application controls to `FunPack Hybrid Euler 2S Sampler`, defaulting off.

### Changed

Removed public registration for `FunPack Template Manager`. Use `FunPack Scene Builder` for new scene/preset workflows.

Updated V2 prompt repair so repaired phrases preserve stopwords and phrase text while still using filtered semantic tokens for matching and categorization.

Reduced repeated Refiner V2 CLIP model calls by caching category and phrase encodes within each run.

Updated negative repair to persist poorly rated or wrong-context tags and append them to future negative prompts before encoding negative conditioning.

## [2.3.3] - 2026-05-08

### Fixed

Fixed Refiner V2 so `CLIP` and pre-encoded `positive_conditioning` can both be optional inputs. When `CLIP` is connected, V2 keeps owning prompt encoding as before. When `CLIP` is not connected but `positive_conditioning` is connected, V2 accepts the finished Gemma3/LTX2 conditioning, uses the prompt for analysis, and loads only the Gemma3 tokenizer.

## [2.3.2] - 2026-05-08

### Added

Added Refiner V2 original-intent alignment memory. When `user_intent_prompt` stays the same but an enhancer produces different `positive_prompt` variants, the refiner now remembers intent-enhance pairs, which variants rated well, which original-intent phrases were omitted, and which enhancer-only phrases were rejected.

### Fixed

Fixed Refiner V2 so learned original-intent omissions can be restored on later runs, while repeatedly rejected enhancer-only additions can be removed before encoding. Rejected enhancer-only full words and adjacent word pairs are stored as omit evidence for that original intent.

## [2.3.1] - 2026-05-08

### Fixed

Fixed Refiner V2 Prompt Repair so missing/wrong ratings only repair from the current prompt or explicit user intent, instead of pulling unrelated learned favorite actions, details, quality cues, camera moves, or styles from memory.

Fixed Prompt Repair memory matching so the same word with different neighboring prompt context is treated as different evidence.

Fixed vague raw user intent handling so prompts like `Figure it out` let the enhanced `positive_prompt` drive repair matching when available.

## [2.3.0] - 2026-05-08

### Added

Added `Wrong appearance` rating to `FunPack Video Refiner V2` for outputs contaminated by remembered clothing, character, subject, or background concepts.

Added `FunPack Refinement Key Loader`, with a selectable key dropdown, create-on-load behavior, and browser-side JSON import/export buttons.

Added a Discord-friendly Refiner V2 quick guide for new users.

### Changed

Updated Refiner V2 Prompt Repair so it only auto-adds safe repair concepts such as action, camera, details, quality, and style. Appearance, subject/character, and environment/background concepts are now blocked from Prompt Repair.

Updated `I'm Feeling Lucky` in Refiner V2 so appearance, subject/character, and environment/background memory is not auto-injected unless the user explicitly includes that phrase in the current prompt.

Updated legacy Void/Lucky token-bank selection to skip appearance, subject/character, and environment/background tokens.

Updated Refiner V2 and `FunPack Apply LoRA Weights` so both can accept a linked refinement key from `FunPack Refinement Key Loader`.

### Fixed

Fixed appearance bleed-over where highly liked clothing or character tags could reappear in unrelated image-to-video prompts.

## [2.2.1] - 2026-05-07

### Fixed

Fixed `FunPack Video Refiner V2` prompt phrase categorization so environment and appearance descriptions are not pulled into action learning by generic `-ing` or `-ed` words.

Updated Refiner V2 category similarity blending so CLIP category comparisons only help uncertain phrases instead of overriding strong local action, camera, appearance, environment, quality, or detail anchors.

Fixed `FunPack Video Refiner V2` so prompt-enhancer refusal text like "I'm sorry, I cannot help..." is passed through without being saved into prompt history, phrase memory, or future learning targets.

Improved `FunPack Video Refiner V2` training data output with clearer sections and extra line breaks for run state, learning, prompt analysis, adaptation, guidance, and LoRA diagnostics.

Updated `FunPack Video Refiner V2` to remember liked action/detail phrase clusters with their neighbors and use those ordered clusters before weaker ngram or token memory when repairing missing axes.

Added `Wrong details`, `Wrong action`, and `Wrong details + action` ratings for good-looking videos that do not match the requested intent; these preserve satisfied quality/composition signals while marking the mismatched action/detail context for repair.

## [2.2.0] - 2026-05-07

### Added

Added `FunPack Video Refiner V2`, a simplified prompt-owned refiner that accepts `positive_prompt` and a connected `CLIP`, owns prompt encoding internally, learns from ratings, and returns refined positive conditioning plus diagnostics.

Added `FunPack Template Manager`, a preset node for storing prompts, activation words, refinement keys, sigma schedules, and FunPack LoRA stacks with import/export support.

Added `I'm Feeling Lucky` mode to `FunPack Video Refiner V2`. It works as a preference composer that can inject learned user-preferred actions, camera moves, details, and styles even when the current prompt is vague.

### Changed

Updated LTX per-block LoRA loading so supported stacks now compare LoRA block fingerprints across the whole stack and apply type-aware conflict balancing before patches are loaded.

Fixed `FunPack Hybrid Euler 2S Sampler` restart timing so `restart_trigger_pct` is respected across the full sigma schedule instead of being clamped to the Euler-to-2S quality transition.

Improved `FunPack LoRA Loader` rerun performance by caching recently used raw LoRA files, model-mapped LoRA patches, and per-block fingerprint analysis.

Reworked `FunPack Video Refiner V2` ratings around explicit missing-axis signals: `Perfect`, single missing axes, paired missing axes, and `Awful`.

Removed the Refiner V2 `mode` input. V2 now accepts whatever connected `CLIP` the workflow provides and stores state in a CLIP-owned namespace.

Renamed visible Refiner and LoRA intent from `concept` to `action`. Old `Missing concept` ratings and old `concept` LoRA rows are still accepted as compatibility aliases, but V2 stores and displays `action`.

Updated `I'm Feeling Lucky` in Refiner V2 so Lucky only composes prompt text when enabled. When disabled, it may train memory from rated runs but does not compose or alter output.

### Fixed

Fixed `I'm Feeling Lucky` token-bank learning for changing prompt/conditioning workflows by falling back to prompt-order token placement when exact tokenizer position matching cannot find enough words.

Fixed `I'm Feeling Lucky` rating attribution for changing prompts so ratings update the previous prompt's learned tokens while the current prompt seeds new neutral discovery tokens.

Updated `I'm Feeling Lucky` filtering to learn poor adjacent token pairs instead of refusing individual tokens outright.

Updated `I'm Feeling Lucky` with uncapped token, pair, and context memory so it can learn which concepts belong together and call strong missing neighbors when prompt anchors are present.

Fixed `I'm Feeling Lucky` composition order so the current generation uses already-learned memory first, then seeds current prompt tokens for future runs.

Fixed `I'm Feeling Lucky` memory-first output so vague or empty incoming conditioning can use the longest compatible learned conditioning canvas instead of being limited to the current prompt's shape/content.

Added an optional `clip` input to `FunPack Video Refiner` so `I'm Feeling Lucky` can compose a learned prompt, re-encode it through the connected CLIP/Gemma text encoder, and refine from that freshly tokenized conditioning.

Improved `I'm Feeling Lucky` runtime by selecting learned conditioning canvases from saved tensor metadata before decoding, capping CLIP/Gemma Lucky prompts to a practical per-run concept count, and decoding only the token vectors selected for the current generation.

Reduced redundant `I'm Feeling Lucky` work by keeping Lucky runs in one stable memory history, skipping normal prompt-variant conditioning scans while Lucky is active, validating large Lucky memories once per loaded session, and updating context relationships locally instead of writing all-to-all token graphs every run.

Fixed `I'm Feeling Lucky` CLIP/Gemma re-encode crashes when the encoded Lucky prompt has a different sequence length than the incoming conditioning by resizing the refinement delta before applying it.

Updated `I'm Feeling Lucky` CLIP/Gemma prompt composition to preserve learned comma/semicolon-separated concept phrases instead of emitting loose word lists when phrase memory is available.

Added Lucky phrase placement memory so learned prompt phrases remember their rated order positions and CLIP/Gemma Lucky prompts can reassemble phrases into a more coherent prompt order instead of sentence salad.

Fixed `I'm Feeling Lucky` bootstrap learning so sessions that start with Lucky enabled now create a real discovery history entry, seed prompt tokens/phrases, and can learn from ratings without first running the classic refinement loop.

Updated Lucky memory so normal non-Lucky runs still seed reusable token, phrase, context, and placement memory for later Lucky runs.

Updated all missing-axis ratings so `Missing details`, `Missing concept`, `Missing quality`, and paired missing ratings now mark prompt tokens as wanted-but-underrepresented instead of weak neutral feedback; repeated missing feedback reserves Lucky composition room for those tokens and their compatible neighbours.

Fixed Lucky diagnostics so the collapsed Lucky memory stream reports real Lucky update counts and learned memory size instead of implying the session is still prompt 1 out of 1.

### Removed

Removed the old public `FunPack Video Refiner` node, the `FunPackGemmaEmbeddingRefiner` compatibility alias, and `FunPack Save Refinement Latent` from the registered node list.

Removed sigma refinement, latent refinement, manual scheduler controls, and feedback-question inputs from the active Refiner workflow. These systems are not part of Refiner V2.

## [2.1.3] - 2026-04-24

### Changed

`FunPack Apply LoRA Weights` now has more user-friendly, compact UI.

`FunPack Video Refiner` now has updated logic to work more stable when provided different prompts and conditioning with each new generation.

## [2.1.1] - 2026-04-24

### Added

Added `-Just forget it-` as a Video Refiner rating. Use it when a generation failed for reasons that should not be learned from, such as a broken reference, bad seed, or workflow mistake.

Added category feedback questions for prompt phrases that the refiner cannot confidently classify. The answer scale is `general`, `concept`, `style`, `quality`, `character`, and `details`.

Added a CLIP Vision output combiner node for workflows that need one combined CLIP Vision output from multiple inputs.

### Changed

Updated the Video Refiner rating categories so feedback can separate missing details, missing concept, missing quality, and fully failed output instead of treating all bad results the same way.

Reduced repeated category feedback prompts after the user has already answered enough about the same concept.

Refreshed README and refiner docs for 2.1.1.

### Fixed

Fixed LoRA weight row restore order when workflows are loaded.

## [2.1.0] - 2026-04-23

### Added

Added `FunPack Apply LoRA Weights` and `FunPack LoRA Loader`, a prompt-exact LoRA weight workflow designed to work with `FunPack Video Refiner`.

Added `FunPack Save Refinement Latent`, which stores latent tensor bundles by refinement key for optional latent refinement in `FunPack Video Refiner`.

Added hidden LTX per-block LoRA redistribution for supported `ltx2` model stacks. The UI still exposes normal LoRA weights, while the loader derives per-block strengths from the LoRA patch magnitudes when the model and LoRA layout support it.

The new workflow uses base LoRA weights on the first run for a prompt, then lets the refiner save prompt-specific suggested LoRA weights into its existing JSON state for later runs.

### Changed

Renamed the visible refiner title from `FunPack Gemma Embedding Refiner` to `FunPack Video Refiner`. The old node key is still available as a compatibility alias.

Split the old single `funpack.py` implementation into focused modules:

- `conditioning.py`
- `samplers.py`
- `image_processing.py`
- `model_management.py`

`funpack.py` remains as a compatibility re-export for older imports.

Updated `FunPack Video Refiner` so it can accept a FunPack LoRA stack and save next-run model LoRA weight suggestions based on prompt concepts, LoRA type hints, and user ratings.

Updated `FunPack Video Refiner` with optional latent input/output refinement. If no matching saved latent exists and both latent input and output are connected, the input latent is saved as the first reference and passed through unchanged.

Updated prompt analysis so quoted speech and backslash-wrapped phrases can be protected as whole prompt units.

### Documentation

Documented unintended and edge-case usage for the new refiner workflow, including disconnected latent paths, saved-latent-only runs, wrong LTX audio/AV latent connections, exact-prompt LoRA lookup behavior, base-weight mismatch behavior, zero-weight LoRA skipping, and unsupported per-block fallback behavior.

## [1.3.3] - 2026-04-22

### Changed

Expanded `/docs` so every node in `funpack.py` now has matching documentation, and refreshed the existing node docs to match the current inputs and outputs.

## [1.3.2] - 2026-04-19

### Changed

Changed the core logic of Self-Refiner.
Removed obsolete nodes.

## [1.3.0 & 1.3.1] - 2026-04-18

### Changed

Added new nodes - User Rating and Gemma Self-Refinement for LTX2.3 video workflows.

### Fixed

Device type mismatch in new nodes.

## [1.2.3] - 2026-01-30

### Fixed

Fixed Transformers library error when running Prompt Enhancer and Story Writer nodes.

## [1.2.2] - 2026-01-26

### Changed

Changed the logic of processing sequences in Story Writer node. Now doesn't append full instructions and previous context to previous messages with each loop iteration, now fully replaces messages with a system prompt and sequence history without appending.

## [1.2.1] - 2026-01-24

### Added

Added experimental LoRA recommendation feature and Sanity Check features to Story Writer node.

## [1.2.0] - 2026-01-23

### Added

Added new Story Writer node, based on existing Prompt Enhancer. It generates up to 5 prompts one after another, based on either user's prompt directly, or on the story generated from the user's prompt.

## [1.1.0] - 2026-01-02

### Added

Added Creative Template and Lorebook Enhancer nodes. The Creative Template is a wildcard-based node that replaces given keywords in the template with ones provided by user. Lorebook Enhancer is a node that takes SillyTavern format .json lorebooks and enhances your prompt by adding required knowledge.

## [1.0.0] - 2026-01-01

Initial release on Comfy Registry.
