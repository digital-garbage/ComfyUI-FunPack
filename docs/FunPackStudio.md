# FunPack Studio

`FunPack Studio` is a single node that combines all core FunPack refinement tools under one interface. It replaces the typical chain of `FunPack Refinement Key Loader - FunPack Apply LoRA Weights - FunPack LoRA Loader - FunPack Video Refiner V2 - FunPack Conditioning Adjust` with a single node and a tabbed popup editor.

The standalone nodes remain fully functional. Studio is an alternative for workflows where you want everything in one place.

## Node Face

Only the `rating` widget and the `Open Studio` button are visible on the node. Everything else is managed inside the popup.

The `rating` widget opens a **rating picker** popup. Alongside the quality ratings it offers `Nailed it` (prompt-adherence positive, weaker than `Perfect`) and a `Wrong` action, plus a per-option **heart modifier** — an axis-blind quality endorsement layered on top of any rating (it replaced the old standalone `Loved it`). The heart is disabled for quality-degraded ratings and for `Awful`. The button shows "Waiting for rating…" until a rating is chosen.

## Inputs

| Input | Type | Notes |
|---|---|---|
| `model` | MODEL | Base diffusion model. LoRAs are applied internally before the direction patch. |
| `clip` | CLIP | Text encoder. Used for prompt encoding, negative prompt encoding, and conditioning adjustments. |
| `advisor_clip` | CLIP | Pre-loaded text generator for the advisor. Overrides the LLM configured in the Advisor tab. |
| `positive_conditioning` | CONDITIONING | Pre-encoded positive conditioning. Used when CLIP is not connected. |
| `negative_conditioning` | CONDITIONING | Pre-encoded negative conditioning passed through unchanged. Takes precedence over `negative_prompt`. |
| `clip_vision_output` | CLIP_VISION_OUTPUT | Advisory image context stored in session state. Not blended into conditioning. |
| `source_image` | IMAGE | Source image or frame batch. V2 stores size, aspect ratio, and a fingerprint to detect changes. |
| `lora_stack` | FUNPACK_LORA_STACK | External LoRA stack. Bypasses Studio's internal LoRA management entirely when connected. |
| `positive_prompt` | STRING | Positive prompt text. |
| `negative_prompt` | STRING | Negative prompt text. Encoded via CLIP and output as negative conditioning. Skipped when `negative_conditioning` is connected. |
| `user_intent_prompt` | STRING | Raw user intent for repair and alignment anchoring. |
| `feedback_prompt` | STRING | Feedback describing what was wrong with the previous output. Highest priority in the advisor. |
| `refinement_key_input` | STRING | Linked refinement key from a Refinement Key Loader. |

## Outputs

| Output | Type | Notes |
|---|---|---|
| `model` | MODEL | Patched model: LoRAs applied, then attn2 direction injection applied on top. |
| `modified_positive` | CONDITIONING | Refined positive conditioning with session memory applied. When split by transitions is enabled, this becomes one conditioning entry per detected scene for the Scene Chain sampler. |
| `negative` | CONDITIONING | Negative conditioning passed through or encoded from `negative_prompt`. |
| `seed` | INT | The seed used this run. Wire to your sampler for a matching generation seed. When connected, Studio can learn successful seeds for the active refinement key. |
| `high_pass_sampler` | SAMPLER | Configured high-pass sampler object. |
| `high_pass_sigmas` | SIGMAS | Sigma schedule for the high-pass sampler. |
| `low_pass_sampler` | SAMPLER | Configured low-pass sampler object. |
| `low_pass_sigmas` | SIGMAS | Sigma schedule for the low-pass sampler. |
| `loss_graph` | IMAGE | Session learning curve visualization. |
| `status` | STRING | Execution summary for this run. |
| `training_info` | STRING | Detailed learning report: memory updates, phrase analysis, adaptation, LoRA suggestions. |
| `encoded_prompts` | STRING | Prompt text as encoded, plus advisor suggestion and diagnostic if the advisor ran. |

## Popup Tabs

### Session

Set the refinement key for the current session. This key links all FunPack memory - phrase memory, conditioning directions, LoRA suggestions, and advisor diagnostics - to a named file on disk.

When a `refinement_key_input` is connected to the node, that value takes precedence at runtime unless the **Override** toggle is enabled.

### Shortcuts

Shortcuts are global positive-prompt expansions shared by Studio and Refiner V2. Each shortcut has one or more activation phrases and one or more replacement phrases. Matching uses exact word or phrase boundaries, preserves surrounding spaces and punctuation, and chooses among replacements with the run seed.

Shortcuts are stored outside refinement keys, can be imported/exported separately, and are not cleared by **Reset session**.

### Refiner

All Refiner V2 execution settings:

- **Mode** - Refine / Prompt only / Learning
- **Advisor mode** - Off / Only diagnostics / Only prompt / Full
- **Advisor thinking** - enables extended reasoning for compatible models
- **Prompt repair** - allows V2 to inject learned phrases for missing axes
- **I'm Feeling Lucky** - compose prompt from phrase memory
- **Reset** - clears session state on next run
- **Negative prompt** - default negative text encoded via CLIP when no conditioning is connected
- **Feedback** - what was wrong with the previous output; highest priority in the advisor
- **Intent override** - overrides the `user_intent_prompt` input
- **Split by transitions** - detects scene transition phrases and outputs one conditioning entry per detected scene through `modified_positive`

Each of the three text inputs (negative prompt, feedback, intent) has an **Override** toggle. When off, a connected node input takes precedence and the popup value is a fallback. When on, the popup value wins regardless.

Split scene conditioning is intended for `FunPack LTXAV Scene Chain Sampler`. Scene labels are order-only: `scene ten` can be the first generated chunk if it is the first detected transition. Keep character description before the first transition so Studio can prepend it to every scene.

When the `seed` output is connected, `Perfect` (and any heart-modified) ratings store the previous run's seed as a concept-matched successful seed. Studio occasionally reuses matching seeds on future runs. In split mode, per-scene seeds are attached to scene conditioning metadata for the Scene Chain sampler.

`perfect_freeze` is disabled: prompt changes you make after a `Perfect` rating are respected on the next run rather than the conditioning being frozen.

#### Value function (reward-guided refinement)

Every rated run trains an online **value function** — a small MLP that predicts reward from conditioning. Training always happens; `value_guidance` only gates whether the learned reward is *applied*, and defaults on. When applied, the positive conditioning is moved by gradient ascent toward higher predicted reward before sampling (single and split-scene paths), displacement-capped to prevent reward hacking, followed by a Monte Carlo search of the neighbourhood and a final gate so the best candidate is chosen rather than the raw ascent endpoint. The value function can be exported/imported from Studio and is cleared on **Reset session**.

For per-step (rather than pre-generation) application of the learned liked-quality direction, enable `embed_guidance` on the `FunPack LTXAV Scene Chain Sampler` with `refinement_key_input` wired to the same key.

### Advisor

Enables and configures an internal HuggingFace CausalLM advisor. Uses the same model cache as the standalone `FunPack Advisor LLM` node, so the model is only loaded once even when both nodes are in the same workflow.

Set a HuggingFace repo ID or absolute local path and pick the dtype. The model loads on the first run and stays cached. Set **Advisor mode** in the Refiner tab to activate it.

An external `advisor_clip` input always overrides this setting.

### LoRA

Configure the LoRA pipeline. Studio runs the full chain internally:

1. `FunPack Apply LoRA Weights` reads session weight suggestions for the current prompt and builds a stack with adjusted model weights.
2. `FunPack LoRA Loader` applies the stack to model and CLIP.
3. The V2 direction patch is applied on top of the LoRA-patched model.

**Model type** (ltx2 / wan) and **Per-block** settings apply to the Apply LoRA Weights step.

Add LoRA entries with name, type, model weight, and CLIP weight. The list is fetched from ComfyUI's configured LoRA folder.

An external `lora_stack` input bypasses this entire tab.

### Sampler

Configure two independent sampler outputs: **High Pass** and **Low Pass**. Each pass independently selects a sampler type and sigma schedule.

**Sampler types:**
- `Hybrid Euler 2S` - ancestral Euler with order-2 extrapolation for motion, late DPM-Solver++ 2S for quality (Heun corrector on rectified-flow/LTXAV models). Exposes eta, eta_final, s_noise, quality phase settings, motion pulse, velocity bias, and reactive rescue controls.
- `Distilled Flow` - ODE sampler for few-step distilled models. Exposes order, final correction steps, and s_noise, plus the same velocity bias and reactive rescue controls as the Hybrid sampler (sharing the same trajectory memory). On few-step schedules these fire less often than on an 8-step run, since fewer steps land on a velocity target.
- `KSampler` - any standard ComfyUI sampler by name.

**Sigmas** are entered as a comma-separated float list. Leave empty to pass sigmas in externally.

For the `Hybrid Euler 2S` type, a **Variability macro** sets the steering features (motion pulse, velocity bias, rescue) toward more or less variation in one move, and an **active-feature readout** summarises which steering features (velocity bias, rescue, embed guidance, value guidance) are live for the current configuration. Velocity bias is a creative / action-injection control (with an emergent scene-cut prior), not a consistency tool; reactive rescue is rating-gated and only acts once you have rated a few generations for the prompt/key.

Sampler type changes refresh the settings section immediately. All settings auto-save as you type.

### Batch Training

Runs a controlled-batch RLHF cycle. With a batch active, the `FunPack LTXAV Scene Chain Sampler` runs the chain `N` times with everything frozen except the per-variant noise seed, producing `N` directly-comparable videos. This tab shows a rating panel where you score every variant; on submit the value function trains on the batch's shared (frozen) conditioning with each variant's reward — same conditioning, N rewards, which is the variance-reduced comparison signal. Phase 3b also learns axis and direction memory from the batch ratings. Batches live in ComfyUI's temp directory (served via the `/funpack/batch` routes) and are wiped on restart.

### Vision

When `source_image` is connected and the `clip` input was loaded from a Gemma3-12B checkpoint via `DualCLIPLoader`, Studio automatically encodes the reference image through the built-in SigLIP vision encoder. The resulting vision tokens are fused with the text conditioning so the model conditions on both the prompt and the actual pixel content of the reference frame.

A **vision-conditioning toggle** in this tab lets you turn the fusion off without disconnecting `source_image` (e.g. to keep the image purely advisory). The capability is otherwise detected at runtime by checking whether the CLIP's Gemma3 transformer has `vision_model` and `multi_modal_projector` attributes. The console logs `[FunPackStudio] Processing input image with Gemma3 vision...` when active and reports the token count on completion.

Vision encoding encodes the full frame including background. For character consistency across varied environments keep the reference image background neutral or ensure the scene prompts are explicit enough to override the environmental bias from the vision tokens.

### Adjustments

Manually adjust the encoded conditioning by blending specific phrase directions. Bypasses V2's category classification - the direction is computed directly from the phrase you type, not inferred from which axis it belongs to.

Connect the same CLIP that encodes the positive prompt. Positive strength pushes conditioning toward the phrase, negative pushes away. Typical useful range: -0.3 to +0.3.

When a refinement key is connected the session phrase bank appears below the list, showing all phrases V2 has learned. Click a chip to add it with a default strength of +0.1.

## Override Toggles

Three text inputs have both a node input connector and a popup field: `refinement_key`, `feedback_prompt`, and `user_intent_prompt`. Each has a small **Override** toggle:

- **Toggle off** (default): the connected input wins; the popup field is a fallback when nothing is connected.
- **Toggle on**: the popup field wins regardless of what is connected.

## Persistence

The popup remembers the last active tab per node across page refreshes via localStorage. All field values auto-save to the node widget 600ms after you stop typing, so changes survive a browser refresh without requiring you to click Close.
