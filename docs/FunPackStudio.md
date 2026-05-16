# FunPack Studio

`FunPack Studio` is a single node that combines all core FunPack refinement tools under one interface. It replaces the typical chain of `FunPack Refinement Key Loader - FunPack Scene Builder - FunPack Apply LoRA Weights - FunPack LoRA Loader - FunPack Video Refiner V2 - FunPack Conditioning Adjust` with a single node and a tabbed popup editor.

The standalone nodes remain fully functional. Studio is an alternative for workflows where you want everything in one place.

## Node Face

Only the `rating` widget and the `Open Studio` button are visible on the node. Everything else is managed inside the popup.

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
| `positive_prompt` | STRING | Positive prompt text. Ignored when Scene Builder mode is not Pass-through. |
| `negative_prompt` | STRING | Negative prompt text. Encoded via CLIP and output as negative conditioning. Skipped when `negative_conditioning` is connected. |
| `user_intent_prompt` | STRING | Raw user intent for repair and alignment anchoring. |
| `feedback_prompt` | STRING | Feedback describing what was wrong with the previous output. Highest priority in the advisor. |
| `refinement_key_input` | STRING | Linked refinement key from a Refinement Key Loader. |

## Outputs

| Output | Type | Notes |
|---|---|---|
| `model` | MODEL | Patched model: LoRAs applied, then attn2 direction injection applied on top. |
| `modified_positive` | CONDITIONING | Refined positive conditioning with session memory applied. |
| `negative` | CONDITIONING | Negative conditioning passed through or encoded from `negative_prompt`. |
| `seed` | INT | The seed used this run. Wire to your sampler for a matching generation seed. |
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

**Scene Builder mode** is also controlled here. Changing from Pass-through to Manual/Auto/Learning activates the scene builder, which then constructs the positive prompt from the Scene tab instead of from the `positive_prompt` input.

### Scene

Available when Scene Builder mode is not Pass-through. Save and load named scene presets. The phrase bank shows all phrases from the session's universal memory. Clicking a chip appends it to the positive prompt composer.

When Scene Builder is active the Refiner tab shows a notice and the intent override field is disabled, since intent derives from the scene prompt.

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

Each of the three text inputs (negative prompt, feedback, intent) has an **Override** toggle. When off, a connected node input takes precedence and the popup value is a fallback. When on, the popup value wins regardless.

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
- `Hybrid Euler 2S` - ancestral Euler with order-2 extrapolation for motion, late DPM-Solver++ 2S for quality. Exposes eta, eta_final, s_noise, quality phase settings, motion pulse, and velocity bias controls.
- `Distilled Flow` - ODE sampler for few-step distilled models. Exposes order, final correction steps, and s_noise.
- `KSampler` - any standard ComfyUI sampler by name.

**Sigmas** are entered as a comma-separated float list. Leave empty to pass sigmas in externally.

Sampler type changes refresh the settings section immediately. All settings auto-save as you type.

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
