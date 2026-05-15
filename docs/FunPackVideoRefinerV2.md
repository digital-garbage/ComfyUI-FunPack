# FunPack Video Refiner V2

Video Refiner V2 is a prompt-owned refiner. It takes prompt text and a connected `CLIP`, encodes the prompt inside the node, learns from ratings, and returns refined positive conditioning.

V2 replaces the old `FunPack Video Refiner` public node. It uses separate fresh state and does not migrate old histories.

## Inputs

**positive_prompt**: Prompt text to encode and refine. When an LLM enhancer is in the workflow this is the enhancer output. V2 treats it as the "suggested prompt" when building advisor context.

**rating**: Feedback for the previous V2 generation:

- `Perfect`
- `Missing details`
- `Missing action`
- `Missing quality`
- `Missing details + action`
- `Wrong details`
- `Wrong action`
- `Wrong details + action`
- `Wrong appearance`
- `Missing details + quality`
- `Missing action + quality`
- `Awful`
- `-Just forget it-`

Older `Missing concept` labels are accepted as aliases for `Missing action`.

Use `Wrong appearance` when the video is polluted by remembered clothing, character traits, subject identity, or background concepts that were not requested for the current image.

**mode**: Execution mode:

- `Refine`: applies learned prompt shaping and conditioning vector adaptation. Default.
- `Prompt only`: applies all prompt shaping but passes conditioning vectors through unchanged. Learning still runs. Use when you want to pause conditioning adaptation while prompt refinement continues.
- `Learning`: observes ratings and updates all memory, but passes both prompt and conditioning through unchanged.

**advisor_mode**: CLIP text-generation advisor:

- `Off`: no advisor call.
- `Only diagnostics`: runs the analysis pass only, stores the finding in feedback history as an `Advisor note:` entry, and returns the prompt unchanged. Use this to preview what the advisor would say before committing to `Full`.
- `Only prompt`: runs the repair pass only (no analysis). Changes the prompt silently; outputs a plain prompt string with no labels.
- `Full`: runs both passes. Pass 1 identifies what specifically needs to change. Pass 2 applies those findings. The advisor's suggestion, analysis text, and pre-repair prompt are all visible in `encoded_prompts`.

The advisor receives four explicit inputs: `ORIGINAL_USER_INTENT` (the user's intent prompt), `LAST_PROMPT` (the previous encoded prompt), `RATING` (the current rating label), and `OPTIONAL_NOTE` (feedback prompt, memory suggestions, history, and analysis combined). Token budget: 1200 for the analysis pass, 1600 for the repair pass.

**clip**: Text encoder. V2 uses this for prompt encoding, phrase and category similarity checks, and advisor text generation when the model exposes `generate`/`decode`.

**advisor_clip**: Optional separate generative CLIP/Gemma model for the advisor. When connected, the advisor uses this model for both analysis and repair passes while `clip` continues handling encoding and similarity. When disconnected, the advisor falls back to `clip`.

**advisor_thinking**: Enables thinking mode for compatible Gemma CLIP text generators.

**feedback_prompt**: Optional natural-language description of what was wrong with the previous output, e.g. `he was supposed to hold her hand not her head`. When connected it is placed first in both advisor passes and the system follows it exactly, bypassing axis-based repair logic, intent distance checks, and protected category rules.

**prompt_repair**: Enables or disables the rule-based phrase injection from phrase memory for missing axes (default on). Turn off early in a session before enough context has accumulated, or when memory suggestions are disrupting the generation.

**refinement_key**: V2 session key. All memory and history is stored under this key.

**refinement_key_input**: Optional linked key input, usually from `FunPack Refinement Key Loader`. When connected, overrides the text widget.

**seed**: Optional seed for deterministic exploration paths.

**reset_session**: Clears all V2 session state for the selected key. Scene Builder memory is preserved.

**lora_stack**: Optional stack from `FunPack Apply LoRA Weights` / `FunPack LoRA Loader`. V2 writes prompt-specific suggested model weights.

**clip_vision_output**: Optional CLIP Vision output for the source image. Stored as advisory context and diagnostics only; not blended into positive conditioning.

**source_image**: Optional source image or frame batch. V2 stores width, height, aspect bucket, and a fingerprint to report when the input image changed between runs. Also passed to vision-capable advisor models.

**user_intent_prompt**: Optional raw user request before any LLM enhancement. V2 compares it with `positive_prompt`, stores intent-enhance pairs, and can restore omitted original phrases or suppress repeatedly rejected enhancer-only additions.

**im_feeling_lucky**: Composes a learned prompt from rated phrase memory and encodes it through the connected CLIP.

## Feedback History

V2 maintains a rolling history of the last ten user-provided `feedback_prompt` entries in session state. Each entry is labelled with its rating, for example:

```
1. Missing action: he was supposed to hold her hand not her head
2. Wrong details: the smoke was rising instead of drifting sideways
3. Advisor note: motion keywords are underrepresented in the suggested prompt
```

Entries from `Only diagnostics` runs are stored as `Advisor note:` and carry forward into `Full` mode runs. Both advisor passes see the full history so patterns across runs influence the analysis and repair.

## Outputs

**modified_positive**: Refined positive conditioning.

**status**: Short runtime summary including mode, advisor status, and repair actions taken.

**training_info**: Full diagnostics including encoded prompt, category classification, adaptation strength, Lucky memory status, and LoRA suggestions.

**loss_graph**: V2 learning loss graph.

**encoded_prompts**: The exact prompt text used for encoding, plus advisor context when the advisor ran:

```
Positive prompt: ...

Advisor suggestion (applied): ...

Advisor analysis: ...

Pre-advisor prompt: ...
```

`Advisor suggestion` is labelled `(applied)` when validation passed and the prompt was changed, or `(rejected)` when validation blocked it. `Pre-advisor prompt` only appears when the suggestion was applied and the prompt changed. `Advisor analysis` only appears when the analysis pass ran (`Full` mode).

## Appearance Safety

Refiner V2 treats image-to-video appearance as image-owned. Learned appearance, subject/character, and environment/background phrases are not auto-added by Prompt Repair or Lucky unless already present in the current prompt.

When `feedback_prompt` is connected, appearance-related changes requested by the user are applied without restriction.

## Refinement Key Loader

`FunPack Refinement Key Loader` lists existing V2 keys, creates a typed key when none is selected, and exposes `Import`, `Export`, and `Refresh` buttons in the browser UI.

Connect its `refinement_key` output to Refiner V2's `refinement_key_input` and to `FunPack Apply LoRA Weights`' `refinement_key_input` so both nodes share the same memory.

## Removed From V2

V2 does not include sigma refinement, latent refinement, manual scheduler modes, or feedback questions.

Scheduler behaviour is automatic: good streaks make V2 gentler, bad streaks make it push harder away from bad directions. In `Prompt only` mode, streaks and reward EMA are not updated so prompt-only ratings do not affect conditioning adaptation strength.
