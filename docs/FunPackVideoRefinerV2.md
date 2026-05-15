# FunPack Video Refiner V2

Video Refiner V2 is a simplified prompt-owned refiner. It takes prompt text and a connected `CLIP`, encodes the prompt inside the node, learns from ratings, and returns refined positive conditioning.

V2 replaces the old `FunPack Video Refiner` public node in version `2.2.0`. It uses separate fresh state and does not migrate old histories.

## Inputs

**positive_prompt**: Prompt text to encode and refine.

**clip**: Connected text encoder. V2 uses this CLIP for prompt encoding, phrase/category similarity checks, and optional advisor text generation when the connected CLIP supports `generate`/`decode`.

**advisor_clip**: Optional separate CLIP/Gemma text generator for the Advisor. When connected, Advisor uses this model for text generation while the main `clip` still owns prompt encoding and similarity checks. When disconnected, Advisor falls back to `clip`. If neither selected CLIP exposes text generation, Advisor skips and V2 continues with the usual workflow.

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

Use `Wrong appearance` when the video is polluted by remembered clothing, character traits, subject identity, or background concepts that were not requested for the current image. This suppresses those concepts from future auto-injection without deleting them for explicit prompts.

**refinement_key**: V2 session key. V2 stores separate `refine_v2` state under this key.

**refinement_key_input**: Optional linked key input, usually from `FunPack Refinement Key Loader`. When connected, it overrides the text widget.

**seed**: Optional seed for deterministic exploration paths.

**reset_session**: Clears the V2 session state for the selected key.

**lora_stack**: Optional stack from `FunPack Apply LoRA Weights` / `FunPack LoRA Loader`. V2 writes prompt-specific suggested model weights.

**clip_vision_output**: Optional CLIP Vision output for the original/source image. V2 stores tensor summaries as advisory context and diagnostics only; it does not blend this into positive conditioning.

**source_image**: Optional source image/frame batch. V2 stores width, height, aspect ratio bucket, and a small fingerprint so it can report whether the input image changed between runs.

**negative_prompt**: Optional negative prompt. V2 can persistently add poorly rated or wrong-context tags to this prompt, then encode the repaired prompt as negative conditioning when `CLIP` is connected.

**user_intent_prompt**: Optional raw/original request. When connected, V2 compares it with the enhanced `positive_prompt`, stores intent-enhance pairs, remembers all provided full-word and adjacent word-pair tokens for that intent, and can omit enhancer-only tokens that repeatedly led to bad ratings.

**im_feeling_lucky**: When enabled, V2 composes a learned prompt from rated phrase memory and encodes it through the connected CLIP. When disabled, V2 may still store Lucky memory from rated runs, but it does not compose or apply a Lucky prompt.

**advisor_mode**: Optional CLIP text-generation advisor:

- `Off`: no advisor call.
- `Diagnostics`: asks the connected generative CLIP for a short diagnosis, but never changes the encoded prompt.
- `Repair prompt`: asks for a repaired positive prompt and applies it only after validation against the current request, original intent, refusal filter, and protected appearance/subject/environment rules.

The advisor uses an explicit Refiner system prompt and receives the current prompt, the prompt that caused the rating, the original user intent, rating label, missing/wrong axes, phrase analysis, and rule-based repair candidates.

In `Repair prompt`, the advisor can also suggest a repaired negative prompt when the previous rating indicates wrong content, an awful result, or strong negative reward. Negative prompt advice is validated so requested subjects, actions, locations, and visual intent are not moved into the negative prompt.

**advisor_thinking**: Enables thinking mode for compatible Gemma CLIP text generators when the advisor is active.

## Appearance Safety

Refiner V2 treats image-to-video appearance as image-owned by default. Learned appearance, subject/character, and environment/background phrases are not auto-added by Prompt Repair or Lucky unless they are already present in the current prompt.

Prompt Repair is only for motion, camera work, small non-identity details, quality, and style. It will not add clothing, body traits, character identity, or background phrases from memory.

Legacy Void memory also skips appearance, subject/character, and environment/background tokens so old token-bank preferences cannot reintroduce those concepts.

## Refinement Key Loader

`FunPack Refinement Key Loader` lists existing V2 keys, creates a typed key when none is selected, and exposes `Import`, `Export`, and `Refresh` buttons in the browser UI.

Connect its `refinement_key` output to Refiner V2's `refinement_key_input` and to `FunPack Apply LoRA Weights`' `refinement_key_input` so both nodes use the same memory key.

## Outputs

**modified_positive**: Refined positive conditioning.

**status**: Short runtime summary.

**training_info**: Diagnostics including encoded prompt, category classification, automatic adaptation strength, Lucky memory status, and LoRA suggestions.

**loss_graph**: V2 learning loss graph.

**modified_negative**: Repaired negative conditioning. If the negative prompt is empty, no stored poor-rated tags are available, or `CLIP` is not connected, this output is empty.

**encoded_prompts**: One string showing the exact prompt text used for positive and negative encoding:

`Positive prompt: ...`

`Negative prompt: ...`

## Vision Context

Vision inputs are intentionally advisory in this version. The refiner stores source-image size, aspect bucket, image fingerprint, changed-image status, and CLIP Vision tensor summaries in the V2 state so ratings can be interpreted with image context. It does not directly mix CLIP Vision tensors into positive conditioning.

When `advisor_mode` is active and `source_image` is connected, V2 also passes that image to compatible CLIP text generators so the advisor can use visual context while still preserving the user's prompt and intent.

## Negative Repair

When ratings mark a prior run as `Awful`, `Wrong details`, `Wrong action`, or otherwise strongly negative, V2 stores matching poorly rated prompt phrases as negative repair tags. Future runs append the strongest stored tags to `negative_prompt` before encoding `modified_negative`.

Positive prompt repair still uses learned phrase memory and intent alignment. Repaired phrases preserve stopwords and phrase text, so phrases like `running through the street` stay intact instead of being collapsed to only content tokens.

## Removed From V2

V2 does not include sigma refinement, latent refinement, manual scheduler modes, or feedback questions.

Scheduler behavior is automatic: good streaks make V2 gentler, bad streaks make it push harder away from bad directions.

## LoRA Type Rename

V2 uses `action` instead of `concept` for motion/action LoRA intent. Old `concept` LoRA rows are treated as `action` aliases.
