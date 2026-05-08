# FunPack Video Refiner V2

Video Refiner V2 is a simplified prompt-owned refiner. It takes prompt text and a connected `CLIP`, encodes the prompt inside the node, learns from ratings, and returns refined positive conditioning.

V2 replaces the old `FunPack Video Refiner` public node in version `2.2.0`. It uses separate fresh state and does not migrate old histories.

## Inputs

**positive_prompt**: Prompt text to encode and refine.

**clip**: Connected text encoder. V2 uses this CLIP for prompt encoding and phrase/category similarity checks. It does not load an external tokenizer.

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

**im_feeling_lucky**: When enabled, V2 composes a learned prompt from rated phrase memory and encodes it through the connected CLIP. When disabled, V2 may still store Lucky memory from rated runs, but it does not compose or apply a Lucky prompt.

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

## Removed From V2

V2 does not include sigma refinement, latent refinement, manual scheduler modes, or feedback questions.

Scheduler behavior is automatic: good streaks make V2 gentler, bad streaks make it push harder away from bad directions.

## LoRA Type Rename

V2 uses `action` instead of `concept` for motion/action LoRA intent. Old `concept` LoRA rows are treated as `action` aliases.
