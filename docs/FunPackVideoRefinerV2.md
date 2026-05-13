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
- `Missing details + quality`
- `Missing action + quality`
- `Awful`
- `-Just forget it-`

Older `Missing concept` labels are accepted as aliases for `Missing action`.

**refinement_key**: V2 session key. V2 stores separate `refine_v2` state under this key.

**seed**: Optional seed for deterministic exploration paths.

**reset_session**: Clears the V2 session state for the selected key.

**lora_stack**: Optional stack from `FunPack Apply LoRA Weights` / `FunPack LoRA Loader`. V2 writes prompt-specific suggested model weights.

**clip_vision_output**: Optional CLIP Vision output for the original/source image. V2 stores tensor summaries as advisory context and diagnostics only; it does not blend this into positive conditioning.

**source_image**: Optional source image/frame batch. V2 stores width, height, aspect ratio bucket, and a small fingerprint so it can report whether the input image changed between runs.

**negative_prompt**: Optional negative prompt. V2 can persistently add poorly rated or wrong-context tags to this prompt, then encode the repaired prompt as negative conditioning.

**im_feeling_lucky**: When enabled, V2 composes a learned prompt from rated phrase memory and encodes it through the connected CLIP. When disabled, V2 may still store Lucky memory from rated runs, but it does not compose or apply a Lucky prompt.

## Outputs

**modified_positive**: Refined positive conditioning.

**status**: Short runtime summary.

**training_info**: Diagnostics including encoded prompt, category classification, automatic adaptation strength, Lucky memory status, and LoRA suggestions.

**loss_graph**: V2 learning loss graph.

**modified_negative**: Repaired negative conditioning. If the negative prompt is empty and V2 has no stored poor-rated tags to add, this output is empty.

## Vision Context

Vision inputs are intentionally advisory in this version. The refiner stores source-image size, aspect bucket, image fingerprint, changed-image status, and CLIP Vision tensor summaries in the V2 state so ratings can be interpreted with image context. It does not directly mix CLIP Vision tensors into positive conditioning.

## Negative Repair

When ratings mark a prior run as `Awful`, `Wrong details`, `Wrong action`, or otherwise strongly negative, V2 stores matching poorly rated prompt phrases as negative repair tags. Future runs append the strongest stored tags to `negative_prompt` before encoding `modified_negative`.

Positive prompt repair still uses learned phrase memory. Repaired phrases preserve stopwords and phrase text, so phrases like `running through the street` stay intact instead of being collapsed to only content tokens.

## Removed From V2

V2 does not include sigma refinement, latent refinement, manual scheduler modes, or feedback questions.

Scheduler behavior is automatic: good streaks make V2 gentler, bad streaks make it push harder away from bad directions.

## LoRA Type Rename

V2 uses `action` instead of `concept` for motion/action LoRA intent. Old `concept` LoRA rows are treated as `action` aliases.
