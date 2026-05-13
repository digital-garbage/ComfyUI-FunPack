# FunPack LoRA Weight Workflow

This workflow lets `FunPack Video Refiner V2` save LoRA weight suggestions for an exact prompt. Loading stays predictable: the Apply node chooses the weights, the Loader loads them, and the Refiner updates suggestions after you rate the result.

## Node Order

Use this pattern:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Video Refiner V2`

The loader does not learn from ratings. The refiner does that part.

## FunPack Apply LoRA Weights

This node lists the LoRAs you want to use and their normal base weights.

Inputs:

- **positive_prompt**: Prompt text used for lookup.
- **refinement_key**: Same key used by `FunPack Video Refiner V2`.
- **mode**: Stack namespace for loader behavior and legacy suggestion fallback. Refiner V2 suggestions are CLIP-agnostic and do not depend on this value.
- **per_block**: For supported `ltx2` stacks, lets the loader analyze LoRA block deltas and balance competing block strengths behind the scenes.
- **lora_N**: LoRA file.
- **lora_N_type**: `general`, `action`, `style`, `quality`, or `character`.
- **lora_N_base_weight**: Trainer-recommended model base weight.

Use **+ Add LoRA** for more rows. Clicking a LoRA name opens a searchable picker, so large LoRA folders can be filtered by partial filename instead of scrolling the full list.

The first run uses base weights. Later runs can use saved V2 suggestions for the same `refinement_key` and prompt.

Notes:

- Changing the prompt can start a new prompt record.
- Saved suggestions are ignored when the LoRA name, LoRA type, or saved base model weight no longer matches the current slot.
- `lora_N_type` is a hint for the refiner, not a loader category. A wrong type will not break loading, but it can make future suggestions less useful.
- `0.0` skips that LoRA for the current run.
- Negative weights are possible after repeated bad ratings.
- If `refinement_key` or prompt differs from the refiner, the node falls back to base weights.

## FunPack LoRA Loader

This node loads the LoRA stack using the weights prepared by `FunPack Apply LoRA Weights`, then passes the same stack forward so the refiner can read it. The `clip` input is optional and uses zero CLIP strength when omitted.

When `per_block` is enabled on an `ltx2` stack, the loader keeps the visible global LoRA weight and derives block weights from the LoRA patches. With multiple supported LoRAs, it compares their block fingerprints before loading and gently boosts or dampens overlapping blocks according to each LoRA's type hint. If the model or LoRA layout is not supported, it just loads the LoRA normally.

Per-block notes:

- Per-block mode only applies to supported `ltx2` image model configs.
- Wan and other non-LTX workflows fall back to normal global LoRA loading even if `per_block` is enabled.
- LoRAs without at least two detectable transformer blocks fall back to normal global loading.
- The loader caches recently used raw LoRA files, model-mapped LoRA patches, and block fingerprints, so adjusting weights and rerunning should avoid most repeated LoRA loading and analysis work.
- The per-block scales are not exposed as separate UI controls.
- Type hints affect conflict balancing: `character` and `action` LoRAs get more protection in contested semantic blocks, `quality` LoRAs stay more supportive, and `style` or `general` LoRAs yield more readily when they overlap heavily.
- The loader status reports whether each LoRA used `global`, `per-block`, `smart-per-block`, or a fallback path.
- `smart-per-block` status includes the strongest detected blocks and the largest overlap score, which is useful for spotting LoRAs that are fighting over the same region.

## Video Refiner Integration

Connect the `lora_stack` output from `FunPack LoRA Loader` into the optional `lora_stack` input on `FunPack Video Refiner V2`.

The refiner compares each LoRA's type and filename with the action, detail, quality, style, and character phrases it extracted from the prompt. After it processes your rating, it saves `lora_weight_suggestions` into the prompt history.

Examples:

- `Perfect` gently reinforces matching LoRAs.
- `Missing details` nudges related action, character, style, and general LoRAs upward.
- `Missing action` boosts matching action and character LoRAs.
- `Missing quality` boosts quality LoRAs, with small support from matching style/general LoRAs.
- Pair ratings boost both named axes in the same run.
- `Awful` boosts details, action, and quality together.
- `-Just forget it-` skips LoRA suggestion updates for that run.

If `lora_stack` is not connected, conditioning refinement still works, but LoRA suggestions are not updated.
