# FunPack LoRA Weight Workflow

This workflow lets `FunPack Video Refiner` save LoRA weight suggestions for an exact prompt. Loading stays predictable: the Apply node chooses the weights, the Loader loads them, and the Refiner updates suggestions after you rate the result.

## Node Order

Use this pattern:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Video Refiner`

The loader does not learn from ratings. The refiner does that part.

## FunPack Apply LoRA Weights

This node lists the LoRAs you want to use and their normal base weights.

Inputs:

- **positive_prompt**: Prompt text used for lookup.
- **refinement_key**: Same key used by `FunPack Video Refiner`.
- **mode**: Same tokenizer mode as the refiner.
- **per_block**: For supported `ltx2` stacks, lets the loader spread the LoRA strength across transformer blocks behind the scenes.
- **lora_N**: LoRA file.
- **lora_N_type**: `general`, `concept`, `style`, `quality`, or `character`.
- **lora_N_base_weight**: Trainer-recommended model base weight.

Use **+ Add LoRA** for more rows.

The first run uses base weights. Later runs can use saved suggestions for the same `refinement_key`, `mode`, and prompt.

Notes:

- Changing the prompt can start a new prompt record.
- Saved suggestions are ignored when the LoRA name, LoRA type, or saved base model weight no longer matches the current slot.
- `lora_N_type` is a hint for the refiner, not a loader category. A wrong type will not break loading, but it can make future suggestions less useful.
- `0.0` skips that LoRA for the current run.
- Negative weights are possible after repeated bad ratings.
- If `refinement_key` or `mode` differs from the refiner, the node falls back to base weights.

## FunPack LoRA Loader

This node loads the LoRA stack using the weights prepared by `FunPack Apply LoRA Weights`, then passes the same stack forward so the refiner can read it. The `clip` input is optional and uses zero CLIP strength when omitted.

When `per_block` is enabled on an `ltx2` stack, the loader keeps the visible global LoRA weight and derives block weights from the LoRA patches. If the model or LoRA layout is not supported, it just loads the LoRA normally.

Per-block notes:

- Per-block mode only applies to supported `ltx2` image model configs.
- Wan and other non-LTX workflows fall back to normal global LoRA loading even if `per_block` is enabled.
- LoRAs without at least two detectable transformer blocks fall back to normal global loading.
- The per-block scales are not exposed as separate UI controls.
- The loader status reports whether each LoRA used `global`, `per-block`, or a fallback path.

## Video Refiner Integration

Connect the `lora_stack` output from `FunPack LoRA Loader` into the optional `lora_stack` input on `FunPack Video Refiner`.

The refiner compares each LoRA's type and filename with the concepts it extracted from the prompt. After it processes your rating, it saves `lora_weight_suggestions` into the prompt history.

Examples:

- `I like it` gently reinforces matching LoRAs.
- `Missing details` nudges related concept, character, and general LoRAs upward.
- `Missing concept` favors the best concept or character match and can reduce weaker competing concept LoRAs.
- `Missing quality` can favor quality LoRAs and reduce unrelated concept LoRAs.
- `I don't like it` can mute or invert a LoRA if it keeps hurting the same prompt.
- `-Just forget it-` skips LoRA suggestion updates for that run.

If `lora_stack` is not connected, conditioning, sigma, and latent refinement still work, but LoRA suggestions are not updated.
