# FunPack LoRA Weight Workflow

This workflow lets `FunPack Video Refiner` suggest prompt-specific LoRA weights while keeping LoRA loading deterministic.

## Node Order

Use this pattern:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Video Refiner`

The loader does not learn. The refiner owns prompt parsing, concept matching, rating interpretation, and saving suggested weights.

## FunPack Apply LoRA Weights

This node defines the selected LoRAs and their base weights.

Inputs:

- **positive_prompt**: Exact prompt text for lookup.
- **refinement_key**: Same key used by `FunPack Video Refiner`.
- **mode**: Same tokenizer mode as the refiner.
- **per_block**: If enabled for `ltx2` stacks, the loader derives hidden transformer block scales from each LoRA's own patch magnitudes. The UI still exposes only the regular base weights.
- **lora_N**: LoRA file.
- **lora_N_type**: `general`, `concept`, `style`, `quality`, or `character`.
- **lora_N_base_weight**: Trainer-recommended model base weight.

Use **+ Add LoRA** to add more LoRA rows as needed.

On the first run for an exact prompt, no saved suggestion exists, so base weights are used. On later runs, the node reads the refiner JSON and applies saved suggestions for that exact prompt only.

Important boundaries:

- Suggestions are keyed by the exact prompt text after the selected mode's prompt normalization. Changing punctuation, wording, or line breaks can intentionally start a different prompt record.
- Saved suggestions are ignored when the LoRA name, LoRA type, or saved base model weight no longer matches the current slot. This prevents stale learning from being applied after you change a LoRA setup.
- `lora_N_type` is a hint for the refiner, not a loader category. A wrong type will not break loading, but it can make future suggestions less useful.
- A saved or base model weight of `0.0` means the loader skips that LoRA for the current run.
- Negative suggested weights are possible after repeated poor ratings when the refiner marks a LoRA as harmful for that prompt.
- If `refinement_key` or `mode` differs from the refiner, the node correctly falls back to base weights because it is looking in a different state file.

## FunPack LoRA Loader

This node only loads the LoRA stack. It applies the model weights prepared by `FunPack Apply LoRA Weights`, then passes the same stack forward so the refiner can learn from it. The `clip` input is optional and uses zero CLIP strength when omitted.

When `per_block` is enabled on an `ltx2` stack, the loader keeps the user-facing global LoRA weight but redistributes it across detected `transformer_blocks.N` patches automatically. Blocks with stronger LoRA magnitude get a bit more weight, weaker blocks get a bit less, and non-block patches stay at the global weight. If the loader cannot detect a usable LTX transformer block layout, it falls back to the normal global application path.

Per-block notes:

- Per-block mode only applies to `ltx2` stacks on supported LTX image model configs.
- Wan and other non-LTX workflows fall back to normal global LoRA loading even if `per_block` is enabled.
- LoRAs without at least two detectable transformer blocks fall back to normal global loading.
- The per-block scales are derived internally and are intentionally not exposed as separate UI controls.
- The loader status reports whether each LoRA used `global`, `per-block`, or a fallback path.

## Video Refiner Integration

Connect the `lora_stack` output from `FunPack LoRA Loader` into the optional `lora_stack` input on `FunPack Video Refiner`.

The refiner compares each LoRA's declared type and filename with the concepts it already extracted from the prompt. After processing the rating, it saves `lora_weight_suggestions` into the same prompt entry in the refiner JSON.

Examples:

- A `quality` LoRA matching a valuable quality concept can be boosted after good ratings.
- A `concept` LoRA matching a bad-rated prompt is treated more aggressively than broad LoRAs, so it can be muted or inverted faster if it keeps distorting that concept.
- Repeated bad ratings can mark a LoRA as a likely culprit, push its automatic weight down to `0.0`, and even invert it with a negative weight if it keeps ruining output.
- Consistently good ratings can stabilize around a saved offset instead of continually drifting.

The refiner can only suggest weights for the stack it receives from `FunPack LoRA Loader`. If the `lora_stack` connection is omitted, existing conditioning, sigma, and latent refinement still work, but LoRA suggestions are not updated.
