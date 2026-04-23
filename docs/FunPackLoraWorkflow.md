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
- **lora_N**: LoRA file.
- **lora_N_type**: `general`, `concept`, `style`, `quality`, or `character`.
- **lora_N_base_weight**: Trainer-recommended model base weight.

Use **+ Add LoRA** to add more LoRA rows as needed.

On the first run for an exact prompt, no saved suggestion exists, so base weights are used. On later runs, the node reads the refiner JSON and applies saved suggestions for that exact prompt only.

## FunPack LoRA Loader

This node only loads the LoRA stack. It applies the model weights prepared by `FunPack Apply LoRA Weights`, then passes the same stack forward so the refiner can learn from it. The `clip` input is optional and uses zero CLIP strength when omitted.

## Video Refiner Integration

Connect the `lora_stack` output from `FunPack LoRA Loader` into the optional `lora_stack` input on `FunPack Video Refiner`.

The refiner compares each LoRA's declared type and filename with the concepts it already extracted from the prompt. After processing the rating, it saves `lora_weight_suggestions` into the same prompt entry in the refiner JSON.

Examples:

- A `quality` LoRA matching a valuable quality concept can be boosted after good ratings.
- A `concept` LoRA matching a bad-rated prompt can be reduced if it may be over-weighting or distorting that concept.
- Repeated bad ratings can mark a LoRA as a likely culprit, push its automatic weight down to `0.0`, and even invert it with a negative weight if it keeps ruining output.
- Consistently good ratings can stabilize around a saved offset instead of continually drifting.
