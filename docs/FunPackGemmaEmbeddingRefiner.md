# FunPack Video Refiner

This node sits after your text encoder and adjusts positive conditioning based on your ratings. It can also refine sigmas, work with a saved video latent, and write prompt-specific LoRA weight suggestions for the next run.

## Parameters

**positive_conditioning**: Conditioning to refine. Place this node after your text encoder and before the downstream video conditioning node.

**mode**: Tokenizer mode used for prompt analysis. Use `ltx2` for LTX-style workflows and `wan` for Wan workflows.

**rating**: Feedback for the previous result:

- `I like it`: good result, or close enough to reinforce.
- `Missing details`: the main idea is there, but smaller requested details are missing or changed.
- `Missing concept`: the output is usable, but the main requested thing did not show up.
- `Missing quality`: the idea may be present, but the result is too messy to count as good.
- `I don't like it`: bad result overall.
- `-Just forget it-`: do not learn from this run. Use it for bad seeds, broken references, interrupted generations, or workflow mistakes.

Older numeric workflows are still understood internally: `9-10` maps to `I like it`, `7-8` to `Missing details`, `5-6` to `Missing concept`, `3-4` to `Missing quality`, and `1-2` to `I don't like it`.

**refinement_key**: Name of the refinement session. The node stores its learned refinement data under this key so you can continue training the same style later.

**scheduler_mode**: Learning behavior preset. `original` keeps the legacy behavior, while `accurate` and `aggressive` adjust how strongly the node updates the refinement.

**positive_prompt**: Optional prompt text used for token analysis and concept-level feedback.

Wrap dialogue in quotes. Wrap any other phrase that should stay together in backslashes, for example `\a woman holds his drawings in her hands\`. This keeps the phrase together during prompt analysis instead of letting it be split into loose words.

If an upstream prompt enhancer rewrites the prompt on every run, pass the final enhanced text here. The refiner keeps exact prompt histories when possible, but it can reuse a similar prompt history when the new prompt has enough concept overlap and the conditioning is close enough. This helps Gemma-style prompt variants share learning without forcing unrelated prompts into the same history.

**sigmas**: Optional input sigma schedule to co-refine alongside the conditioning. When connected, the node preserves the first and last sigma exactly and only adjusts the middle values.

**sigma_strength**: Preset strength for sigma refinement. `off` disables sigma changes, while `subtle`, `medium`, `strong`, and `max` allow progressively larger movement of the middle sigma values.

**reset_session**: If enabled, clears the saved training history for the selected refinement key and starts a fresh session.

**unlimited_history**: If enabled, disables history pruning.

**seed**: Seed used for exploration behavior during refinement.

**feedback_enabled**: If enabled, the node can return a concept-focused feedback question alongside the refined conditioning.

**feedback_rating**: Answer to the queued feedback question. The `feedback_question` output prints the scale for the current question. Category questions use `1=general`, `2=concept`, `3=style`, `4=quality`, `5=character`, and `6=details`.

**lora_stack**: Optional stack from `FunPack LoRA Loader`. The refiner uses it to save next-run model LoRA weight suggestions.

**latent**: Optional latent input. It is only used when the `refined_latent` output is also connected. If no saved latent exists yet, the node uses the connected input latent as the current reference.

## Compatibility Name

The node is now displayed as `FunPack Video Refiner`. The old `FunPackGemmaEmbeddingRefiner` node key remains available as a compatibility alias for existing workflows and older documentation links.

## Outputs

**modified_positive**: Refined conditioning. Connect this output to the next conditioning stage in your workflow.

**status**: Short node summary.

**feedback_question**: Optional follow-up question generated when feedback mode is active. It may ask how strongly a phrase should matter, or what category the phrase belongs to.

**training_info**: Diagnostics for the current step, including concept weights, LoRA suggestions, and sigma/latent refinement notes.

**loss_graph**: Graph image showing learning loss over total session iterations for the current refinement key.

**refined_sigmas**: Refined sigma schedule. If no `sigmas` input is connected, this output is empty.

**refined_latent**: Refined latent output. Connecting this output enables latent refinement logic; leaving it disconnected disables latent refinement even if the latent input is connected.

## Purpose

Use this node when you want a workflow to learn from your ratings over several runs instead of rewriting the prompt from scratch every time. Each `refinement_key` gets its own saved profile.

## Latent Refinement

Use `FunPack Save Refinement Latent` to save a reference video latent under the same `refinement_key` and `mode`. When the refiner later receives a matching latent, it can make a small rating-driven adjustment.

For LTX audio/video workflows, the latent path must use video latents only:

- Split sampler output with `LTXVSeparateAVLatent`.
- Send the video latent output to `FunPack Save Refinement Latent`.
- Send the plain video latent to `FunPack Video Refiner` latent input.
- Send `FunPack Video Refiner` `refined_latent` to `LTXVConcatAVLatent` `video_latent`.
- Send the original audio latent to `LTXVConcatAVLatent` `audio_latent`.

Do not connect the combined AV latent from `LTXVConcatAVLatent`, the sampler AV output, or the audio latent to this node.

Some LTX separated video latents keep `type: audio` metadata even after `LTXVSeparateAVLatent`; the node accepts those when their `samples` tensor is the 5D video tensor shape.

Connection behavior:

- If the latent input is disconnected and `refined_latent` is disconnected, latent tweaking is disabled.
- If the latent input is disconnected, `refined_latent` is connected, and no saved latent exists, the node errors with: `No available latent to operate. Please connect reference latent to input of Video Refiner.`
- If the latent input is disconnected, `refined_latent` is connected, and a saved latent exists, the node reports: `Running refinement on saved latent. Changing reference latent shape and size will cause no effect to generation.`
- If the latent input is connected and `refined_latent` is disconnected, latent tweaking is disabled.
- If both latent input and `refined_latent` are connected but no saved latent exists, the node passes the input latent through unchanged and saves it as the current reference.
- If both are connected and the saved latent shape does not match the current input, the node rewrites the reference and passes the current input through unchanged.
- If both are connected and the saved latent shape matches the current input, the node tweaks the current latent using the saved latent as the reference.

Resetting the session deletes the saved latent reference too.

Latent refinement is conservative by design. It tracks shape changes and leaves zero-valued positions alone.

## LoRA Suggestions

Connect the `lora_stack` output from `FunPack LoRA Loader` if you want the refiner to save suggested LoRA model weights for this exact prompt. `FunPack Apply LoRA Weights` reads those suggestions on later runs.

LoRA suggestion notes:

- If `lora_stack` is not connected, no LoRA suggestions are updated.
- If `positive_prompt` is empty, the refiner can still process conditioning, but exact prompt matching for LoRA suggestions is not useful.
- Suggestions are scoped to the same `refinement_key`, `mode`, and normalized prompt text.
- A LoRA can be reduced to `0.0` or moved negative if repeated bad ratings make it look harmful for that exact prompt.
- Changing a LoRA's base weight later prevents stale suggestions for the old base from being applied.

## Prompt Protection

Quoted text and backslash-wrapped phrases protect prompt spans during word-group analysis. Use this for dialogue, names, and short actions that lose meaning when split into separate words.

Examples:

- `"You should go for it, young man."`
- `\a woman holds his drawings in her hands\`

Unwrapped speech can still work, but the analyzer may treat some words inside it as low-value filler.

## Capability Boundary

This node cannot teach a model a concept it does not know. If a subject, action, style, object, anatomy detail, or relationship is outside the model's ability, repeated ratings will not add it.

The refiner can only adjust the strength, balance, stability, and emphasis of concepts that are already present in the model's output space. Use model choice, LoRAs, ControlNet/reference tools, better source images, or prompt changes when a concept is absent rather than merely under-weighted.
