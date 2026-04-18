# FunPack Gemma Embedding Refiner

## Parameters:

**rating** - Rating for previously generated video, from 1 to 5, where 1 means "awful, totally not what I asked for" and 5 means "masterpiece, exactly what I have asked for".

**refinement_key** - Name for the .json file that contains the tuned deltas for conditioning. You can re-use previously trained .json files - they are stored in /custom_nodes/ComfyUI-FunPack/refinements.

**reset_session** - If set to true, resets the training progress and re-creates the .json file.

**exploration_strength** - General multiplier for deltas. Lower value = lesser changes in deltas = less noticeable but more stable result, in case you need a tiny refinement. If you need the model to do what it was not supposed to produce, try higher values.

**similarity_threshold** - If you use a different prompt (conditioning on the input of the node is different from what you have used before), and the difference between these prompts reaches certain threshold, enables smart merging that adds previous "good" deltas to new ones you provided.

**token_prioritization_strength** - Multiplier of priority of certain tokens that are responsible for bad or good results, depending on the ratings. 0 means all tokens are treated equally. Lower value means certain tokens are slightly boosted, higher value means they are boosted significantly in comparison to others.

**max_history** - How many previously generated deltas to store inside the .json. Lower value would mean less refinement overtime due to lesser amount of information available to tweak deltas. Higher value is preferred, but in case you don't have enough free space - do not set it too high.

## Outputs:

**modified_conditioning** - tweaked deltas. Put it after CLIP Text Encode, but before LTXVConditioning nodes. Correct link: DualCLIPLoader -> CLIP Text Encode -> FunPack Gemma Embedding Refiner -> LTXVConditioning -> KSampler (or CFGGuider, or whatever sampler you are using).

**status** - Shows the current status of refinement: iteration ID, rating for previous generation, rating for the generation before it and history size.

## Purpose:

This node's purpose is to analyze the text embeddings and change them according to the given rating. Basically, this node tries to find the best values for deltas that you will more likely rate higher, and suppresses ones you rated lower, leading to consistently better output.

## How similarity_threshold works:

The node calculates cosine similarity between the current input embedding and the stored reference embedding.Similarity = 1.0 → almost identical prompts
Similarity = 0.7–0.85 → somewhat similar prompts (same style, different subject)
Similarity = < 0.6 → very different prompts

Current logic in the code:If similarity >= similarity_threshold → treat as "same/similar prompt" → do normal per-prompt refinement (using the last delta).
If similarity < similarity_threshold → treat as "new/different prompt" → trigger merging of good (rating ≥4) past deltas from history.

So, what should the value be?Higher threshold (e.g. 0.85 or 0.90) → more strict
Only very similar prompts will use normal refinement.
Most new prompts will trigger merging of good past deltas.
Lower threshold (e.g. 0.70 or 0.75) → more lenient
Even somewhat different prompts will be treated as "same" and use normal refinement.
Merging only triggers on quite different prompts.
