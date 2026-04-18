# FunPack Gemma Embedding Refiner

## Parameters:

**rating** - Rating for previously generated video, from 1 to 5, where 1 means "awful, totally not what I asked for" and 5 means "masterpiece, exactly what I have asked for".

**refinement_key** - Name for the .json file that contains the tuned deltas for conditioning. You can re-use previously trained .json files - they are stored in /custom_nodes/ComfyUI-FunPack/refinements.

**reset_session** - If set to true, resets the training progress and re-creates the .json file.

**exploration_strength** - General multiplier for deltas. Lower value = lesser changes in deltas = less noticeable but more stable result, in case you need a tiny refinement. If you need the model to do what it was not supposed to produce, try higher values.

**token_prioritization_strength** - Multiplier of priority of certain tokens that are responsible for bad or good results, depending on the ratings. 0 means all tokens are treated equally. Lower value means certain tokens are slightly boosted, higher value means they are boosted significantly in comparison to others.

**max_history** - How many previously generated deltas to store inside the .json. Lower value would mean less refinement overtime due to lesser amount of information available to tweak deltas. Higher value is preferred, but in case you don't have enough free space - do not set it too high.

## Outputs:

**modified_conditioning** - tweaked deltas. Put it after CLIP Text Encode, but before LTXVConditioning nodes. Correct link: DualCLIPLoader -> CLIP Text Encode -> FunPack Gemma Embedding Refiner -> LTXVConditioning -> KSampler (or CFGGuider, or whatever sampler you are using).

**status** - Shows the current status of refinement: iteration ID, rating for previous generation, rating for the generation before it and history size.

## Purpose:

This node's purpose is to analyze the text embeddings and change them according to the given rating. Basically, this node tries to find the best values for deltas that you will more likely rate higher, and suppresses ones you rated lower, leading to consistently better output.
