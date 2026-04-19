# FunPack Gemma Embedding Refiner

## Parameters:

**rating** - Rating for previously generated video, from 1 to 10, where 1 means "awful, totally not what I asked for" and 10 means "masterpiece, exactly what I have asked for".

Rating is NOT balanced. 5 is not the true middle, 5.5 is.

**refinement_key** - Name for the .json file that contains the tuned deltas for conditioning. You can re-use previously trained .json files - they are stored in /custom_nodes/ComfyUI-FunPack/refinements.

**positive_prompt** - Connect your prompt into this input to provide tokens for tokenization and better refinement based on used tokens.

**reset_session** - If set to true, resets the training progress and re-creates the .json file.

**unlimited_history** - If set to true, limitation of maximum embeddings that can be stored is removed (default: 200).

## Outputs:

**modified_conditioning** - tweaked deltas. Put it after CLIP Text Encode, but before LTXVConditioning nodes. Correct link: DualCLIPLoader -> CLIP Text Encode -> FunPack Gemma Embedding Refiner -> LTXVConditioning -> KSampler (or CFGGuider, or whatever sampler you are using).

**status** - Shows the current status of refinement: iteration ID, rating for previous generation, prompt similarity threshold, last reward change, EMA, ratio of good deltas, exploration strength, exploration status and top 10 tokens the current embedding emphasizes on.

## Purpose:

This node's purpose is to analyze the text embeddings and change them according to the given rating. Basically, this node tries to find the best values for deltas that you will more likely rate higher, and suppresses ones you rated lower, leading to consistently better output.
