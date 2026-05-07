# Changelog

## [2.2.0] - 2026-05-07

### Added

Added `FunPack Video Refiner V2`, a simplified prompt-owned refiner that accepts `positive_prompt` and a connected `CLIP`, owns prompt encoding internally, learns from ratings, and returns refined positive conditioning plus diagnostics.

Added `FunPack Template Manager`, a preset node for storing prompts, activation words, refinement keys, sigma schedules, and FunPack LoRA stacks with import/export support.

Added `I'm Feeling Lucky` mode to `FunPack Video Refiner V2`. It works as a preference composer that can inject learned user-preferred actions, camera moves, details, and styles even when the current prompt is vague.

### Changed

Updated LTX per-block LoRA loading so supported stacks now compare LoRA block fingerprints across the whole stack and apply type-aware conflict balancing before patches are loaded.

Fixed `FunPack Hybrid Euler 2S Sampler` restart timing so `restart_trigger_pct` is respected across the full sigma schedule instead of being clamped to the Euler-to-2S quality transition.

Improved `FunPack LoRA Loader` rerun performance by caching recently used raw LoRA files, model-mapped LoRA patches, and per-block fingerprint analysis.

Reworked `FunPack Video Refiner V2` ratings around explicit missing-axis signals: `Perfect`, single missing axes, paired missing axes, and `Awful`.

Removed the Refiner V2 `mode` input. V2 now accepts whatever connected `CLIP` the workflow provides and stores state in a CLIP-owned namespace.

Renamed visible Refiner and LoRA intent from `concept` to `action`. Old `Missing concept` ratings and old `concept` LoRA rows are still accepted as compatibility aliases, but V2 stores and displays `action`.

Updated `I'm Feeling Lucky` in Refiner V2 so Lucky only composes prompt text when enabled. When disabled, it may train memory from rated runs but does not compose or alter output.

### Fixed

Fixed `I'm Feeling Lucky` token-bank learning for changing prompt/conditioning workflows by falling back to prompt-order token placement when exact tokenizer position matching cannot find enough words.

Fixed `I'm Feeling Lucky` rating attribution for changing prompts so ratings update the previous prompt's learned tokens while the current prompt seeds new neutral discovery tokens.

Updated `I'm Feeling Lucky` filtering to learn poor adjacent token pairs instead of refusing individual tokens outright.

Updated `I'm Feeling Lucky` with uncapped token, pair, and context memory so it can learn which concepts belong together and call strong missing neighbors when prompt anchors are present.

Fixed `I'm Feeling Lucky` composition order so the current generation uses already-learned memory first, then seeds current prompt tokens for future runs.

Fixed `I'm Feeling Lucky` memory-first output so vague or empty incoming conditioning can use the longest compatible learned conditioning canvas instead of being limited to the current prompt's shape/content.

Added an optional `clip` input to `FunPack Video Refiner` so `I'm Feeling Lucky` can compose a learned prompt, re-encode it through the connected CLIP/Gemma text encoder, and refine from that freshly tokenized conditioning.

Improved `I'm Feeling Lucky` runtime by selecting learned conditioning canvases from saved tensor metadata before decoding, capping CLIP/Gemma Lucky prompts to a practical per-run concept count, and decoding only the token vectors selected for the current generation.

Reduced redundant `I'm Feeling Lucky` work by keeping Lucky runs in one stable memory history, skipping normal prompt-variant conditioning scans while Lucky is active, validating large Lucky memories once per loaded session, and updating context relationships locally instead of writing all-to-all token graphs every run.

Fixed `I'm Feeling Lucky` CLIP/Gemma re-encode crashes when the encoded Lucky prompt has a different sequence length than the incoming conditioning by resizing the refinement delta before applying it.

Updated `I'm Feeling Lucky` CLIP/Gemma prompt composition to preserve learned comma/semicolon-separated concept phrases instead of emitting loose word lists when phrase memory is available.

Added Lucky phrase placement memory so learned prompt phrases remember their rated order positions and CLIP/Gemma Lucky prompts can reassemble phrases into a more coherent prompt order instead of sentence salad.

Fixed `I'm Feeling Lucky` bootstrap learning so sessions that start with Lucky enabled now create a real discovery history entry, seed prompt tokens/phrases, and can learn from ratings without first running the classic refinement loop.

Updated Lucky memory so normal non-Lucky runs still seed reusable token, phrase, context, and placement memory for later Lucky runs.

Updated all missing-axis ratings so `Missing details`, `Missing concept`, `Missing quality`, and paired missing ratings now mark prompt tokens as wanted-but-underrepresented instead of weak neutral feedback; repeated missing feedback reserves Lucky composition room for those tokens and their compatible neighbours.

Fixed Lucky diagnostics so the collapsed Lucky memory stream reports real Lucky update counts and learned memory size instead of implying the session is still prompt 1 out of 1.

### Removed

Removed the old public `FunPack Video Refiner` node, the `FunPackGemmaEmbeddingRefiner` compatibility alias, and `FunPack Save Refinement Latent` from the registered node list.

Removed sigma refinement, latent refinement, manual scheduler controls, and feedback-question inputs from the active Refiner workflow. These systems are not part of Refiner V2.

## [2.1.3] - 2026-04-24

### Changed

`FunPack Apply LoRA Weights` now has more user-friendly, compact UI.

`FunPack Video Refiner` now has updated logic to work more stable when provided different prompts and conditioning with each new generation.

## [2.1.1] - 2026-04-24

### Added

Added `-Just forget it-` as a Video Refiner rating. Use it when a generation failed for reasons that should not be learned from, such as a broken reference, bad seed, or workflow mistake.

Added category feedback questions for prompt phrases that the refiner cannot confidently classify. The answer scale is `general`, `concept`, `style`, `quality`, `character`, and `details`.

Added a CLIP Vision output combiner node for workflows that need one combined CLIP Vision output from multiple inputs.

### Changed

Updated the Video Refiner rating categories so feedback can separate missing details, missing concept, missing quality, and fully failed output instead of treating all bad results the same way.

Reduced repeated category feedback prompts after the user has already answered enough about the same concept.

Refreshed README and refiner docs for 2.1.1.

### Fixed

Fixed LoRA weight row restore order when workflows are loaded.

## [2.1.0] - 2026-04-23

### Added

Added `FunPack Apply LoRA Weights` and `FunPack LoRA Loader`, a prompt-exact LoRA weight workflow designed to work with `FunPack Video Refiner`.

Added `FunPack Save Refinement Latent`, which stores latent tensor bundles by refinement key for optional latent refinement in `FunPack Video Refiner`.

Added hidden LTX per-block LoRA redistribution for supported `ltx2` model stacks. The UI still exposes normal LoRA weights, while the loader derives per-block strengths from the LoRA patch magnitudes when the model and LoRA layout support it.

The new workflow uses base LoRA weights on the first run for a prompt, then lets the refiner save prompt-specific suggested LoRA weights into its existing JSON state for later runs.

### Changed

Renamed the visible refiner title from `FunPack Gemma Embedding Refiner` to `FunPack Video Refiner`. The old node key is still available as a compatibility alias.

Split the old single `funpack.py` implementation into focused modules:

- `conditioning.py`
- `samplers.py`
- `image_processing.py`
- `model_management.py`

`funpack.py` remains as a compatibility re-export for older imports.

Updated `FunPack Video Refiner` so it can accept a FunPack LoRA stack and save next-run model LoRA weight suggestions based on prompt concepts, LoRA type hints, and user ratings.

Updated `FunPack Video Refiner` with optional latent input/output refinement. If no matching saved latent exists and both latent input and output are connected, the input latent is saved as the first reference and passed through unchanged.

Updated prompt analysis so quoted speech and backslash-wrapped phrases can be protected as whole prompt units.

### Documentation

Documented unintended and edge-case usage for the new refiner workflow, including disconnected latent paths, saved-latent-only runs, wrong LTX audio/AV latent connections, exact-prompt LoRA lookup behavior, base-weight mismatch behavior, zero-weight LoRA skipping, and unsupported per-block fallback behavior.

## [1.3.3] - 2026-04-22

### Changed

Expanded `/docs` so every node in `funpack.py` now has matching documentation, and refreshed the existing node docs to match the current inputs and outputs.

## [1.3.2] - 2026-04-19

### Changed

Changed the core logic of Self-Refiner.
Removed obsolete nodes.

## [1.3.0 & 1.3.1] - 2026-04-18

### Changed

Added new nodes - User Rating and Gemma Self-Refinement for LTX2.3 video workflows.

### Fixed

Device type mismatch in new nodes.

## [1.2.3] - 2026-01-30

### Fixed

Fixed Transformers library error when running Prompt Enhancer and Story Writer nodes.

## [1.2.2] - 2026-01-26

### Changed

Changed the logic of processing sequences in Story Writer node. Now doesn't append full instructions and previous context to previous messages with each loop iteration, now fully replaces messages with a system prompt and sequence history without appending.

## [1.2.1] - 2026-01-24

### Added

Added experimental LoRA recommendation feature and Sanity Check features to Story Writer node.

## [1.2.0] - 2026-01-23

### Added

Added new Story Writer node, based on existing Prompt Enhancer. It generates up to 5 prompts one after another, based on either user's prompt directly, or on the story generated from the user's prompt.

## [1.1.0] - 2026-01-02

### Added

Added Creative Template and Lorebook Enhancer nodes. The Creative Template is a wildcard-based node that replaces given keywords in the template with ones provided by user. Lorebook Enhancer is a node that takes SillyTavern format .json lorebooks and enhances your prompt by adding required knowledge.

## [1.0.0] - 2026-01-01

Initial release on Comfy Registry.
