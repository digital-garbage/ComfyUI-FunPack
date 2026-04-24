# Changelog

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
