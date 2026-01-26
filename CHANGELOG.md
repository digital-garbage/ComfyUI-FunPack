# Changelog

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
