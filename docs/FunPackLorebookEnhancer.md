# FunPack Lorebook Enhancer

This node loads SillyTavern-style lorebook JSON files and appends matching lore entries to the end of your prompt.

## Parameters

**prompt**: Base prompt that should be enhanced with lorebook content.

**lorebook_1, lorebook_2, lorebook_3, lorebook_4**: Full paths to lorebook `.json` files. You can use one lorebook or combine up to four.

**entry_delimiter**: Optional prefix added before every injected lorebook entry. This is useful when the enhanced prompt is fed into an LLM and you want each injected entry to start with a recognizable marker.

**context_history**: Additional context text that is scanned together with the current prompt when looking for matching lorebook entries.

**scan_depth**: How many trailing lines from `context_history + prompt` should be scanned for lorebook triggers.

## Outputs

**enhanced_prompt**: Your original prompt with the activated lorebook entries appended to the end. If `entry_delimiter` is set, each injected entry is prefixed with it.

**injected_content**: A readable list of the exact lorebook entries that were injected, including the applied delimiter prefix.

## Purpose

Use this node to bring character, world, or setting information from SillyTavern lorebooks into Prompt Enhancer or Story Writer workflows without manually copying lore into every prompt.
