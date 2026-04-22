# FunPack Lorebook Enhancer

This node loads SillyTavern-style lorebook JSON files and appends matching lore entries to the end of your prompt.

## Parameters

**prompt**: Base prompt that should be enhanced with lorebook content.

**lorebook_1, lorebook_2, lorebook_3, lorebook_4**: Full paths to lorebook `.json` files. You can use one lorebook or combine up to four.

**entry_delimiter**: Extra delimiter text intended for separating injected entries. At the moment, the node always appends entries line by line, so this input currently has no visible effect.

**context_history**: Additional context text that is scanned together with the current prompt when looking for matching lorebook entries.

**scan_depth**: How many trailing lines from `context_history + prompt` should be scanned for lorebook triggers.

## Outputs

**enhanced_prompt**: Your original prompt with the activated lorebook entries appended to the end.

**injected_content**: A readable list of the exact lorebook entries that were injected.

## Purpose

Use this node to bring character, world, or setting information from SillyTavern lorebooks into Prompt Enhancer or Story Writer workflows without manually copying lore into every prompt.
