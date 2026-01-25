# FunPack Lorebook Enhancer

This node helps you with loading SillyTavern format lorebooks to use with Prompt Enhancer and Story Writer nodes, enhancing the prompt with related tags from lorebooks.

## Parameters

**prompt**: Your prompt that has to be enhanced with lorebooks tags.

**lorebook**: Full path to your .json lorebook file.

**context_history**: If enabled, remembers the previous context to use for the next generation.

**scan_depth**: How many previous messages to include into context. Min 1 (only last message), max 12.

## Outputs

**enhanced_prompt**: Your revamped prompt with injected knowledge from lorebooks.

**injected_content**: What's exactly injected into your prompt.
