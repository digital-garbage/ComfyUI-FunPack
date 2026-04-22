# FunPack Prompt Combiner

This node merges one shared base prompt with up to 5 per-shot prompt additions.

## Parameters

**main_prompt**: Base prompt that should appear in every output.

**prompt1, prompt2, prompt3, prompt4, prompt5**: Optional prompt additions for each output slot.

## Outputs

**out1, out2, out3, out4, out5**: Combined prompts. Each output contains `main_prompt` plus its matching optional prompt input. If an optional prompt is empty, the output falls back to just the main prompt.

## Purpose

Use this node when you want to keep shared quality or style text in one place while still producing separate prompts for multiple shots or branches.
