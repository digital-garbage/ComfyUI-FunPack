# FunPack Story Writer

This node generates up to 5 sequential prompts for video shots based on a user request, optional per-shot instructions, and an LLM.

It is built around LLaMA-style chat models and is intended for story-driven multi-shot generation workflows.

## Parameters

**user_prompt**: Main instruction describing the overall story, scene, or video idea.

**prompt1, prompt2, prompt3, prompt4, prompt5**: Optional extra instructions for specific sequence slots. Leave them empty if you want the node to decide each shot from the main request alone.

**story_system_prompt**: System instruction used when the node first generates a hidden story outline.

**sequence_system_prompt**: System instruction used when generating the actual sequence prompts.

**model_path_type**: Choose between a local `.safetensors` LLM file and a HuggingFace pretrained model.

**model_path**: HuggingFace repository path used when `model_path_type` is set to pretrained.

**llm_safetensors_file**: Local `.safetensors` model file from your ComfyUI `models/clip` folder.

**prompt_count**: Number of sequence prompts to generate, from 1 to 5.

**top_p, top_k, min_p, temperature, max_new_tokens, repetition_penalty**: Text generation settings that affect variety, strictness, and output length.

**mode**: `Sequences from story` first creates a hidden story and then expands it into shots. `Sequences from user prompt` skips the hidden story step and generates shots directly from the user request.

**vision_input**: Optional text from a vision-language model describing the starting image so the generated shots can stay aligned with it.

**sanity_check**: If enabled, each generated sequence is reviewed by the same model and corrected if it breaks the requested rules.

**sanity_check_system_prompt**: System instruction used for the sanity-check pass.

**disable_continuity**: If enabled, later sequences are generated without feeding the previously generated shot history back into the model.

**provide_current_id**: When continuity is disabled, this controls whether the current sequence number is still provided to the model.

## Outputs

**prompt1, prompt2, prompt3, prompt4, prompt5**: Generated sequence prompts. Outputs beyond `prompt_count` are returned as empty strings.

## Purpose

Use this node when you want one LLM pass to turn a high-level idea into a short sequence of video prompts that still feel connected from shot to shot.
