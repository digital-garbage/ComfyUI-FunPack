# FunPack Story Writer

This node is designed for generating up to 5 prompts for 5 video sequences, based on user's prompt and previous context.

It's finetuned to be used with LLaMA-3 LLM family (3, 3.1, 3.1-instruct), technically it can support LLaMA-based merges but good results are not guaranteed.

**Attention: GGUF models, as well as other newer quantized models variants are not supported.**

## Parameters

**user_prompt**: Your original prompt for generating the story.
**story_system_prompt**: Instruction for LLM on how to process your prompt and convert it into a story.
**sequence_system_prompt**: Instruction for LLM on how to generate the prompt for the next sequence.
**model_path_type**: Choose between local .safetensors file path or download a Diffusers-format model from Huggingface.
**model_path**: Path to Huggingface repository.
**llm_safetensors_file**: Selector for local .safetensors file (should be downloaded into ComfyUI/models/clip).
**prompt_count**: How many prompts to generate (min 1, max 5).
**top_p, top_k, temperature, max_new_tokens, repetition_penalty**: Parameters responsible for more creative or more strict text generation.
**mode**: Generate sequences based on the story and previous sequences or generate based on user's prompt and previous sequences (story generation is skipped).
**recommend_loras**: Experimental feature. After generating all sequences, analyzes the list of LoRAs you have and suggests which to use in each sequence. Output is in console.
**sanity_check**: Perform an additional check after generating a prompt to make sure resulting prompt is matching requirements.
**sanity_check_system_prompt**: Instruction for LLM on how to perform sanity check.
