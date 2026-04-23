# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## What's New in 2.0.0

FunPack is now split into focused Python modules for easier editing and maintenance:

- `conditioning.py`
- `samplers.py`
- `image_processing.py`
- `model_management.py`

Version 2.0.0 also adds a prompt-exact LoRA weight workflow that works together with `FunPack Gemma Embedding Refiner`.

Use this order:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Gemma Embedding Refiner`

`FunPack Apply LoRA Weights` defines selected LoRAs, LoRA type, and model base weights. If the refiner has saved suggested weights for the exact prompt, it applies them; otherwise it uses the base weights.

`FunPack LoRA Loader` loads the prepared LoRA stack into the model. Its CLIP input is optional and is left untouched when omitted.

`FunPack Gemma Embedding Refiner` does the prompt/concept/rating work and saves next-run LoRA weight suggestions into its existing refinement JSON.

## Installation

FunPack is available on Comfy Registry and can be installed in any of these ways:

1. With `comfy-cli`:
   `comfy node install ComfyUI-FunPack`
2. With git, inside your `ComfyUI/custom_nodes` directory:
   `git clone https://github.com/aimfordeb/ComfyUI-FunPack`
3. With ComfyUI-Manager:
   Open `Install custom nodes`, search for `ComfyUI-FunPack`, and click `Install`.

## Dependencies

FunPack now includes a [`requirements.txt`](requirements.txt) file for the Python packages it expects outside the standard library.

Install them with:

`pip install -r requirements.txt`

The requirements file intentionally avoids pip command flags because some ComfyUI/registry installers treat those as invalid requirement entries.

FunPack expects your existing ComfyUI environment to already have a compatible PyTorch install, and the requirements file only lists the extra Python packages this node set needs.

The expected baseline is:

- `torch >= 2.8.0`
- `transformers >= 5.0.0`

Higher Torch versions are fine. The important part is avoiding older Torch releases that may break FunPack workflows. If your ComfyUI environment has an older Torch build, upgrade Torch in that environment separately instead of through `requirements.txt`.

## Known Limitation

`FunPack Gemma Embedding Refiner` can optionally refine `SIGMAS`, but that currently breaks audio generation in workflows that also produce audio streams. If you need audio to generate and mux correctly, do not connect `sigmas` to the refiner. Leave sigma schedules on their original path and use the refiner only for conditioning.

UPD: This is not caused by Embedding Refiner node. Updating to latest nightly version of ComfyUI solves the problem.

`hpsv3` is still an optional dependency used only by the `FunPack StoryMem Keyframe Extractor` quality filter, so it is not included in the default requirements.

Install it manually only if you need that feature:

`pip3 install hpsv3 --no-build-isolation`

## Important Note About `hpsv3`

Installing `hpsv3` can break `Prompt Enhancer` and `Story Writer`, because `hpsv3` depends on a `transformers` version that conflicts with the version those LLM-based nodes require.

FunPack's LLM nodes require `transformers >= 5.0`.

If you install `hpsv3`, use `--no-build-isolation`. Otherwise the install may appear to succeed while the node still cannot detect a working `hpsv3` package.

## Documentation

Per-node documentation is available in the [`docs`](docs) folder.

The LoRA/refiner helper workflow is documented in [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md).

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an Issue or submit a pull request.

Thanks to teams of OpenAI, xAI, DeepSeek, Anthropic and Google and their respective AI large language models for providing all the help with the code and transforming my stupid ideas into something actually running in your UI. Thanks to all testers and users who use FunPack in their workflows daily and provide feedback in any way. Without all of you, this would have been a dream of a wannabe coder nerd.
