# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## What's New in 2.1.0

Version 2.1.0 promotes the latest refiner workflow from `dev` to `main`.

`FunPack Gemma Embedding Refiner` is now displayed as `FunPack Video Refiner`. The old node key remains available as a compatibility alias, so existing workflows that reference `FunPackGemmaEmbeddingRefiner` can continue to load.

The Video Refiner now supports:

- concept-aware feedback and clearer split `status` / `training_info` outputs
- protected prompt phrases with either quoted speech or backslash-wrapped phrases
- optional sigma refinement
- optional video-latent refinement through `FunPack Save Refinement Latent`
- prompt-exact LoRA weight suggestions through `FunPack Apply LoRA Weights` and `FunPack LoRA Loader`
- hidden per-block LoRA redistribution for supported LTX model stacks

For the LoRA/refiner workflow, use this order:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Video Refiner`

For LTX audio/video latent refinement, split the AV latent first:

`LTXVSeparateAVLatent` video output -> `FunPack Save Refinement Latent` / `FunPack Video Refiner` -> `LTXVConcatAVLatent` video input

Do not send combined AV latents or audio latents into the FunPack latent refinement path. Those structures are intentionally rejected so the workflow does not silently refine the wrong tensor.

## Installation

FunPack is available on Comfy Registry and can be installed in any of these ways:

1. With `comfy-cli`:
   `comfy node install ComfyUI-FunPack`
2. With git, inside your `ComfyUI/custom_nodes` directory:
   `git clone https://github.com/digital-garbage/ComfyUI-FunPack`
3. With ComfyUI-Manager:
   Open `Custom Nodes Manager`, search for `ComfyUI-FunPack`, and click `Install`.

## Dependencies

FunPack includes a [`requirements.txt`](requirements.txt) file for the Python packages it expects outside the standard library.

Install them with:

`pip install -r requirements.txt`

The requirements file intentionally avoids pip command flags because some ComfyUI/registry installers treat those as invalid requirement entries.

FunPack expects your existing ComfyUI environment to already have a compatible PyTorch install, and the requirements file only lists the extra Python packages this node set needs.

The expected baseline is:

- `torch >= 2.8.0`
- `transformers >= 5.0.0`

Higher Torch versions are fine. The important part is avoiding older Torch releases that may break FunPack workflows. If your ComfyUI environment has an older Torch build, upgrade Torch in that environment separately instead of through `requirements.txt`.

`hpsv3` is still an optional dependency used only by the `FunPack StoryMem Keyframe Extractor` quality filter, so it is not included in the default requirements.

Install it manually only if you need that feature:

`pip3 install hpsv3 --no-build-isolation`

## Important Note About `hpsv3`

Installing `hpsv3` can break `Prompt Enhancer` and `Story Writer`, because `hpsv3` depends on a `transformers` version that conflicts with the version those LLM-based nodes require.

FunPack's LLM nodes require `transformers >= 5.0`.

If you install `hpsv3`, use `--no-build-isolation`. Otherwise the install may appear to succeed while the node still cannot detect a working `hpsv3` package.

## Documentation

Per-node documentation is available in the [`docs`](docs) folder.

Start with:

- [`docs/FunPackGemmaEmbeddingRefiner.md`](docs/FunPackGemmaEmbeddingRefiner.md) for `FunPack Video Refiner`
- [`docs/FunPackSaveRefinementLatent.md`](docs/FunPackSaveRefinementLatent.md) for latent references
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an Issue or submit a pull request.

Thanks to teams of OpenAI, xAI, DeepSeek, Anthropic and Google and their respective AI large language models for providing all the help with the code and transforming my stupid ideas into something actually running in your UI. Thanks to all testers and users who use FunPack in their workflows daily and provide feedback in any way. Without all of you, this would have been a dream of a wannabe coder nerd.
