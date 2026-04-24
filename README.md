# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## What's New in 2.1.1

Version 2.1.1 brings the current Video Refiner work from `dev` to `main`.

`FunPack Gemma Embedding Refiner` now appears in ComfyUI as `FunPack Video Refiner`. Old workflows still load, because the original node key is kept as an alias.

The main changes are:

- clearer rating choices, including `-Just forget it-` for bad runs you do not want saved
- concept feedback questions that can ask what kind of thing a prompt phrase is: concept, style, quality, character, details, or general text
- fewer repeated category questions once the refiner has enough information
- optional sigma and video-latent refinement
- prompt-specific LoRA weight suggestions through `FunPack Apply LoRA Weights` and `FunPack LoRA Loader`
- hidden per-block LoRA weighting for supported LTX model stacks
- a CLIP Vision output combiner node for workflows that need to join multiple CLIP Vision outputs

For the LoRA/refiner workflow, connect the nodes in this order:

`FunPack Apply LoRA Weights` -> `FunPack LoRA Loader` -> `FunPack Video Refiner`

For LTX audio/video latent refinement, split the AV latent before using FunPack's latent path:

`LTXVSeparateAVLatent` video output -> `FunPack Save Refinement Latent` / `FunPack Video Refiner` -> `LTXVConcatAVLatent` video input

Do not send the combined AV latent or the audio latent into the FunPack latent nodes. Use the separated video latent, then put it back with `LTXVConcatAVLatent`.

## Installation

FunPack is available on Comfy Registry and can be installed in any of these ways:

1. With `comfy-cli`:
   `comfy node install ComfyUI-FunPack`
2. With git, inside your `ComfyUI/custom_nodes` directory:
   `git clone https://github.com/digital-garbage/ComfyUI-FunPack`
3. With ComfyUI-Manager:
   Open `Custom Nodes Manager`, search for `ComfyUI-FunPack`, and click `Install`.

## Dependencies

FunPack includes a [`requirements.txt`](requirements.txt) file for its Python dependencies.

Install them with:

`pip install -r requirements.txt`

The requirements file does not include pip command flags, because some ComfyUI and registry installers reject those entries.

FunPack expects your existing ComfyUI environment to already have a compatible PyTorch install, and the requirements file only lists the extra Python packages this node set needs.

The expected baseline is:

- `torch >= 2.8.0`
- `transformers >= 5.0.0`

Higher Torch versions are fine. If your ComfyUI environment has an older Torch build, upgrade Torch there instead of through `requirements.txt`.

`hpsv3` is optional and only used by the `FunPack StoryMem Keyframe Extractor` quality filter, so it is not installed by default.

Install it manually only if you need that feature:

`pip3 install hpsv3 --no-build-isolation`

## Important Note About `hpsv3`

Installing `hpsv3` can break `Prompt Enhancer` and `Story Writer`, because `hpsv3` depends on a `transformers` version that conflicts with the version those LLM-based nodes require.

FunPack's LLM nodes require `transformers >= 5.0`.

If you install `hpsv3`, use `--no-build-isolation`.

## Documentation

Per-node documentation is available in the [`docs`](docs) folder.

Start with:

- [`docs/FunPackGemmaEmbeddingRefiner.md`](docs/FunPackGemmaEmbeddingRefiner.md) for `FunPack Video Refiner`
- [`docs/FunPackSaveRefinementLatent.md`](docs/FunPackSaveRefinementLatent.md) for latent references
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an issue or submit a pull request.

Thanks to the teams behind OpenAI, xAI, DeepSeek, Anthropic, and Google for the AI tools that helped with the code. Thanks also to everyone testing FunPack in real workflows and sending feedback. This project would still be a folder of half-working experiments without you.
