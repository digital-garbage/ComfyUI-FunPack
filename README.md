# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Installation

FunPack is available on Comfy Registry and can be installed in any of these ways:

1. With `comfy-cli`:
   `comfy node install funpack`
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

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an Issue or submit a pull request.
