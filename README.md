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

The requirements file uses the CUDA 12.9 PyTorch wheel index and requires:

- `torch >= 2.8.0`
- `transformers >= 5.0.0`

Higher Torch versions are fine. The important part is avoiding older Torch releases that may break FunPack workflows.

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
