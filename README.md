# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Updates in 2.1.3

Updated UI for `FunPack Apply LoRA Weights`. Now it's a one-line rgthree-like styled loader which takes up significantly less space in the workflow and looks more pleasent.

Updated rating for `FunPack Video Refiner`. Now it uses a rating system more understandable to end user:

- Rating `"I like it"` - resulting video is an exact match or really close to what was requested;
- Rating `Missing details` - really close to what was requested, main concept is present but some details/actions were omitted;
- Rating `Missing concept` - visually pleasent in terms of quality/anatomy/style/scene, but neither main concept nor details/actions are present;
- Rating `Missing quality` - visually pleasent, but lacking proper anatomy/style/scene and neither main concept nor details/actions are present;
- Rating `I don't like it` - visual garbage, totally not matching the requirements.
- Rating `-Just forget it-` - in case of failing sampling or any other result when you are unable to see the video to rate it, select this rating so the previous generation won't count in learning process.

Updated logic for `FunPack Video Refiner`. Now it has a boost in stability in cases when user provides different conditioning and positive prompt with each new generation (e.g. from prompt enhancement nodes).

## Dev Branch

The `dev` branch is intended for testing unfinished changes, implementing new logic and basically, flipping everything just because I can. It can be broken, renamed, or changed without warning.

Use only `main` if you want the most stable version of this node pack. Bug reports based on `dev` version will be ignored.

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

FunPack uses your existing ComfyUI/PyTorch install. The expected baseline is `transformers >= 5.0.0`

`hpsv3` is optional and only used by the `FunPack StoryMem Keyframe Extractor` quality filter, so it is not installed by default.

Install it manually only if you need that feature:

`pip3 install hpsv3 --no-build-isolation`

## Important Note About `hpsv3`

Installing `hpsv3` can break `Prompt Enhancer` and `Story Writer`, because `hpsv3` depends on a `transformers` version that conflicts with the version those LLM-based nodes require.

FunPack's LLM nodes require `transformers >= 5.0`. The version required for `hpsv3` is strictly `transformers==4.45.2`. Installing any version different from it will result in broken quality detector.

If you install `hpsv3`, use `--no-build-isolation`. Optionally, specify the exact version - `pip install transformers==4.45.2 --no-build-isolation`.

## Documentation

Per-node documentation is available in the [`docs`](docs) folder.

Start with:

- [`docs/FunPackGemmaEmbeddingRefiner.md`](docs/FunPackGemmaEmbeddingRefiner.md) for `FunPack Video Refiner`
- [`docs/FunPackSaveRefinementLatent.md`](docs/FunPackSaveRefinementLatent.md) for latent references
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an issue or submit a pull request.

## Thank you

I want to say thanks to teams behind OpenAI (ChatGPT/Codex), xAI (Grok), DeepSeek, Anthropic (Claude) and Google (Gemini) for all the help with coding and transforming my ideas into something working in real UI. Thanks to all the testers and users who regularly use FunPack in their workflows, request features and report bugs. Without all of you, this project would've been just wet dreams of a wannabe coder begging on Discord for someone to add nodes he wants. Seriously, you are cool. I love you all. <3
