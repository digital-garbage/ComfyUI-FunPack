# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

<p align="left">
  <a href="https://ko-fi.com/M4M61MBGIT"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Ko-fi" height="36"></a>&nbsp;
  <a href="https://patreon.com/digitalgarbage?utm_medium=unknown&amp;utm_source=join_link&amp;utm_campaign=creatorshare_creator&amp;utm_content=copyLink"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Patreon" height="36"></a>&nbsp;
  <a href="https://cloud.vast.ai/?ref_id=308176"><img src="https://img.shields.io/badge/Run%20on-Vast.ai-BCFF41?style=for-the-badge&amp;logoColor=000000" alt="Vast.ai" height="36"></a>
</p>

## Dev and Cutting Edge branches

The `dev` branch is intended for testing unfinished changes, implementing new logic and basically, flipping everything just because I can. It can be broken, renamed, or changed without warning.

Use only `main` if you want the most stable version of this node pack. Bug reports based on `dev` version will be ignored.

Cutting edge branch is designed for testing the most advanced yet most breaking changes and features. I do not recommend using cutting_edge branch unless it's strictly necessary for your workflow (e.g. a feature you wanted got a rework or else) - any sort of stability is not guaranteed.

## Version 3.0 — Cutting Room and development direction

FunPack **3.0** introduces the **Cutting Room** — a dedicated montage editor at `/funpack/movie` inside ComfyUI. It is the primary surface for building multi-scene projects, previewing on a real timeline, and driving FunPack Studio + the LTXAV Scene Chain Sampler without wiring a graph by hand.

Going forward, **new features and UX work target the Cutting Room frontend**. The classic ComfyUI node popups (Studio, Refiner V2, Scene Chain Sampler, and other pre-3.0 nodes) remain fully usable but will receive **bugfixes only** — no significant UI reworks unless a fix requires it. If you live in ComfyUI graphs, nothing breaks; if you want the full montage workflow, use the Cutting Room.

**3.0.1** adds per-shortcut **refinement keys** (a shortcut can mark which key it trains, with per-scene multi-key steering and a timeline preview of the keys each scene will use) and fixes Cutting Room splitting/anchor bugs — see the [CHANGELOG](CHANGELOG.md).

See [`docs/MovieEditor.md`](docs/MovieEditor.md) for complete Cutting Room documentation.

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

- [`docs/MovieEditor.md`](docs/MovieEditor.md) for the FunPack Cutting Room (Movie Editor)
- [`docs/FunPackVideoRefinerV2.md`](docs/FunPackVideoRefinerV2.md) for `FunPack Video Refiner V2`
- [`docs/FunPackVideoRefinerV2QuickGuide.md`](docs/FunPackVideoRefinerV2QuickGuide.md) for a short Discord-friendly Refiner V2 guide
- [`docs/FunPackLTXAVSceneChainSampler.md`](docs/FunPackLTXAVSceneChainSampler.md) for split-scene LTXV/LTXAV continuation
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an issue or submit a pull request.

## Intent

FunPack is a hobby project, provided to you by a fellow AI enthusiast who "lives in the trenches" and knows exactly what people seek in video/audio generation workflows.

FunPack is provided under GNU General Public License V3, which gives you broad rights to use, modify and distribute the original/modified version of it as long as the original license text is included. FunPack places no limitations on types of content you can generate by using it, meaning both SFW and NSFW content are fine as long as you don't violate your local laws. GPLv3 does not grant you rights for such violations.

However, I do not endorse using FunPack and/or demonstrating it alongside morally and legally questionable/prohibited content, including:
- Non-consensual explicit depiction of a real person;
- Explicit depiction of minors;
- Depiction of violence and gore targeted at a real person.

I do not provide support to users who use FunPack in such cases, and in case I detect it, any support will be immediately ceased.

Thanks for understanding.

## Thank you

I express my deepest gratitude to:

- OpenAI and ChatGPT Codex;
- xAI and Grok Build;
- Anthropic and Claude Code;
- Team Cursor and Composer model;
- DeepSeek team and model;
- Google and Gemini;
- Lightricks and LTX-Video model;
- [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) — LTX model loaders and nodes used by the built-in Cutting Room pipeline;
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — video combine and helper nodes for montage export;
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — utility nodes used by the built-in pipeline;
- [OpenCut](https://github.com/opencut-app/opencut-classic) — the in-browser non-linear video editor whose UI and interaction patterns inspired the FunPack Movie Editor;
- ComfyUI and its whole community.

Without all of you, this project would've been impossible.

With <3

DigitalGarbage
