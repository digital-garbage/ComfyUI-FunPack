# ComfyUI-FunPack

**FunPack turns ComfyUI into an AI movie studio.** At its heart is the **Cutting Room** — a full browser-based non-linear video editor where you write a script, arrange scenes on a real timeline, and generate a complete montage with one click. Behind it sits a self-improving generation engine: rate your clips and FunPack learns your taste, steering future generations toward what you liked. The nodes that power all of this (Studio, Video Refiner V2, the LTXAV Scene Chain Sampler, and more) remain fully usable in classic ComfyUI graph workflows with WAN, HunyuanVideo, LTX, and similar models.

<p align="left">
  <a href="https://ko-fi.com/M4M61MBGIT"><img src="https://img.shields.io/badge/Support%20me%20on%20Ko--fi-FF5E5B?style=for-the-badge&amp;logo=kofi&amp;logoColor=white" alt="Ko-fi" height="36"></a>&nbsp;
  <a href="https://patreon.com/digitalgarbage?utm_medium=unknown&amp;utm_source=join_link&amp;utm_campaign=creatorshare_creator&amp;utm_content=copyLink"><img src="https://img.shields.io/badge/Become%20a%20Patron-FF424D?style=for-the-badge&amp;logo=patreon&amp;logoColor=white" alt="Patreon" height="36"></a>&nbsp;
  <a href="https://cloud.vast.ai/?ref_id=308176"><img src="https://img.shields.io/badge/Run%20on-Vast.ai-BCFF41?style=for-the-badge&amp;logoColor=000000" alt="Vast.ai" height="36"></a>
</p>

> [!IMPORTANT]
> FunPack is the project maintained by one person. Finding bugs, or features not working as intended, all alone - is nearly impossible - especially in an advanced piece of software like Editor. I've tried my best to make setup and acquaintance with FunPack as easy as possible, introducing multiple failsafes, but I can't envision how exactly you will use it.
>
> With this said - FunPack is provided "as is". Stable, uninterrupted work is NOT guaranteed even on main branch. But that doesn't mean bugs won't be fixed - you are free to file a bug report and I will most likely fix it in the nearest time.

## What it looks like

FunPack 3.x ships the **Cutting Room** - a full non-linear video editor living in your browser, inside ComfyUI. A real timeline with per-scene prompts, ratings and audio lanes, a program monitor, a media bin, and one-click generation driving FunPack Studio + the Scene Chain Sampler - no graph wiring required.

![FunPack Cutting Room - timeline, preview monitor, media bin, and scene inspector](docs/images/cutting-room.png)

| The Composer - global prompt, shortcuts, split markers, `$variables` | Engine Settings - Studio + Chain Sampler knobs without touching a graph |
| --- | --- |
| ![Composer window](docs/images/composer.png) | ![Engine Settings dialog](docs/images/engine-settings.png) |

Want a guided look at every panel? Open the editor and press **Help ▸ Welcome tour** - the same tour also auto-generates a full annotated gallery and walkthrough video via [`tools/showcase`](tools/showcase).

## Dev and other non-main branches

The `dev` branch is intended for testing unfinished changes, implementing new logic and basically, flipping everything just because I can. It can be broken, renamed, or changed without warning.

Use only `main` if you want the most stable version of FunPack. Bug reports based on `dev` version will be ignored.

From time to time other short-lived branches may appear for the most advanced yet most breaking changes and features (e.g. a large rework in progress). I do not recommend using any non-`main` branch unless it's strictly necessary for your workflow — any sort of stability is not guaranteed, and such branches can be force-pushed or deleted at any time.

## Version 3.0 — Cutting Room and development direction

FunPack **3.0** introduces the **Cutting Room** — a dedicated montage editor at `/funpack/movie` inside ComfyUI. It is the primary surface for building multi-scene projects, previewing on a real timeline, and driving FunPack Studio + the LTXAV Scene Chain Sampler without wiring a graph by hand.

Going forward, **new features and UX work target the Cutting Room frontend**. The classic ComfyUI node popups (Studio, Refiner V2, Scene Chain Sampler, and other pre-3.0 nodes) remain fully usable but will receive **bugfixes only** — no significant UI reworks unless a fix requires it. If you live in ComfyUI graphs, nothing breaks; if you want the full montage workflow, use the Cutting Room.

**3.0.1** adds per-shortcut **refinement keys** (a shortcut can mark which key it trains, with per-scene multi-key steering and a timeline preview of the keys each scene will use) and fixes Cutting Room splitting/anchor bugs — see the [CHANGELOG](CHANGELOG.md).

**3.1.1** adds prompt **`$variables`** and global-prompt **templates** in the Composer, a
cross-shot **JoyAI-Echo** audio/video memory mode, contrastive-pair **FreeSliders**, and a
seed-routing **path-outcome planner** that steers future generations away from disliked paths —
see the [CHANGELOG](CHANGELOG.md) for the full list.

**3.1.2** fixes a Cutting Room regression where splitting a clip could undo itself (the second
half got pushed out as if a full-length clip had been appended) — see the
[CHANGELOG](CHANGELOG.md).

**3.1.3** brings several **experimental sampler techniques** (ALG anchor de-staticking and
Momentum Guidance on Distilled Flow, Bounded Attention for multi-subject scenes), an **Auto
Montage** trailer builder, a **Temp files** browser, **reconnect-after-reload** for in-flight
generations, and a Blackwell (sm_120) guide-scene crash fix — see the [CHANGELOG](CHANGELOG.md)
for the full list.

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

If you have suggestions, questions, or ideas for new features, feel free to open an issue or submit a pull request.

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
