# ComfyUI-FunPack

**FunPack is a UI that unlocks novel conditioning and sampling enhancements, wires cutting-edge techniques and allows simple non-linear video montage - all while using regular ComfyUI backend.**
Graphs replaced with neat cards, get/set replaced with Linked inputs, and simple text fields leveled up to Composer - prefix, prompt, postfix, $variables, Ideas button, shortcuts and splits - for your prompting comfort.

<p align="left">
  <a href="https://ko-fi.com/M4M61MBGIT"><img src="https://img.shields.io/badge/Support%20me%20on%20Ko--fi-FF5E5B?style=for-the-badge&amp;logo=kofi&amp;logoColor=white" alt="Ko-fi" height="36"></a>&nbsp;
  <a href="https://patreon.com/digitalgarbage?utm_medium=unknown&amp;utm_source=join_link&amp;utm_campaign=creatorshare_creator&amp;utm_content=copyLink"><img src="https://img.shields.io/badge/Become%20a%20Patron-FF424D?style=for-the-badge&amp;logo=patreon&amp;logoColor=white" alt="Patreon" height="36"></a>&nbsp;
  <a href="https://cloud.vast.ai/?ref_id=308176"><img src="https://img.shields.io/badge/Run%20on-Vast.ai-BCFF41?style=for-the-badge&amp;logoColor=000000" alt="Vast.ai" height="36"></a>
</p>

> [!IMPORTANT]
> FunPack is a hobby project maintained by a single person. It's crafted for my personal needs mostly, and I can't test every single possible variant of usage to confirm it's bug-free and working as intended.
> If you had found a bug or you want to request a new feature, feel free to add an entry to Issues - that would be the most appreciated.

## What's next — FunPack v5 "Cunning Cranberry"

FunPack v4.0.0 is going to be the latest release of classic FunPack.
Next update is another milestone update, or shall I say, a full rework.

Maintenance of current FunPack became a mess — forgetting to add or remove buttons from/to UI, finding functions that don't do anything but still present in the code absolutely randomly, trying to keep the UI neat while it has no strict boundaries on how it should look like, and of course — making it easy for users to understand how FunPack operates and how you can operate it — this is all time, money and token-consuming.

General idea behind the FunPack v5 rework is writing it from scratch, with a new architecture where a feature can't be forgotten, where a wire can't be missed, where one failing module never stops whole process from executing. Loaders and features would tell the UI by themselves what sorts of inputs they need and where to put it — and UI will compose them according to design guidelines it would have. Samplers would become independent from features — if a feature is available and enabled, sampler would call it. If it's not — it gets skipped and doesn't even appear in the UI so it won't confuse you.

Most of this comes from a FunUI backend draft that got shelved after understanding on how much time has to be invested into polishing it. Without a community support and wide usage it would become another dead project only one person in the world needs but no one else.

Come to think of it, FunPack already became something alike, with an exception — this project is more than alive, and while I can, I will support it.

Final goal is to make FunPack an ultimate answer for a novice's question "I don't know how to generate — what do I install and how do I start?" and for a professional's question "I need a UI with all the cutting edge techniques available on day 0 — which one?"

Preparation for the v5 had started. Work happens on the [`v5`](https://github.com/digital-garbage/ComfyUI-FunPack/tree/v5) branch.

**FunPack v5.0 "Cunning Cranberry" — coming fall 2026.**

## What's new in FunPack 4

- One app instead of two. Easy Gen is now Simple mode of Cutting Room, and both live at `/funpack/`. The Hub is gone.
- Light and dark themes, picked in Settings -> Appearance.
- Project Setup Wizard on first run, ending in the Interactive Guide.
- Models are cards on a shelf now, not a list. Nothing saves until you close the editor, and Cancel really cancels.
- FunPack's own diffusion, CLIP and VAE loaders, with weight/compute dtype and an attention backend per model.
- MiniMax H3 support, including its audio stream, keyframe pins and reference images.
- Settings live in one window. Three of them can be pinned to the toolbar.
- Clip effects on the timeline - flip, crop, fill, reverse - with preview matching the render.
- Export settings as a picture, so you can tell later what produced a video.

Experimental features that never proved out have been removed in this release. The full list is in [CHANGELOG.md](CHANGELOG.md).

Since v4.0.0 all updates will target ONLY Cutting Room editor. Nodes will receive updates but their compatibility and proper display inside regular ComfyUI frontend won't be guaranteed and/or tested.

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

## Documentation

To understand how to use FunPack Cutting Room, please consider walking through the Interactive Guide.

It's available in the end of Project Setup Wizard or at anytime from the top bar menu - Help -> Welcome tour.

Additional information can be found in [docs folder](https://github.com/digital-garbage/ComfyUI-FunPack/tree/main/docs).

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new features, feel free to open an issue or submit a pull request.

## Intent

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
