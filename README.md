# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Updates in 2.7.0

Added `FunPack LTXAV Scene Chain Sampler` for split-scene LTXV/LTXAV continuation in one ComfyUI run. Enable `split_by_transitions` in `FunPack Studio` or `FunPack Video Refiner V2`; the existing `modified_positive` output becomes one conditioning entry per detected scene. The sampler consumes those entries in order, samples each scene chunk with seed increments, preserves overlap from the previous chunk, and supports both plain video latents and nested LTXAV video/audio latents.

Scene splitting now uses transition order only. Written labels like `scene ten`, `scene -999999`, or `scene minus infinity` are treated as transition text, not as real scene numbers. The text before the first transition is kept as a character/global anchor and is prepended to every scene for consistency. Standalone `then` is no longer a split trigger; more specific transition phrases remain supported.

Important: the Scene Chain sampler is resource heavy. Long scene chains can create very large final latents, and you may run out of memory during VAE Decode even if sampling itself succeeds. Start with short scenes and modest `max_scenes`, then increase carefully.

Also improved the Studio Scene Builder mode dropdown refresh and expanded the transition phrase list for camera moves, final shots, and scene progression language.

Studio now learns successful sampler seeds when the `seed` output is connected. `Perfect` and `Loved it` ratings can store concept-matched seeds under the active refinement key, and split-scene mode passes per-scene seed metadata to the Scene Chain sampler. The sampler can either use those unique scene seeds or reuse one seed for every scene with `use_same_seed`.

## Updates in 2.6.0

Added `FunPack Studio` - a single node that combines Refinement Key management, Scene Builder, LoRA loading, Refiner V2, Conditioning Adjust, and sampler configuration into one interface. The node face shows only the rating widget and an Open Studio button. All settings live in a tabbed popup editor.

**Popup tabs:** Session (refinement key, scene builder mode), Scene (prompt composer, phrase bank), Refiner (all V2 settings, negative prompt, feedback, intent), Advisor (internal HuggingFace LLM), LoRA (session weight suggestions + loading with per-block support), Sampler (Hybrid Euler 2S / Distilled Flow / KSampler for high and low pass, sigma schedules as comma-separated floats), Adjustments (phrase-level conditioning shifts with session phrase bank).

Negative prompt is encoded internally via CLIP when no pre-encoded conditioning is connected, removing the need for an external CLIPTextEncode node in the negative path.

The full LoRA pipeline runs inside Studio: session weight suggestions are read first, LoRAs are applied to model and CLIP, and the Refiner direction patch is applied on top. The model output carries everything.

Two sampler outputs (high pass and low pass) sit between the seed and loss graph outputs. Wire sampler and sigmas directly to your LTX or WAN sampling nodes.

Studio popup remembers the last active tab per node across page refreshes. Field values auto-save 600ms after typing stops.

Also added as standalone nodes: `FunPack Conditioning Adjust` (phrase-level conditioning adjustment with CLIP-encoded directions, popup editor with session phrase bank).

Fixed `FunPackAdvisorLLM` crashing with newer transformers versions (`apply_chat_template` returning `BatchEncoding` instead of a plain tensor). Fixed advisor producing empty or garbled output for Qwen3 and other thinking models (thinking tokens stripped correctly, attention mask now explicitly provided). Fixed conditioning adjustments not applying for LTX/Gemma3 workflows (was reading `pooled_output` which is `None` for T5 encoders, now uses sequence mean). Fixed `prompt_repair=False` not preventing phrase injection from perfect-repair memory and emphasis. Removed the advisor gate that blocked repair for Perfect-rated outputs.

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

- [`docs/FunPackVideoRefinerV2.md`](docs/FunPackVideoRefinerV2.md) for `FunPack Video Refiner V2`
- [`docs/FunPackVideoRefinerV2QuickGuide.md`](docs/FunPackVideoRefinerV2QuickGuide.md) for a short Discord-friendly Refiner V2 guide
- [`docs/FunPackLTXAVSceneChainSampler.md`](docs/FunPackLTXAVSceneChainSampler.md) for split-scene LTXV/LTXAV continuation
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow
- [`docs/FunPackSceneBuilder.md`](docs/FunPackSceneBuilder.md) for scene preset workflows

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

I want to say thanks to teams behind OpenAI (ChatGPT/Codex), xAI (Grok), DeepSeek, Anthropic (Claude) and Google (Gemini) for all the help with coding and transforming my ideas into something working in real UI. Thanks to all the testers and users who regularly use FunPack in their workflows, request features and report bugs. Without all of you, this project would've been just wet dreams of a wannabe coder begging on Discord for someone to add nodes he wants. Seriously, you are cool. I love you all. <3
