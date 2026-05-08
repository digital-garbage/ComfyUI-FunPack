# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Updates in 2.3.2

Refiner V2 now learns original-intent alignment when `user_intent_prompt` stays the same but an enhancer gives different `positive_prompt` variants. Ratings teach it which intent-enhance pairs represented the original request, which original phrases were missing, and which enhancer-only additions were rejected.

Learned original-intent omissions can now be restored on later runs, while repeatedly rejected enhancer-only full words and adjacent word pairs can be omitted before encoding.

## Updates in 2.3.1

Refiner V2 Prompt Repair now keeps all missing/wrong ratings tied to the current prompt or explicit user intent. Learned favorite actions, details, quality cues, camera moves, and styles are not repaired into unrelated requests just because they scored well before.

Prompt Repair now treats the same word in different neighbor contexts as separate evidence, so a liked phrase does not automatically transfer to a different request that happens to share one word.

When the optional raw user intent is vague, such as `Figure it out`, Refiner V2 now treats the enhanced `positive_prompt` as the stronger repair anchor.

## Updates in 2.3.0

Refiner V2 now blocks learned appearance, character, subject, and background concepts from Prompt Repair, Lucky auto-injection, and legacy Void memory unless the current prompt explicitly asks for them.

Added `Wrong appearance` for outputs polluted by remembered clothing or character traits. This suppresses the responsible appearance memory without penalizing unrelated action, camera, detail, or quality learning.

Added `FunPack Refinement Key Loader` for selecting, creating, importing, and exporting Refiner V2 keys. The loader can feed both Refiner V2 and `FunPack Apply LoRA Weights`.

Added a paste-friendly [`Refiner V2 quick guide`](docs/FunPackVideoRefinerV2QuickGuide.md) for new users.

## Updates in 2.2.1

Fixed Refiner V2 prompt phrase categorization so background, appearance, quality, camera, and action phrases stay aligned before the node updates Lucky memory or LoRA suggestions.

## Updates in 2.2.0

Added `FunPack Video Refiner V2`, a simpler prompt-owned refiner that takes `positive_prompt` and a connected `CLIP`, encodes the prompt inside the node, learns from ratings, and returns refined positive conditioning plus diagnostics.

Removed Refiner V2's `mode` input. The node now accepts whatever connected `CLIP` the workflow provides and stores V2 state in a CLIP-owned namespace.

Refiner V2 removes the old sigma refinement, latent refinement, manual scheduler controls, and feedback question workflow. It now adapts internally: consistently good ratings make updates gentler, while consistently bad ratings make updates stronger.

Renamed visible Refiner and LoRA intent from `concept` to `action`. Internally, `action` means action plus motion, so movement verbs, physical motion, subject motion, and camera motion are treated together. Old `Missing concept` ratings and old `concept` LoRA rows are accepted as aliases.

Updated `I'm Feeling Lucky` in Refiner V2 so Lucky is only a prompt composer. When Lucky is off, it can still train memory from rated runs, but it does not compose or alter the output.

Removed the old public `FunPack Video Refiner`, its compatibility alias, and `FunPack Save Refinement Latent` from the node list. Use `FunPack Video Refiner V2` for new refinement workflows.

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
- [`docs/FunPackLoraWorkflow.md`](docs/FunPackLoraWorkflow.md) for the LoRA/refiner helper workflow

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an issue or submit a pull request.

## Thank you

I want to say thanks to teams behind OpenAI (ChatGPT/Codex), xAI (Grok), DeepSeek, Anthropic (Claude) and Google (Gemini) for all the help with coding and transforming my ideas into something working in real UI. Thanks to all the testers and users who regularly use FunPack in their workflows, request features and report bugs. Without all of you, this project would've been just wet dreams of a wannabe coder begging on Discord for someone to add nodes he wants. Seriously, you are cool. I love you all. <3
