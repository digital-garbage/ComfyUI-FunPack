# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Updates in 2.5.1

Added `FunPack Advisor LLM` node - loads any HuggingFace CausalLM (sharded or single-file) as an advisor for Refiner V2 or as a drop-in replacement for the built-in `TextGenerate` node. Set a HuggingFace repo ID or local path, pick dtype, connect the output to `advisor_clip`. Model is cached after first load. Compatible with `skip_template`, `min_p`, `presence_penalty`, and any model architecture that `AutoModelForCausalLM.from_pretrained` supports.

Advisor prompt format rewritten to plain natural language so enhancement-type models (Sulphur, Qwen prompt enhancers) produce useful output rather than ignoring structured field-value instructions. System prompt reduced to one sentence.

Training info Adaptation section now shows direction memory in plain language: run count per slot, magnitude, whether each axis is in direction mode or lerp fallback, and what was applied this run. Model patch status shows which directions were injected into cross-attention and which phrase texts are being emphasized.

Fixed several bugs: advisor returning None when tokenization succeeded (generate was inside the wrong branch), session reset not clearing newer memory fields, system prompt bleeding into advisor output, and repetition loops from the 1.05 penalty being too weak for 8B+ models.

Added persistent cross-run encode cache so phrase encodings are not recomputed every run.

## Updates in 2.5.0

Added a two-pass CLIP text-generation advisor to `FunPack Video Refiner V2`. Pass 1 analyses what specifically needs to change in the suggested prompt. Pass 2 applies those findings. The advisor uses a structured input format — `ORIGINAL_USER_INTENT`, `LAST_PROMPT`, `RATING`, and `OPTIONAL_NOTE` — so the model knows exactly what failed and what to fix. Token budget: 1200 tokens for analysis, 1600 for repair.

Added `feedback_prompt` input. When connected, the user's natural-language description of what was wrong is placed first in both advisor passes and the system follows it exactly, overriding axis-based repair logic and validation guards.

Added `Prompt only` execution mode: all prompt shaping runs normally but conditioning vectors pass through unchanged. Added `prompt_repair` boolean to disable the rule-based phrase injection from memory when not enough context has been built yet.

Added `encoded_prompts` STRING output. When the advisor ran it shows up to four sections: `Positive prompt` (what was encoded), `Advisor suggestion (applied/rejected)` (the advisor's generated prompt), `Advisor analysis` (diagnostic from the analysis pass), and `Pre-advisor prompt` (the prompt before the advisor rewrote it).

Added `eta_final` parameter to `FunPack Hybrid Euler 2S Sampler`. When set below `eta`, ancestral noise decays toward this value as sigma approaches the quality phase boundary. The early phase now uses order-2 denoised extrapolation (Adams-Bashforth 2-step) at zero extra model-call cost, and the quality phase uses progressive correction blending to reduce the number of expensive 2S evaluations.

Removed `negative_prompt` input and `modified_negative` conditioning output from Refiner V2. Negative conditioning has no effect at CFG=1.0 and was adding a redundant generation call to every advisor run.

Fixed greedy decoding in the advisor: `do_sample` was `False`, causing the model to always produce its highest-probability default output. Fixed `Only prompt` mode running two generation calls per invocation (positive + the now-removed negative advisor). Fixed `encoded_prompts` always showing only the positive prompt regardless of advisor activity. Also fixed four logic bugs: intent example lookup reading a nonexistent field, streak updates contaminating conditioning strength in Prompt only mode, unfiltered global phrase memory injecting unrelated vocabulary, and the first-run rating label being forwarded when no previous output existed.

## Updates in 2.4.2

Added Refiner V2 `Learning` mode. It observes prompts, conditioning, ratings, phrase memory, and diagnostics while passing positive and negative prompt conditioning through unchanged.

Fixed Scene Builder mode handling so the live Mode control stays independent from the selected saved scene, and fixed prompt editing so the cursor can move past a final phrase chip with the mouse or right arrow key.

## Updates in 2.4.1

Improved Scene Builder database editing: phrase rows now expose the full phrase as a hover hint, and double-click editing opens a wider multiline editor with explicit OK/Cancel buttons.

## Updates in 2.4.0

Added `FunPack Scene Builder`, a replacement for `FunPack Template Manager`. It stores named scenes from editable positive/negative prompt text, takes prompt/intent text through connection-only inputs, passes the current LoRA stack through unchanged, and can auto-apply a scene when its name or alias appears in an intent prompt.

Added Scene Builder `Learning` mode for changing-prompt generations: it collects phrase memory inside the selected refinement key while passing the connected prompt data through unchanged, and Refiner resets preserve that Scene Builder memory.

Redesigned Scene Builder as a compact button-driven node with centered editor menus for scene name, mode, aliases, Positive Prompt, Negative Prompt, and Database controls. First use asks for a scene name before editing, prompt editors preserve freely typed text while phrase chips can be clicked or dragged in from the database, used chips are highlighted in the prompt editor, database words can be double-clicked for inline editing, connected prompts now teach useful words as well as phrase chunks, and wildcard random choice is a simple checkbox.

Added searchable LoRA selection to `FunPack Apply LoRA Weights` while keeping the compact row UI and serialized LoRA stack format.

Added advisory image/CLIP Vision context and repaired negative conditioning to `FunPack Video Refiner V2`. Vision inputs are stored as metadata and diagnostics only; they are not blended into positive conditioning.

Added an opt-in experimental early velocity bias mode to `FunPack Hybrid Euler 2S Sampler` for capturing/applying averaged early denoise directions around normalized sigma 0.9 and 0.8.

## Updates in 2.3.x

Refiner V2 now supports pre-encoded conditioning workflows, original-intent alignment memory, improved prompt repair scoping, and a `Wrong appearance` rating for outputs polluted by remembered appearance concepts. Added `FunPack Refinement Key Loader`.

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
- [`docs/FunPackSceneBuilder.md`](docs/FunPackSceneBuilder.md) for scene preset workflows

Version history is available in [CHANGELOG.md](CHANGELOG.md).

## Feedback

If you have suggestions, questions, or ideas for new nodes, feel free to open an issue or submit a pull request.

## Thank you

I want to say thanks to teams behind OpenAI (ChatGPT/Codex), xAI (Grok), DeepSeek, Anthropic (Claude) and Google (Gemini) for all the help with coding and transforming my ideas into something working in real UI. Thanks to all the testers and users who regularly use FunPack in their workflows, request features and report bugs. Without all of you, this project would've been just wet dreams of a wannabe coder begging on Discord for someone to add nodes he wants. Seriously, you are cool. I love you all. <3
