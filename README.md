# ComfyUI-FunPack

A set of ComfyUI nodes for experimenting with video generation workflows based on WAN, HunyuanVideo, LTX, and similar models.

## Support

Right now, this is purely a passion project bringing me $0 per month and I'd rather have it like that, but it bites my wallet quite a lot.
I feel uncomfortable asking for financial support, yet I'd still appreciate it if you can help to sustain this project and let it live.
Vast.ai are cool guys I would recommend to you without being paid a dime (and I did that before) - https://cloud.vast.ai/?ref_id=308176 - these are providers of my AI instances I use for experiments.

You can also ask for a personal coaching where I will guide you through every single button of FunPack and explain in details what works and why.
Please contact me on Discord if you are interested - @digitalgarbage

Just in case - I will NEVER hide a single feature behind paywall or refuse to respond to your question like "What it is?" or "How does it work?" or "How do I set it up?".
Everything on this repo comes for free, available for everyone to download, use and refine to their own taste - this is covered by GPLv3 license.

With <3

DigitalGarbage

## Updates in 2.7.7

**Batch Training** brings controlled-batch RLHF to the workflow. Activate a batch and the Scene Chain sampler runs the chain `N` times with everything frozen except the noise seed, giving you `N` directly-comparable videos. Studio shows a rating panel; on submit the value function trains on the shared (frozen) conditioning with each variant's reward — same conditioning, N rewards, the cleanest comparison signal there is. Spans the engine, a Studio variant producer, the `/funpack/batch` routes and rating window, and the value-function intake. Batches live in ComfyUI's temp dir and clear on restart.

**Reactive in-flight rescue** steers the Hybrid Euler-2S sampler away from known-bad trajectories mid-sampling, using rating-gated, prompt-clustered memory of past good and bad runs. The full velocity/rescue feature set now also runs on the LTXAV rectified-flow path, with audio-safe steering (video latent only) and disk-persisted trajectory banks.

Also: **Monte Carlo conditioning search** after value-function ascent (best of a sampled neighbourhood, not the raw endpoint); `velocity_bias_strength` unlocked to `3.0` for creative action injection; Refiner V2 always trains the value function (`value_guidance` only gates application); and a Studio Variability macro with an active-feature readout.

## Updates in 2.7.6

**Value-function steering** matured. Conditioning is moved by gradient ascent toward higher predicted reward before generation (displacement-capped to prevent reward hacking), the value function exports/imports from Studio, and learned reward shapes phrase boosting, attention weights, and temperature. Added concept-in-context guidance, concept-pair bad-direction repulsion, and a **motion floor** in embed guidance that auto-boosts temporal variance when output goes static.

**Rating UI overhaul**: a rating picker popup, a `Wrong` action, a `Nailed it` rating (prompt-adherence positive, weaker than `Perfect`), and a per-option **heart modifier** replacing the standalone `Loved it` (an axis-blind quality endorsement, disabled for degraded/`Awful` ratings). `perfect_freeze` is disabled — prompt changes after a `Perfect` are now respected.

## Updates in 2.7.5

**Embed guidance** nudges conditioning toward the learned liked-quality direction each step (near-free, needs a few liked generations on the wired refinement key). An **online value function** predicts reward from rated runs and can steer sampling. **Mid-scene guide** replaces the broken `self_consistency`, using LTX guide-attention to preserve static-element layout across a scene without the audio corruption joint-attention injection caused.

Cleanup: the predefined transition-phrase list is gone (splitting is fully user/auto-driven), and the per-scene vision re-encoding + `clip` input added to the Scene Chain Sampler in 2.7.4 were removed as unstable, along with `self_consistency` and `i2i_scene_cut`.

## Updates in 2.7.4

**K/V in-context conditioning** keeps character identity across scene cuts. Reference hidden states captured at the start of each scene are prepended as attention tokens to the model's identity-formation blocks during every denoising step. Strong character consistency across view changes, orientation flips, and hard cuts — no LoRAs required.

**Gemma3 vision prompting** activates automatically when `source_image` is connected to Studio and the CLIP was loaded from a Gemma3-12B checkpoint. The reference image is encoded through the built-in SigLIP vision encoder and fused with text conditioning. No extra node needed.

**Per-scene vision re-encoding** in the Scene Chain Sampler. Connect the optional `clip` input and the sampler re-encodes each scene's conditioning using the previous scene's decoded last frame as visual context. Identical scene texts produce genuinely distinct conditioning based on what was actually generated before them.

**Soft continuation rework**: `frame_overlap=0` now carries 4 partially-denoised frames (mask 0.4) from the previous scene instead of 1 fully-pinned frame. Temporal context without pose lock.

Also: duplicate scene text now gets a "Returning to an earlier scene" prefix to break shared conditioning; anchor text punctuation is respected when joining scene segments (no stray comma after a full stop); `FunPackGemmaVision` node removed as redundant.

## Updates in 2.7.2

Shortcuts let you write compact activation words that expand into full cinematic descriptions before the prompt is encoded. Empty replacement removes unwanted phrases (e.g. game character tags). Longer phrases always win over shorter overlapping triggers. Managed in the new Studio Shortcuts tab.

Custom transitions extend the built-in scene split list with your own phrases. Each entry supports a placement override: `start` (transition opens the new scene), `end` (transition closes the previous scene), or `silent` (split point only - phrase is stripped from output). A global placement setting lives in the Studio Refiner tab with per-entry override in the Transitions tab.

Several split reliability fixes: removed single-word temporal markers that caused false splits on normal prose, fixed over-broad scene label matching, fixed dangling trailing segments, stray comma artifacts, and custom triggers ending with punctuation.

## Updates in 2.7.1

Studio now learns successful sampler seeds when the `seed` output is connected. `Perfect` and `Loved it` ratings can store concept-matched seeds under the active refinement key, and split-scene mode passes per-scene seed metadata to the Scene Chain sampler. The sampler can either use those unique scene seeds or reuse one seed for every scene with `use_same_seed`.

Scene Chain i2v guide carry is now opt-in and fixed for compact masks. The default path keeps guide carry disabled for cleaner cuts, while the experimental option correctly broadcasts compact i2v masks to full spatial chunk masks.

## Updates in 2.7.0

Added `FunPack LTXAV Scene Chain Sampler` for split-scene LTXV/LTXAV continuation in one ComfyUI run. Enable `split_by_transitions` in `FunPack Studio` or `FunPack Video Refiner V2`; the existing `modified_positive` output becomes one conditioning entry per detected scene. The sampler consumes those entries in order, samples each scene chunk with seed increments, preserves overlap from the previous chunk, and supports both plain video latents and nested LTXAV video/audio latents.

Scene splitting now uses transition order only. Written labels like `scene ten`, `scene -999999`, or `scene minus infinity` are treated as transition text, not as real scene numbers. The text before the first transition is kept as a character/global anchor and is prepended to every scene for consistency. Standalone `then` is no longer a split trigger; more specific transition phrases remain supported.

Important: the Scene Chain sampler is resource heavy. Long scene chains can create very large final latents, and you may run out of memory during VAE Decode even if sampling itself succeeds. Start with short scenes and modest `max_scenes`, then increase carefully.

Also improved the Studio Scene Builder mode dropdown refresh and expanded the transition phrase list for camera moves, final shots, and scene progression language.

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
