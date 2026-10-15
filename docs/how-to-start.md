# How to start generating with FunPack

This is the short path from "installed" to "first clip". It does not explain every node or loader: the app does that where you use them (hover a control, read its hint).

## 1. Install

Any one of these:

- `comfy node install ComfyUI-FunPack`
- `git clone https://github.com/digital-garbage/ComfyUI-FunPack` inside `ComfyUI/custom_nodes`
- ComfyUI-Manager: Custom Nodes Manager, search `ComfyUI-FunPack`, Install.

Then `pip install -r requirements.txt` in the same Python environment ComfyUI uses, and restart ComfyUI. FunPack uses your existing ComfyUI and PyTorch; it never upgrades them on its own.

## 2. Open FunPack

Start ComfyUI as usual and open `/funpack/` on the same address, for example `http://127.0.0.1:8188/funpack/`. The chip at the top right says whether ComfyUI is live.

## 3. Make a project

**File ▸ Project Setup Wizard.** Give the project a name and pick **Start from**: the pipeline of the model you will use. Say yes to the tour if you want a walk through the screen (Help ▸ Start tour runs it again).

| Model | State |
|---|---|
| MiniMax H3 | The model FunPack is built and tested around. |
| Stable Diffusion 1.5 | Experimental. |
| LTX 2.x | **UNTESTED.** Never run on a real LTX model, and nobody plans to test it soon. You are free to try it; if something breaks, please report it in [Issues](https://github.com/digital-garbage/ComfyUI-FunPack/issues). |

## 4. Pick your files

The model's pipeline starts empty: it needs your files.

1. Open **Settings ▸ Ready to generate?**. It lists everything still missing.
2. Open **Settings ▸ Models & Pipeline** and choose a file in each card marked with an orange dot (diffusion model, text encoder, VAE, and so on).

Put model files where ComfyUI expects them (`models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/loras`). When nothing is missing, Ready to generate? says so.

`.gguf` files load only for the architectures ComfyUI-GGUF understands. Anything else is not supported yet (Gemma 4 `.gguf` is one such case).

## 5. Write the story

Click **Composer** and write all scenes in one box, separated by the cut word (see the Cuts tab). Each piece becomes one clip on the timeline. Composer also keeps `$variables`, templates and shortcuts, and has an optional **Enhance** tab: a language model rewrites each finished prompt. On H3, pick a chat encoder (for example Gemma 4) under Enhance ▸ Encoder, because H3's own encoder cannot write prose.

## 6. Generate

- **▶ Generate** makes every scene on the timeline (not left-out scenes, video clips or Auto Montage pieces). An earlier render of a scene stays as a take.
- **Selected** makes only the picked scene. **Takes ×N** makes the same scene several times, differing only by seed.
- No picture needed: on H3, a scene with no source image is generated from text alone.

Watch the Log window if nothing happens: any module that fails says why there, and the rest of the run goes on without it.

## 7. Cut, render, export

Trim and split clips on the timeline. **⚡ Auto Montage** builds a new cut at the end from your rendered clips; your originals stay as they are. **⧉ Render** joins the timeline into one file; **⤓ Export** saves the picked clip.

## 8. Teach it your taste (optional)

Rate a result with **Rate scene…** (liked, disliked, bad image, bad composition). Set a **Taste key** (the ⚠ chip on the timeline, or **Settings ▸ Refinement & Taste**) so ratings are remembered. Modules that learn from ratings start acting after a few liked clips and say so when they need more.

## 9. Features marked experimental

**Settings ▸ Modules** switches each feature on or off and shows its state. Experimental ones are built and run, but not yet proven over many runs. If one fails it is switched off for the rest of that run and the log says why.

## If something goes wrong

- Nothing happens: check the ComfyUI chip and the Log window.
- Wrong look: Settings ▸ Appearance. Lost a layout: View ▸ Reset Layout. Lost a project: File ▸ Open recent.
- A bug or a missing feature: please open an [issue](https://github.com/digital-garbage/ComfyUI-FunPack/issues).

## Known limits in 5.0

- LTX 2.x is untested (see above).
- On H3 the scene picture sets the video's shape only; using it as a start frame is not implemented.
- The H3 "second pass" preset needs a compatible H3 latent upscaler file and has not been verified.
