# How to start generating with FunPack

This is the short path from "installed" to "first clip". It does not explain every node or loader: the app does that where you use them (hover a control, read its hint).

## 1. Install

With git, inside `ComfyUI/custom_nodes`:

`git clone -b v5 https://github.com/digital-garbage/ComfyUI-FunPack`

Without `-b v5` you get the classic 4.x. `comfy node install ComfyUI-FunPack` and ComfyUI-Manager install what is published on Comfy Registry, which is 5.0 only once it is published there; check Settings ▸ About after installing.

Then `pip install -r requirements.txt` in the same Python environment ComfyUI uses, and restart ComfyUI. FunPack uses your existing ComfyUI and PyTorch. If pip wants to upgrade a package ComfyUI compiles against (torch, numpy, transformers, comfy-kitchen), read what it will change first: FunPack's own updater only installs what is missing and reports the rest.

## 2. Open FunPack

Start ComfyUI as usual and open `/funpack/` on the same address, for example `http://127.0.0.1:8188/funpack/`. The chip at the top right says whether ComfyUI is live.

## 3. Make a project

**File ▸ Project Setup Wizard.** Give the project a name and pick **Start from**: the pipeline of the model you will use. Say yes to the tour if you want a walk through the screen (Help ▸ Start tour runs it again).

| Model | State |
|---|---|
| MiniMax H3 | The model FunPack is built and tested around. |
| Stable Diffusion 1.5 | Experimental. |
| LTX 2.x (UNTESTED) | **UNTESTED.** Never run on a real LTX model, and nobody plans to test it soon. You are free to try it; if something breaks, please report it in [Issues](https://github.com/digital-garbage/ComfyUI-FunPack/issues). |

## 4. Pick your files

The model's pipeline starts empty: it needs your files.

1. Open **Settings ▸ Ready to generate?**. It lists everything still missing.
2. Open **Settings ▸ Models & Pipeline** and choose a file in each card marked with an orange dot (diffusion model, text encoder, VAE, and so on).

Put model files where ComfyUI expects them (`models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/loras`). When nothing is missing, Ready to generate? says so.

`.gguf` text encoders work only for architectures ComfyUI-GGUF understands; any other is refused with a message (Gemma 4 `.gguf` is one such case). Use the `.safetensors` file of the same model.

## 5. Write the story

Click **Composer** and write all scenes in one box, separated by the cut word (see the Cuts tab). Each piece becomes one clip on the timeline. Composer also keeps `$variables`, templates and shortcuts, and has an optional **Enhance** tab: a language model rewrites each finished prompt. On H3, pick a chat encoder (for example Gemma 4) under Enhance ▸ Encoder, because H3's own encoder cannot write prose. Leave Encoder family at its default unless your encoder belongs to another family (Gemma 4 worked with the default).

## 6. Generate

- **▶ Generate** makes every scene on the timeline (not left-out scenes, video clips or Auto Montage pieces). An earlier render of a scene stays as a take.
- **Selected** makes only the picked scene. **Takes ×N** makes the same scene several times, differing only by seed.
- No picture needed: on H3, a scene with no source image is generated from text alone.

Watch the Log window if nothing happens: any module that fails says why there, and the rest of the run goes on without it.

## 7. Cut, render, export

Trim and split clips on the timeline. **⚡ Auto Montage** builds a new cut at the end from your rendered clips; your originals stay as they are. **⧉ Render** joins the timeline into one file; **⤓ Export** saves the picked clip.

## 8. Teach it your taste (optional)

Rate a result with **Rate scene…** (liked, disliked, bad image, bad composition). Set a **Taste key** (the ⚠ chip on the timeline points to it: **Settings ▸ Engine ▸ System**) so ratings are remembered. **Settings ▸ Refinement & Taste** lists, exports and deletes keys. Modules that learn from ratings start acting after a few liked clips and say so when they need more.

## 9. Features marked experimental

**Settings ▸ Modules** switches each feature on or off. Several features are experimental: built and run, but not yet proven over many runs. If a module fails while generating, the run carries on without it, the Log says why, and the module stays off until its code is repaired or you turn it back on in Settings ▸ Modules.

## If something goes wrong

- Nothing happens: check the ComfyUI chip and the Log window.
- Wrong look: Settings ▸ Appearance. Lost a layout: View ▸ Reset Layout. Lost a project: File ▸ Open recent.
- A bug or a missing feature: please open an [issue](https://github.com/digital-garbage/ComfyUI-FunPack/issues).

## Known limits in 5.0

- LTX 2.x is untested (see above).
- On H3 the scene picture sets the video's shape only; using it as a start frame is not implemented.
- The H3 "second pass" preset is not offered yet: it has not been verified.
