"""The text encoder loader.

One node, up to four files, one CLIP. Unlike the LoRA loader this cannot be a
chain: the files are combined into a single encoder, so they have to arrive
together. One slot for LTX-2.5 (Gemma4); two for LTX-2.3 (Gemma3 plus its
connector).

The slots are numbered combos, which is what core's DualCLIPLoader and
TripleCLIPLoader use. Two alternatives were tried and rejected:

* **ComfyUI's autogrow input** looks like the right answer and is not.
  `_AutogrowTemplate.__init__` sets `force_input = True` on any WidgetInput, so a
  Combo template becomes a row of SOCKETS instead of dropdowns -- the file picker
  stops being a picker. Every core usage of autogrow is a connection type (Image,
  Audio, Splat), never a Combo.
* **v4's `funpack_list`**, a JSON array in a STRING widget, needed a hand-written
  canvas widget to draw it, and without one it renders as raw JSON. The file it
  documented (`web/funpack_list.js`) did not even exist, so on the stock canvas
  the list was raw JSON in v4 too.

The cost of numbered slots is a fixed ceiling. Four is above anything shipping
(LTX-2.3 needs two), and raising it later only adds optional inputs, which is
backwards-compatible with every saved graph.
"""

import comfy.sd
import comfy.utils
import folder_paths
import torch
from comfy_api.latest import io

from ..._core import log
from .. import gguf_support

MAX_ENCODERS = 4


def clip_types():
    try:
        return [t.name.lower() for t in comfy.sd.CLIPType]
    except Exception:                            # noqa: BLE001
        return ["stable_diffusion"]


def encoder_files():
    """Text encoder files, `.gguf` included. Same reasoning as the diffusion
    model loader's own file-list helper."""
    return list(folder_paths.get_filename_list("text_encoders")) + \
        gguf_support.gguf_names("text_encoders")


class FunPackCLIPLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        types = clip_types()
        files = encoder_files()
        default_type = "ltxv" if "ltxv" in types else (types[0] if types else "stable_diffusion")

        slots = [io.Combo.Input("clip_name1", options=files,
                                tooltip="Text encoder file. The first slot is the encoder itself.")]
        # Optional, and "" means "no file here" -- an empty slot must not be a
        # validation failure, because most families only need one.
        slots += [
            io.Combo.Input(f"clip_name{n}", options=[""] + files, default="", optional=True,
                           tooltip="Another encoder file, loaded after the ones above. "
                                   "Leave empty if the family needs no more.")
            for n in range(2, MAX_ENCODERS + 1)
        ]

        return io.Schema(
            node_id="FunPackCLIPLoader",
            display_name="FunPack CLIP Loader",
            category="FunPack/Loaders",
            description="Load one or more text encoder files into a single CLIP.",
            inputs=[
                *slots,
                io.Combo.Input("type", options=types, default=default_type,
                               tooltip="Which family the encoder is for. LTX-2 is 'ltxv'; "
                                       "MiniMax H3 is 'minimax'."),
                io.Combo.Input("device", options=["default", "cpu"], default="default",
                               optional=True,
                               tooltip="'cpu' keeps the encoder off the GPU: slower prompts, "
                                       "more VRAM for the diffusion model."),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, clip_name1: str, type: str, device: str = "default", **slots) -> io.NodeOutput:
        # Order matters to load_clip, so the slots are read in slot order rather
        # than in whatever order kwargs arrive.
        names = [clip_name1] + [slots.get(f"clip_name{n}") for n in range(2, MAX_ENCODERS + 1)]
        names = [name for name in names if name]
        if not names:
            raise RuntimeError("FunPack CLIP Loader: pick at least one text encoder file.")

        # Refuse rather than repair. Falling back to STABLE_DIFFUSION loads the
        # encoder against the wrong family and reports success, and a family
        # mismatch in this project reads as an unrelated phantom fault rather
        # than as a family error. Reachable when CLIPType renames a member that
        # a saved workflow still names.
        clip_type = getattr(comfy.sd.CLIPType, str(type).upper(), None)
        if clip_type is None:
            known = ", ".join(sorted(t.name.lower() for t in comfy.sd.CLIPType))
            raise RuntimeError(
                f"FunPack CLIP Loader: this ComfyUI has no encoder family {type!r}. "
                f"Known families: {known}.")
        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        # Each name resolved to a path exactly once, whatever happens next --
        # a second lookup per file would double every filesystem stat this
        # loader makes. GGUF-ness is then read off that same path: by
        # extension, or by container magic for a renamed file.
        paths = {}
        for n in names:
            if gguf_support.is_gguf(n):
                p = gguf_support.gguf_path("text_encoders", n)
                if not p:
                    # Listed when the combo was built, gone by execute time --
                    # reachable now that the pipeline window can sit open for a
                    # while and re-list files without reloading. Refuse by name
                    # here, or the loader below gets None and fails deep inside
                    # gguf/pack machinery with no mention of which file it was.
                    raise RuntimeError(f"{n} is no longer where it was listed from.")
                paths[n] = p
            else:
                paths[n] = folder_paths.get_full_path_or_raise("text_encoders", n)
        misnamed = {n for n, p in paths.items()
                    if not gguf_support.is_gguf(n) and gguf_support.has_gguf_magic(p)}

        gguf_notes = []
        if misnamed or any(gguf_support.is_gguf(n) for n in names):
            # A .gguf encoder cannot go through load_clip(), which reads files
            # itself. Every slot becomes a state dict instead, so a GGUF and a
            # .safetensors connector can sit in the same list -- the normal
            # LTX-2.3 shape.
            state_dicts = []
            for n in names:
                if gguf_support.is_gguf(n) or n in misnamed:
                    if n in misnamed:
                        gguf_notes.append(f"{n} is named .safetensors but is a GGUF container")
                    sd, gopts, gnote = gguf_support.load_clip_state_dict(paths[n])
                    state_dicts.append(sd)
                    model_options = {**model_options, **gopts}
                    gguf_notes.append(f"{n}: {gnote}")
                else:
                    state_dicts.append(comfy.utils.load_torch_file(paths[n]))
            clip = comfy.sd.load_text_encoder_state_dicts(
                state_dicts,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )
        else:
            clip = comfy.sd.load_clip(
                ckpt_paths=[paths[n] for n in names],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type,
                model_options=model_options,
            )
        if clip is None:
            # comfy returns None for an unrecognised encoder rather than raising,
            # and a None CLIP fails at encode time, far from the mistake.
            raise RuntimeError(
                f"Could not load a text encoder from {', '.join(names)} as type {type!r}.")

        log.info("FunPack CLIP Loader",
                 f"{len(names)} encoder file(s) loaded as type {type}, device {device}: "
                 + ", ".join(names))
        listing = "\n".join(f"  {i + 1}. {name}" for i, name in enumerate(names))
        listing += "".join(f"\n  {gn}" for gn in gguf_notes)
        return io.NodeOutput(clip, f"FunPack CLIP Loader | type={type} device={device}\n{listing}")
