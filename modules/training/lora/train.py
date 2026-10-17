"""The training run: the set's media become latents, its captions become conditioning, ComfyUI's own trainer does the learning.

Blocking, and run on ComfyUI's worker. Everything that needs torch or comfy is imported inside the
functions, so the rest of the module (sets, tags, the editor's routes) loads without them.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..._core import log
from . import captions, jobs, media_io, store

POSITIVE = ("own", "liked")
NEGATIVE = ("disliked", "bad_composition", "bad_image")
USES = {"normal": "liked and your own media", "avoid": "disliked results, to train an 'avoid' LoRA", "all": "everything in the set"}
ALGORITHMS = ("LoRA", "LoHa", "LoKr", "OFT")


class Stopped(Exception):
    """The person pressed Stop."""


def pick(doc: dict, use: str) -> list[dict]:
    """The items a run trains on. 'normal' leaves out what was disliked, 'avoid' keeps only that, 'all' keeps everything."""
    keep = {"normal": POSITIVE, "avoid": NEGATIVE, "all": POSITIVE + NEGATIVE}.get(use)
    if keep is None:
        raise ValueError(f"use is one of {sorted(USES)}, not {use!r}")
    return [i for i in doc["items"] if i["label"] in keep]


def roots() -> dict:
    import folder_paths
    return {"output": folder_paths.get_output_directory(), "temp": folder_paths.get_temp_directory(),
            "input": folder_paths.get_input_directory()}


def bin_path(media_id: str):
    from ..._core import media
    try:
        return media.path_for(media_id)
    except Exception:                                    # noqa: BLE001 -- a bin entry that is gone is a missing file, said by the caller
        return None


def plan_items(set_name: str, use: str):
    """(items with their files, problems). Nothing here touches torch."""
    doc = store.get(set_name)
    if doc is None:
        return [], [f"there is no training set called {set_name!r}"]
    problems, found = [], []
    chosen = pick(doc, use)
    if not chosen:
        problems.append(f"the set {set_name!r} has nothing to train on for 'use: {use}' ({USES[use]})")
    for item in chosen:
        path = store.resolve(item["src"], roots(), bin_path)
        if path is None:
            problems.append(f"{item['name'] or item['id']} is no longer there")
            continue
        found.append((item, path))
    return found, problems


def _encode_text(clip, text: str):
    tokens = clip.tokenize(text)
    return clip.encode_from_tokens_scheduled(tokens)


def _encode_pixels(vae, pixels):
    """Pixels [N,H,W,3] -> latent tensor, in the VAE's own layout."""
    return vae.encode(pixels[:, :, :, :3])


def encode_item(item, path, size, cfg, vae, adapt, audio_vae):
    """One item's latents. For video models a clip stays a clip (cfg['frames'] frames); for picture models a clip gives one frame."""
    import torch
    width, height = size
    if adapt and adapt.get("video"):
        frames = adapt["frames"](cfg["frames"])
        if item["media_kind"] == "video":
            pixels, rate = media_io.clip_frames(path, frames, cfg["fps"] or adapt.get("fps", 24), cfg["start"], width, height)
        else:
            from PIL import Image, ImageOps
            with Image.open(path) as img:
                still = media_io.cover(ImageOps.exif_transpose(img).convert("RGB"), width, height)
            pixels, rate = media_io.to_tensor([still] * frames), float(adapt.get("fps", 24))
        video = _encode_pixels(vae, pixels)
        if video.ndim == 4:                               # a 2D VAE gave a stack of frames: [T,C,h,w] -> [1,C,T,h,w]
            video = video.permute(1, 0, 2, 3)[None]
        sound = None
        if audio_vae is not None and adapt.get("silence"):
            sound = _sound_latent(path, item["media_kind"], frames / rate, audio_vae, adapt)
        return video, sound
    img = media_io_picture(path, item["media_kind"], width, height)
    return _encode_pixels(vae, media_io.to_tensor([img])), None


def media_io_picture(path, kind, width, height):
    from . import frames
    return media_io.cover(frames.picture(path), width, height)


def _sound_latent(path, kind, seconds, audio_vae, adapt):
    import comfy.audio
    wave = media_io.read_audio(path, seconds) if kind == "video" else None
    if wave is None:
        return adapt["silence"](audio_vae, seconds)
    waveform, rate = wave
    target = int(getattr(audio_vae, "audio_sample_rate", 44100))
    if rate != target:
        waveform = comfy.audio.resample(waveform, rate, target)
    return audio_vae.encode(waveform.movedim(1, -1))


def run(model, clip, vae, audio_vae, set_name: str, cfg: dict, prompt_id: str = "") -> str:
    """Train and save. Returns the saved LoRA's file name inside the loras folder. Raises with a reason a person can act on."""
    import torch
    import comfy.model_management as mm
    import comfy_extras.nodes_train as nt
    import folder_paths
    import safetensors.torch

    from ..._core import registry
    run_id = jobs.begin(set_name, cfg["name"], cfg["steps"], cfg, prompt_id)        # first, so a refusal below is shown in the window and not only in ComfyUI's log
    try:
        adapt = registry.current().ask("training_adapt", model)
        items, problems = plan_items(set_name, cfg["use"])
        if problems and not items:
            raise ValueError("; ".join(problems))
        if problems:
            log.warning("FunPack LoRA training", f"left out of the run: {'; '.join(problems[:5])}{' ...' if len(problems) > 5 else ''}")
        if adapt is None and getattr(model, "cached_patcher_init", None) is None:
            raise ValueError("this model was loaded in a way the trainer cannot copy; load it with the FunPack Diffusion Model Loader from a plain file")
        batch = cfg["batch_size"]
        if adapt and batch > adapt.get("batch_limit", batch):
            log.warning("FunPack LoRA training", f"{adapt['name']} trains one sample at a time; batch size {batch} is lowered to {adapt['batch_limit']}")
            batch = adapt["batch_limit"]
        jobs.say(f"Reading {len(items)} items")
        step = (adapt or {}).get("step") or 64
        sizes = []
        for item, path in items:
            try:
                sizes.append(media_io.size_of(path, item["media_kind"]))
            except Exception as exc:                      # noqa: BLE001
                raise ValueError(f"{item['name'] or item['id']} cannot be read: {exc}") from exc
        single = bool(adapt) or not cfg["aspect_buckets"]
        buckets = media_io.buckets_for([w / h for w, h in sizes], cfg["resolution"], step, single)
        trigger = cfg["trigger"]
        texts = [captions.with_trigger(item["caption"], trigger) for item, _ in items]

        videos, sounds, conds = [], [], []
        with torch.inference_mode():
            for number, ((item, path), size, text) in enumerate(zip(items, buckets, texts), 1):
                if jobs.stop_requested():
                    raise Stopped()
                jobs.say(f"Preparing {number} of {len(items)}")
                latent, sound = encode_item(item, path, size, cfg, vae, adapt, audio_vae)
                videos.append(latent.clone())
                sounds.append(None if sound is None else sound.clone())
                conds.append(_encode_text(clip, text))
        if adapt and adapt.get("pack") and audio_vae is not None and all(s is not None for s in sounds):
            latents = adapt["pack"](videos, sounds)
            payload = [latents]
        elif adapt and adapt.get("pack"):
            raise ValueError("this model trains picture and sound together and needs its audio VAE: add the audio VAE loader to the pipeline")
        elif len({tuple(v.shape) for v in videos}) == 1:
            payload = [{"samples": torch.cat(videos, 0)}]
        else:
            payload = [{"samples": v} for v in videos]

        losses = _loss_hook(nt, cfg["steps"] * cfg["grad_accumulation"])
        jobs.say("Training", "training")
        shim = adapt["shims"]() if adapt and adapt.get("shims") else _nothing()
        with shim, losses:
            out = nt.TrainLoraNode.execute(
                model=[model], latents=payload, positive=conds, batch_size=[batch], steps=[cfg["steps"]],
                grad_accumulation_steps=[cfg["grad_accumulation"]], learning_rate=[cfg["learning_rate"]], rank=[cfg["rank"]],
                optimizer=[cfg["optimizer"]], loss_function=[cfg["loss"]], seed=[cfg["seed"]], training_dtype=[cfg["training_dtype"]],
                lora_dtype=[cfg["lora_dtype"]], quantized_backward=[False], algorithm=[cfg["algorithm"]],
                gradient_checkpointing=[cfg["gradient_checkpointing"]], checkpoint_depth=[1], offloading=[cfg["offloading"]],
                existing_lora=["[None]"], bucket_mode=[False], bypass_mode=[cfg["bypass"]])
        lora = (out.result if hasattr(out, "result") else out)[0]
        file = _save(lora, cfg["name"], folder_paths, safetensors.torch, set_name, cfg)
        jobs.finish("done", lora=file)
        return file
    except Stopped:
        jobs.finish("stopped")
        raise
    except mm.InterruptProcessingException:
        jobs.finish("stopped")
        raise
    except Exception as exc:                              # noqa: BLE001 -- recorded for the window, then raised for ComfyUI to show
        jobs.finish("error", error=f"{type(exc).__name__}: {exc}")
        raise


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _loss_hook:
    """While open, every step's loss goes to the run record, and Stop (or ComfyUI's interrupt) ends the loop at the next step."""

    def __init__(self, nt, total):
        self.nt, self.total = nt, total

    def __enter__(self):
        import comfy.model_management as mm
        nt = self.nt
        self.original = nt.TrainSampler

        def sampler(*args, **kwargs):
            inner = kwargs.get("loss_callback")

            def callback(loss):
                if inner:
                    inner(loss)
                jobs.step(float(loss))
                mm.throw_exception_if_processing_interrupted()
                if jobs.stop_requested():
                    raise Stopped()
            kwargs["loss_callback"] = callback
            return self.original(*args, **kwargs)

        nt.TrainSampler = sampler
        return self

    def __exit__(self, *exc):
        self.nt.TrainSampler = self.original
        return False


def _save(lora, name, folder_paths, safetensors_torch, set_name, cfg) -> str:
    """Into the first loras folder, as <name>.safetensors, or <name>_2 and so on rather than over an earlier one."""
    folder = Path(folder_paths.get_folder_paths("loras")[0])
    folder.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in name if c.isalnum() or c in "-_ .") .strip() or "funpack_lora"
    n, path = 1, folder / f"{stem}.safetensors"
    while path.exists():
        n += 1
        path = folder / f"{stem}_{n}.safetensors"
    meta = {"software": "ComfyUI-FunPack", "training_set": set_name, "algorithm": cfg["algorithm"], "rank": str(cfg["rank"]),
            "steps": str(cfg["steps"]), "trigger": cfg["trigger"]}
    safetensors_torch.save_file(lora, str(path), metadata=meta)
    return path.name
