"""The training run: the set's media become latents, its captions become conditioning, ComfyUI's own trainer does the learning.

Blocking, and run on ComfyUI's worker. Everything that needs torch or comfy is imported inside the
functions, so the rest of the module (sets, tags, the editor's routes) loads without them.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..._core import log
from . import captions, jobs, loop, media_io, store, swap

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


def _fingerprint(obj) -> str:
    """Tells one set of VAE or text-encoder weights from another cheaply: class, size and a few values."""
    import hashlib
    if obj is None:
        return ""
    inner = getattr(obj, "first_stage_model", None) or getattr(obj, "cond_stage_model", None) or getattr(obj, "model", None) or obj
    h = hashlib.sha256(type(inner).__name__.encode())
    try:
        params = list(inner.parameters())
        h.update(str(sum(p.numel() for p in params)).encode())
        for p in (params[0], params[-1]) if params else ():
            h.update(p.detach().flatten()[:16].float().cpu().numpy().tobytes())
    except Exception:                                      # noqa: BLE001 -- an object with no parameters: its class has to do
        pass
    return h.hexdigest()[:16]


def _pair(result):
    return result[0].clone(), None if result[1] is None else result[1].clone()


def _cached(kind, parts, make, torch):
    """What an item encodes to, kept on disk for the next run: same file, same settings, same weights give the same answer. Returns (value, was_cached)."""
    import hashlib
    import json
    from ..._core import config
    folder = config.TRAINING_DIR / "cache"
    path = folder / f"{kind}-{hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:24]}.pt"
    if path.is_file():
        try:
            return torch.load(path, weights_only=False), True
        except Exception:                                  # noqa: BLE001 -- a damaged or foreign-device file: encode again
            pass
    value = make()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(value, tmp)
        tmp.replace(path)
    except Exception as exc:                               # noqa: BLE001 -- the cache is only a speed-up
        log.warning("FunPack LoRA training", f"could not keep {kind} in the cache: {exc}")
    return value, False


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
    """Train and save. Returns the saved LoRA's file name inside the loras folder. Raises with a reason a person can act on.

    A failed run lets go of everything it held: ComfyUI keeps the error (and so every frame's locals, the model copy
    among them) alive after it is shown, and the next run then starts with that much of the card already taken.
    """
    import gc
    import traceback
    import torch
    import comfy.model_management as mm
    gc.collect()                                          # what an earlier failed run left behind in reference cycles
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _card("at the start of the run", mm, torch)
    try:
        result = _run(model, clip, vae, audio_vae, set_name, cfg, prompt_id)
        model = clip = vae = audio_vae = None             # a paused or finished run lets the card go: the pipeline's preview or the next run needs it
        _free_the_card(mm, torch)
        return result
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        model = clip = vae = audio_vae = None
        _free_the_card(mm, torch)
        raise


def _run(model, clip, vae, audio_vae, set_name: str, cfg: dict, prompt_id: str = "") -> str:
    import torch
    import comfy.model_management as mm
    import comfy_extras.nodes_train as nt
    import folder_paths
    import safetensors.torch

    from ..._core import registry
    resume, failed = None, None
    if cfg.get("resume"):
        try:
            resume = loop.load_checkpoint(cfg["resume"], torch)
        except ValueError as exc:
            failed = exc
    run_id = jobs.begin(set_name, cfg["name"], cfg["steps"], cfg, prompt_id, {"run": cfg["resume"], "loss": resume["loss"]} if resume else None)        # first, so a refusal below is shown in the window and not only in ComfyUI's log
    try:
        if failed:
            raise failed
        if model is None:
            model = load_model(cfg.get("model_name"), cfg.get("weight_dtype", "default"), cfg.get("compute_dtype", "default"))
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
        texts = [captions.with_parts(item["caption"], cfg["trigger"], cfg.get("postfix", "")) for item, _ in items]

        if adapt and adapt.get("video"):
            cfg = _fit_frames(cfg, adapt, items)
        videos, sounds, conds = [], [], []
        hits = 0
        with torch.inference_mode():
            vae_id, audio_id, clip_id = _fingerprint(vae), _fingerprint(audio_vae), _fingerprint(clip)
            for number, ((item, path), size, text) in enumerate(zip(items, buckets, texts), 1):
                if jobs.stop_requested():
                    raise Stopped()
                jobs.say(f"Preparing {number} of {len(items)}")
                shape = [str(Path(path).resolve()), Path(path).stat().st_mtime_ns, Path(path).stat().st_size, item["media_kind"], list(size), cfg["frames"], cfg["fps"], cfg["start"],
                         (adapt or {}).get("name"), vae_id, audio_id if audio_vae is not None else None]
                (latent, sound), hit = _cached("latent", shape, lambda: _pair(encode_item(item, path, size, cfg, vae, adapt, audio_vae)), torch)
                cond, hit_text = _cached("text", [text, clip_id], lambda: _encode_text(clip, text), torch)
                hits += hit and hit_text
                videos.append(latent.clone())
                sounds.append(None if sound is None else sound.clone())
                conds.append(cond)
        log.info("FunPack LoRA training", f"prepared {len(items)} items, {hits} entirely from the cache")
        videos, sounds, conds = _normal(videos), _normal(sounds), _normal(conds)         # made in inference mode: copies a backward pass can use
        if adapt and adapt.get("pack") and audio_vae is not None and all(s is not None for s in sounds):
            latents = adapt["pack"](videos, sounds)
            payload = [latents]
        elif adapt and adapt.get("pack"):
            raise ValueError("this model trains picture and sound together and needs its audio VAE: add the audio VAE loader to the pipeline")
        elif len({tuple(v.shape) for v in videos}) == 1:
            payload = [{"samples": torch.cat(videos, 0)}]
        else:
            payload = [{"samples": v} for v in videos]

        _sizes(payload, conds)
        _card("before freeing", mm, torch)
        del vae, clip, audio_vae                          # their work is done; the trainer copies the model next and needs the room
        _free_the_card(mm, torch)
        _card("after freeing", mm, torch)
        _train_on_a_plain_copy(model)
        jobs.say("Training", "training")
        shim = adapt["shims"]() if adapt and adapt.get("shims") else _nothing()
        watch = _watch(nt, mm, torch, _swapper(adapt, cfg, mm, torch))
        if adapt and adapt.get("lora_targets"):
            def keep(state, step):                        # a copy of the LoRA so far, in the loras folder, never over an earlier file
                name = _save(state, f"{cfg['name']}_step{step}", folder_paths, safetensors.torch, set_name, cfg)
                log.info("FunPack LoRA training", f"saved {name} at step {step}")
            lora, paused = loop.train(model, payload, conds, cfg, adapt, nt, mm, torch, watch, shim, resume, run_id, keep)       # our own loop, ComfyUI carries the batches
            if paused:
                preview = _save(lora, f"{cfg['name']}_preview", folder_paths, safetensors.torch, set_name, cfg, keep=True)
                jobs.pause(paused.reason, paused.step, preview)
                log.info("FunPack LoRA training", f"paused at step {paused.step} ({paused.reason}); the LoRA so far is {preview}")
                return ""
        else:
            losses = _loss_hook(nt, cfg["steps"] * cfg["grad_accumulation"])
            with shim, losses, watch:
                out = nt.TrainLoraNode.execute(
                    model=[model], latents=payload, positive=conds, batch_size=[batch], steps=[cfg["steps"]],
                    grad_accumulation_steps=[cfg["grad_accumulation"]], learning_rate=[cfg["learning_rate"]], rank=[cfg["rank"]],
                    optimizer=[cfg["optimizer"]], loss_function=[cfg["loss"]], seed=[cfg["seed"]], training_dtype=[cfg["training_dtype"]],
                    lora_dtype=[cfg["lora_dtype"]], quantized_backward=[False], algorithm=[cfg["algorithm"]],
                    gradient_checkpointing=[cfg["gradient_checkpointing"]], checkpoint_depth=[(adapt or {}).get("checkpoint_depth", 1)], offloading=[cfg["offloading"]],
                    existing_lora=["[None]"], bucket_mode=[False], bypass_mode=[cfg["bypass"]])
            lora = (out.result if hasattr(out, "result") else out)[0]
        file = _save(lora, cfg["name"], folder_paths, safetensors.torch, set_name, cfg)
        discard(run_id, cfg["name"])
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


_WEIGHT = {"fp8_e4m3fn": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2", "fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def load_model(name, weight_dtype="default", compute_dtype="default"):
    """The model file, read once and never moved to the card here. The trainer's own copy shares it, so one model is in memory."""
    import torch
    import comfy.sd
    import folder_paths
    if not name:
        raise ValueError("no model to train on: the pipeline's model loader names no file")
    if str(name).lower().endswith(".gguf"):
        raise ValueError("a GGUF model cannot be trained: pick a .safetensors file in the model loader")
    path = folder_paths.get_full_path_or_raise("diffusion_models", name)
    options = {}
    if weight_dtype in _WEIGHT:
        options["dtype"] = getattr(torch, _WEIGHT[weight_dtype])
    with torch.inference_mode(False):                    # ComfyUI runs nodes in inference mode; weights made there cannot take part in a backward pass
        model = comfy.sd.load_diffusion_model(path, model_options=options, disable_dynamic=True)
    if compute_dtype in _WEIGHT and hasattr(model, "set_model_compute_dtype"):
        model.set_model_compute_dtype(getattr(torch, _WEIGHT[compute_dtype]))
    model.cached_patcher_init = (lambda *args, **kwargs: model, ())      # the trainer's "fresh copy" is this model: no second read of the file
    return model


def _normal(value):
    """Tensors made in inference mode, copied so autograd may save them; lists, tuples and dicts are walked."""
    import torch
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.clone() if value.is_inference() else value
    if isinstance(value, list):
        return [_normal(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_normal(v) for v in value)
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    return value


def _fit_frames(cfg, adapt, items):
    """Every clip is read for the same number of frames, so the shortest clip sets the limit: fewer than asked is used, and said."""
    rate = cfg["fps"] or adapt.get("fps", 24)
    shortest = min((media_io.frames_at(path, rate) for item, path in items if item["media_kind"] == "video"), default=None)
    wanted = adapt["frames"](cfg["frames"])
    if shortest is None or shortest >= wanted:
        return cfg
    fit = adapt["fit"](shortest) if adapt.get("fit") else shortest
    if fit < 1:
        raise ValueError(f"the shortest clip has only {shortest} frames at {rate:g} fps: too short to train on")
    log.warning("FunPack LoRA training", f"the shortest clip gives {shortest} frames at {rate:g} fps, so every clip is read for {fit} frames and not {wanted}")
    return {**cfg, "frames": fit}


def _sizes(payload, conds):
    """One Log line: the shape of the first item's latent and of its text, to tell a run that is too big from one that is not."""
    def shape(x):
        tensors = getattr(x, "tensors", None)
        if tensors is not None:
            return " + ".join(str(list(t.shape)) for t in tensors)
        return str(list(x.shape)) if hasattr(x, "shape") else type(x).__name__
    try:
        first = payload[0]["samples"] if isinstance(payload[0], dict) else payload[0]
        text = conds[0][0] if isinstance(conds[0], (list, tuple)) else conds[0]
        log.info("FunPack LoRA training", f"{len(conds)} text(s); first latent {shape(first)}, first text {shape(text)}")
    except Exception as exc:                              # noqa: BLE001 -- a report must never stop a run
        log.info("FunPack LoRA training", f"sizes: not readable ({type(exc).__name__})")


def _card(when, mm, torch):
    """One line in the Log: what the card holds and which models ComfyUI still has loaded."""
    try:
        device = mm.get_torch_device()
        gib = 1024 ** 3
        loaded = ", ".join(f"{type(getattr(m.model, 'model', m.model)).__name__} {m.model_loaded_memory() / gib:.1f} GiB" for m in mm.current_loaded_models) or "none"
        line = f"card {when}: tensors {torch.cuda.memory_allocated(device) / gib:.1f} GiB, free {mm.get_free_memory(device) / gib:.1f} GiB, loaded models: {loaded}"
        jobs.trail(line)
    except Exception as exc:                              # noqa: BLE001 -- a report must never stop a run
        log.info("FunPack LoRA training", f"card {when}: not readable ({type(exc).__name__})")


class _watch:
    """While open, a Log line after each stage of the trainer's setup, so an out-of-memory says which stage filled the card."""
    STAGES = ("_prepare_latents_and_count", "_setup_lora_adapters", "_setup_lora_adapters_bypass", "_create_optimizer")

    def __init__(self, nt, mm, torch, after_load=None):
        self.nt, self.mm, self.torch, self.put, self.after_load = nt, mm, torch, [], after_load
        self.state = {"backward": False, "recomputes": 0}

    def __enter__(self):
        def wrap(owner, name, label):
            original = getattr(owner, name, None)
            if original is None:
                return

            def watched(*args, **kwargs):
                if label == "loading the model":
                    jobs.trail("loading the model onto the card now")
                out = original(*args, **kwargs)
                if label == "loading the model" and self.after_load:
                    self.after_load(args[0] if args else kwargs.get("models"))
                _card(f"after {label}", self.mm, self.torch)
                return out
            setattr(owner, name, watched)
            self.put.append((owner, name, original))
        for name in self.STAGES:
            wrap(self.nt, name, name.strip("_").replace("_", " "))
        wrap(self.mm, "load_models_gpu", "loading the model")
        self._backward()
        self._recomputes()
        return self

    def _recomputes(self):
        """A Log line for the first block recomputes of the backward pass: how the card fills block by block."""
        nt, mm, torch, state = self.nt, self.mm, self.torch, self.state
        original, counter = nt.patch, [0]

        def patch(m, offloading=False):
            idx, forward = counter[0], getattr(m, "forward", None)
            counter[0] += 1
            if forward is not None:
                def logged(*args, **kwargs):
                    if state["backward"]:
                        state["recomputes"] += 1
                        n = state["recomputes"]
                        if n <= 8 or n % 10 == 0:
                            _card(f"backward recompute {n} (module {idx}, peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB)", mm, torch)
                    return forward(*args, **kwargs)
                m.forward = logged
            return original(m, offloading)
        nt.patch = patch
        self.put.append((nt, "patch", original))

    def _backward(self):
        """A Log line just before the first backward passes: what the forward pass left on the card (the part that grows with clip length)."""
        autograd, torch, seen = self.torch.autograd, self.torch, [0]
        original = autograd.backward

        def watched(*args, **kwargs):
            self.state["backward"] = True
            if seen[0] < 2:
                seen[0] += 1
                _card(f"before backward pass {seen[0]} (peak so far {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB)", self.mm, torch)
            return original(*args, **kwargs)
        autograd.backward = watched
        self.put.append((autograd, "backward", original))

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.put):
            setattr(owner, name, original)
        return False


def _swapper(adapt, cfg, mm, torch):
    """After the model is on the card: the weights of the chosen blocks go to system memory (block swap), or None when not asked for."""
    count = int(cfg.get("swap_blocks") or 0)
    if count <= 0:
        return None
    if not (adapt and adapt.get("blocks")):
        log.warning("FunPack LoRA training", "block swap is asked for, but this model has no block list to swap: the whole model stays on the card")
        return None

    def after(patchers):
        for patcher in patchers or []:
            blocks = adapt["blocks"](patcher.model.diffusion_model)
            done = swap.move(blocks, count, torch)
            if done["blocks"]:
                log.info("FunPack LoRA training", f"block swap: {done['blocks']} of {len(blocks)} blocks ({done['bytes'] / 1024 ** 3:.1f} GiB) now wait in system memory")
                _card("after block swap", mm, torch)
            elif not swap.still_away(blocks, count):
                log.warning("FunPack LoRA training", "block swap: ComfyUI moved the weights back to the card")
    return after


def _free_the_card(mm, torch):
    import gc
    gc.collect()
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _train_on_a_plain_copy(model):
    """The trainer copies the model with clone(force_deepcopy=True). On a dynamic-VRAM model that copy is dynamic too, and the
    trainer then loads it in full beside the original: two models on the card. Ask for a plain copy, loaded once."""
    is_dynamic = getattr(model, "is_dynamic", None)
    if not (callable(is_dynamic) and is_dynamic()):
        return
    clone = model.clone

    def plain(*args, **kwargs):
        return clone(*args, **{**kwargs, "disable_dynamic": True})
    model.clone = plain


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


def _preview_name(name) -> str:
    stem = "".join(c for c in f"{name}_preview" if c.isalnum() or c in "-_ .").strip() or "funpack_lora_preview"
    return f"{stem}.safetensors"


def discard(run, name) -> None:
    """A run that is over (finished, stopped) keeps neither its saved state nor its preview file."""
    import folder_paths
    loop.forget_checkpoint(run)
    try:
        (Path(folder_paths.get_folder_paths("loras")[0]) / _preview_name(name.removesuffix("_preview"))).unlink()
    except OSError:
        pass


def _save(lora, name, folder_paths, safetensors_torch, set_name, cfg, keep=False) -> str:
    """Into the first loras folder, as <name>.safetensors, or <name>_2 and so on rather than over an earlier one. `keep`: replace the file of that name (the preview)."""
    folder = Path(folder_paths.get_folder_paths("loras")[0])
    folder.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in name if c.isalnum() or c in "-_ .") .strip() or "funpack_lora"
    n, path = 1, folder / f"{stem}.safetensors"
    while path.exists() and not keep:
        n += 1
        path = folder / f"{stem}_{n}.safetensors"
    meta = {"software": "ComfyUI-FunPack", "training_set": set_name, "algorithm": cfg["algorithm"], "rank": str(cfg["rank"]),
            "steps": str(cfg["steps"]), "trigger": cfg["trigger"]}
    safetensors_torch.save_file(lora, str(path), metadata=meta)
    return path.name
