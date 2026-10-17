"""FunPack's own LoRA training loop, for models that tell the module what to train (H3 today).

ComfyUI stays the backend: its model classes and ops, its sampler and model management that carry a
batch through the network. What ComfyUI's trainer decides is done here instead: which layers get a
LoRA, how the blocks are checkpointed, how a step is drawn and weighed, how the optimizer steps.

The approach follows musubi-tuner by kohya-ss (Apache-2.0): LoRA on the attention and feed-forward
projections of the transformer blocks only, a checkpoint around each block, uniform noise levels,
rank = alpha, the learning rate and optimizer it recommends. The code is written for this module, not copied.
"""

from __future__ import annotations

from ..._core import log
from . import jobs


class Paused(Exception):
    """The loop stops at a step boundary and keeps its state: `reason` is user (Pause pressed) or preview (the step asked for)."""

    def __init__(self, reason, step):
        super().__init__(f"paused at step {step}")
        self.reason, self.step = reason, step


def _checkpoint_file(run):
    from ..._core import config
    folder = config.TRAINING_DIR / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{run}.pt"


def save_checkpoint(run, lora, optimizer, step, torch):
    """Everything a resumed run needs besides the data: the LoRA, the optimizer's memory, the loss so far and where it stopped."""
    torch.save({"lora": {n: tuple(p.detach().cpu() for p in pair) for n, pair in lora.pairs.items()}, "optimizer": optimizer.state_dict(),
                "step": int(step), "loss": jobs.losses()}, _checkpoint_file(run))


def load_checkpoint(run, torch):
    """The saved state of a paused run, or a ValueError saying why it cannot be carried on."""
    import re
    if not re.fullmatch(r"[0-9a-f]{12}", str(run)):
        raise ValueError("that is not the name of a paused run")
    path = _checkpoint_file(run)
    if not path.is_file():
        raise ValueError("the paused run's saved state is gone: start the training again")
    return torch.load(path, map_location="cpu", weights_only=False)


def forget_checkpoint(run):
    try:
        _checkpoint_file(run).unlink()
    except OSError:
        pass


class Lora:
    """A low-rank pair per target layer, applied by a forward hook so the layer, its ops and its weight casting stay ComfyUI's own."""

    def __init__(self, targets, rank, alpha, dtype, device, torch):
        self.torch, self.rank, self.alpha = torch, int(rank), float(alpha)
        self.names = [name for name, _ in targets]
        self.pairs, self.handles = {}, []
        nn = torch.nn
        for name, module in targets:
            weight = module.weight
            down = nn.Parameter(torch.empty(self.rank, weight.shape[1:].numel(), dtype=dtype, device=device))
            up = nn.Parameter(torch.zeros(weight.shape[0], self.rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(down, a=5 ** 0.5)
            self.pairs[name] = (down, up)
            self.handles.append(module.register_forward_hook(self._hook(down, up)))

    def _hook(self, down, up):
        scale = self.alpha / self.rank
        F = self.torch.nn.functional

        def add(module, args, out):
            x = args[0]
            delta = F.linear(F.linear(x.to(down.dtype), down), up) * scale
            return out + delta.to(out.dtype)
        return add

    def restore(self, saved):
        """Put a checkpoint's weights back; the layers must be the ones it was saved from."""
        if set(saved) != set(self.pairs):
            raise ValueError("the paused run was made for a different model or LoRA layout: start the training again")
        with self.torch.no_grad():
            for name, (down, up) in self.pairs.items():
                if saved[name][0].shape != down.shape:
                    raise ValueError("the paused run used another rank: start the training again")
                down.copy_(saved[name][0])
                up.copy_(saved[name][1])

    def parameters(self):
        return [p for pair in self.pairs.values() for p in pair]

    def state(self, prefix="diffusion_model."):
        """The file's tensors, named the way ComfyUI loads a LoRA (lora_down / lora_up / alpha per layer)."""
        out = {}
        for name, (down, up) in self.pairs.items():
            out[f"{prefix}{name}.lora_down.weight"] = down.detach().to("cpu").contiguous()
            out[f"{prefix}{name}.lora_up.weight"] = up.detach().to("cpu").contiguous()
            out[f"{prefix}{name}.alpha"] = self.torch.tensor(self.alpha)
        return out

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def checkpoint_modules(modules, torch):
    """Each module runs inside a non-reentrant checkpoint: its activations are dropped after the forward and recomputed in the backward, one module at a time. Returns an undo."""
    from torch.utils.checkpoint import checkpoint
    undo = []
    for module in modules:
        original = module.forward

        def wrapped(*args, __original=original, **kwargs):
            def run(args, kwargs):
                return __original(*args, **kwargs)
            return checkpoint(run, args, kwargs, use_reentrant=False)
        module.forward = wrapped
        undo.append((module, original))

    def restore():
        for module, original in undo:
            module.forward = original
    return restore


def optimizer_for(name, params, lr, torch):
    """AdamW is musubi's recommendation; the other names are the module's older choices."""
    return {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD, "RMSprop": torch.optim.RMSprop}[name](params, lr=lr)


class Sampler:
    """Handed to ComfyUI's guider in place of a sampler: instead of denoising, it trains. Built lazily so this file imports without comfy."""

    @staticmethod
    def make(nt, torch, comfy_samplers, custom_sampler, lora, optimizer, criterion, cfg, dtype, stopped, first=0):
        class OwnSampler(comfy_samplers.Sampler):
            def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
                model_wrap.conds = nt.process_cond_list(model_wrap.conds)
                cond = model_wrap.conds["positive"]
                size = sigmas.size(0)
                accumulate, seed = cfg["grad_accumulation"], cfg["seed"]
                sampling = model_wrap.inner_model.model_sampling
                total = cfg["steps"] * accumulate
                every = -(-int(cfg.get("preview_every") or 0) // accumulate) * accumulate        # a preview waits for a whole optimizer step
                for i in range(first, total):
                    gen = torch.Generator().manual_seed(seed + i)
                    pick = torch.randperm(size, generator=gen)[:1].tolist()
                    latent = torch.stack([latent_image[j] for j in pick])
                    noise_i = custom_sampler.Noise_RandomNoise(seed + i * 1000).generate_noise({"samples": latent}).to(latent.device)
                    # uniform noise level, kept off the very ends where the 1/sigma^2 weight below explodes
                    level = 0.05 + 0.95 * torch.rand((1,), generator=gen).item()
                    sigma = torch.tensor([level]).to(latent.device)
                    jobs.trail(f"step {i + 1}: forward (item {pick[0]}, noise level {level:.2f}, card peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB)" if torch.cuda.is_available() else f"step {i + 1}: forward")
                    xt = sampling.noise_scaling(sigma, noise_i, latent, False)
                    model_wrap.conds["positive"] = [cond[j] for j in pick]
                    batch_extra = nt.make_batch_extra_option_dict(extra_args, pick, full_size=size)
                    with torch.autocast(xt.device.type, dtype=dtype):
                        pred = model_wrap(xt.requires_grad_(True), sigma.requires_grad_(True), **batch_extra)
                        # the network answers with the clean latent; weighing its error by 1/sigma^2 makes it the velocity loss musubi trains on
                        loss = criterion(pred.float(), latent.float()) / (level ** 2)
                    jobs.trail(f"step {i + 1}: backward")
                    (loss / accumulate).backward()
                    value = float(loss.item())
                    del pred, xt, loss
                    jobs.trail(f"step {i + 1}: loss {value:.4f}")
                    stopped(value)
                    if (i + 1) % accumulate == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        if (i + 1) < total:
                            if jobs.pause_requested():
                                raise Paused("user", i + 1)
                            if every and (i + 1) % every == 0:
                                raise Paused("preview", i + 1)
                torch.cuda.empty_cache()
                return torch.zeros_like(latent_image)
        return OwnSampler()


def train(model, payload, conds, cfg, adapt, nt, mm, torch, watch, shim, resume=None, run=None):
    """Train a LoRA on `model` with ComfyUI's guider carrying each batch. Returns the file's tensors."""
    import comfy.samplers
    import comfy_extras.nodes_custom_sampler as custom_sampler
    import node_helpers
    dtype_name = cfg["training_dtype"]
    dtype = node_helpers.string_to_torch_dtype(dtype_name if dtype_name != "none" else "bf16")
    lora_dtype = node_helpers.string_to_torch_dtype(cfg["lora_dtype"])
    with torch.inference_mode(False), shim:
        model = model.clone(force_deepcopy=True)         # the loader made the model's "copy" be the model itself, so this holds one model; made fresh outside inference mode otherwise
        model.set_model_compute_dtype(dtype)
        model.model.requires_grad_(False).train()
        latents, count, multi = nt._prepare_latents_and_count(nt._process_latents_standard_mode(payload), dtype, False)
        positive = nt._validate_and_expand_conditioning(nt._process_conditioning(conds), count, False)
        diffusion = model.model.diffusion_model
        lora = Lora(adapt["lora_targets"](diffusion), cfg["rank"], cfg["rank"], lora_dtype, model.load_device, torch)
        jobs.trail("model prepared, starting")
        log.info("FunPack LoRA training", f"LoRA on {len(lora.names)} layers, rank {cfg['rank']}, {sum(p.numel() for p in lora.parameters()) / 1e6:.1f}M trainable values")
        optimizer = optimizer_for(cfg["optimizer"], lora.parameters(), cfg["learning_rate"], torch)
        criterion = nt._create_loss_function(cfg["loss"])
        restore = checkpoint_modules(adapt["checkpoint"](diffusion), torch) if cfg["gradient_checkpointing"] else (lambda: None)

        def stopped(loss):
            jobs.step(loss)
            mm.throw_exception_if_processing_interrupted()
            if jobs.stop_requested():
                from .train import Stopped
                raise Stopped()
        first = 0
        if resume:
            lora.restore(resume["lora"])
            optimizer.load_state_dict(resume["optimizer"])
            first = int(resume["step"])
            log.info("FunPack LoRA training", f"carrying on from step {first}")
        sampler = Sampler.make(nt, torch, comfy.samplers, custom_sampler, lora, optimizer, criterion, cfg, dtype, stopped, first)
        if cfg["offloading"]:
            log.warning("FunPack LoRA training", "ComfyUI's own 'keep weights in system memory' is ignored by this training loop: it has brought ComfyUI down on H3. Use 'Blocks kept in system memory' instead")
        guider = nt.TrainGuider(model, offloading=False)
        guider.set_conds(positive)
        paused = None
        try:
            mm.in_training = True
            with watch:
                try:
                    nt._run_training_loop(guider, sampler, latents, count, cfg["seed"], False, multi)
                except Paused as stop:
                    paused = stop
                    save_checkpoint(run, lora, optimizer, stop.step, torch)
        finally:
            mm.in_training = False
            restore()
            state = lora.state() if lora else {}
            lora.remove()
            del optimizer, sampler, guider                # a failed run leaves nothing on the card for the next one
        return state, paused
