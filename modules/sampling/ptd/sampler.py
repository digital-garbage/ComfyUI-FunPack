"""PTD's fixed-hop sampler: one trained expert set per hop, Euler on the shared-clock velocity, state in fp32.

The recurrence follows ComfyUI-PTD (EPFL VITA Lab, Apache-2.0); the console bar, VRAM housekeeping between
hops and the retry without ComfyUI's allocation compiler follow ComfyUI-PTD-Compact (Nynxz, Apache-2.0).
"""

import time

import torch

from ..._core import log
from . import schedule


def _streams(value):
    if getattr(value, "is_nested", False):
        streams = tuple(value.unbind())
        if len(streams) != 2:
            raise ValueError("PTD expects exactly a picture and a sound stream")
        return streams
    return (value,)


def _map(value, op):
    result = [op(s) for s in _streams(value)]
    return type(value)(result) if getattr(value, "is_nested", False) else result[0]


def _pair(left, right, op):
    a, b = _streams(left), _streams(right)
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        raise ValueError("The model output must keep the latent's streams and shapes")
    result = [op(x, y) for x, y in zip(a, b)]
    return type(left)(result) if getattr(left, "is_nested", False) else result[0]


def _refuse_unsupported(model, extra_args):
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("PTD cannot work on a masked or partly denoised latent: it always starts from pure noise")
    start = getattr(model, "latent_image", None)
    if start is not None and any(bool(torch.count_nonzero(s)) for s in _streams(start)):
        raise ValueError("PTD starts from an empty latent; give the picture as a reference instead of as the "
                         "starting latent (no second pass, no image-to-image)")


def _between_hops(model, hop):
    """Give PyTorch's cached activation memory back after a hop and log where the GPU memory is.

    A hop's activations are freed when it ends but stay cached, and with DynamicVRAM the base model's
    weights are evicted to make room; raising the watermark again lets them fault back in before the
    next hop. Housekeeping only: never stops a run."""
    try:
        if not torch.cuda.is_available():
            return
        before = torch.cuda.memory_reserved() / 2**30
        torch.cuda.empty_cache()
        patcher = getattr(getattr(model, "inner_model", None), "model_patcher", None)
        vbar = patcher._vbar_get() if hasattr(patcher, "_vbar_get") else None
        if vbar is not None:
            import comfy.model_management
            comfy.model_management.reset_cast_buffers()
            vbar.set_watermark(vbar.max_size)
        free, total = torch.cuda.mem_get_info()
        resident = patcher.loaded_size() / 2**30 if patcher is not None else float("nan")
        log.info("FunPack PTD", f"hop {hop} memory: model resident {resident:.1f} GB; cache {before:.1f} -> "
                 f"{torch.cuda.memory_reserved() / 2**30:.1f} GB; card {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB used")
    except Exception as exc:       # noqa: BLE001
        log.once("ptd:memory", log.WARNING, "FunPack PTD", f"could not release cached memory between hops: {exc}")


@torch.no_grad()
def _run(model, x, extra_args, controller, step):
    state = _map(model.noise, lambda s: (s.to(torch.float64) * schedule.T_MAX).to(torch.float32)) \
        if getattr(model, "noise", None) is not None else _map(x, lambda s: s.float())
    _pair(state, x, lambda a, b: a)
    first = _streams(state)[0]
    grid = controller.grid.to(first.device)
    for hop in range(controller.hops):
        time_now = float(grid[hop])
        with controller.activate(hop, time_now):
            denoised = model(state, grid[hop].expand(first.shape[0]), **extra_args)
        velocity = _pair(state, denoised, lambda s, d: (s.float() - d.float()) / time_now)
        step(hop, {"x": state, "i": hop, "sigma": grid[hop], "sigma_hat": grid[hop], "denoised": denoised})
        delta = (grid[hop + 1] - grid[hop]).float()
        state = _pair(state, velocity, lambda s, v: s.float() + delta * v.float())
    return state


def sample_ptd(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller):
    """A KSAMPLER function. `sigmas` must be the adapter's own hop times."""
    from tqdm.auto import tqdm
    expected = controller.grid
    if sigmas.shape != expected.shape or not torch.allclose(sigmas.detach().cpu().double(), expected, atol=2e-6):
        raise ValueError(f"PTD runs only the {controller.hops} hops it was trained on; its own times are "
                         f"{', '.join(f'{float(t):.3f}' for t in expected)}")
    extra_args = {} if extra_args is None else dict(extra_args)
    _refuse_unsupported(model, extra_args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()      # whatever the last node cached is VRAM the allocation compiler cannot get
    bar = tqdm(total=controller.hops, disable=disable)
    done = [0]

    def step(hop, record):
        done[0] += 1
        bar.update(1)
        _between_hops(model, hop)
        if callback is not None:
            callback(record)

    started = time.perf_counter()
    try:
        try:
            out = _run(model, x, extra_args, controller, step)
        except RuntimeError as exc:
            if "aimdo memory compile error" not in str(exc) or not controller.allocation_compiler:
                raise
            # A known intermittent ComfyUI failure when VRAM is nearly full. Drop the half-recorded graph and
            # sample this run again without the compiler; the hops are deterministic, so nothing changes.
            log.warning("FunPack PTD", f"ComfyUI's allocation compiler failed at hop {done[0]} ({exc}); "
                        "sampling this run again without it")
            import comfy.model_prefetch
            comfy.model_prefetch.cleanup_prefetch_queues()
            controller.allocation_compiler = False
            bar.reset()
            done[0] = 0
            started = time.perf_counter()
            try:
                out = _run(model, x, extra_args, controller, step)
            finally:
                controller.allocation_compiler = True
    finally:
        bar.close()
    seconds = time.perf_counter() - started
    log.info("FunPack PTD", f"{controller.hops} hops in {seconds:.1f} s ({seconds / controller.hops:.2f} s/hop)")
    return out
