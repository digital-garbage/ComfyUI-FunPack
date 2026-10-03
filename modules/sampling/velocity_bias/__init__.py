"""Velocity bias: at the step where the clip's structure forms, rotate the picture toward the
motion your liked clips had at that same step.

Every run records the direction the model was moving the picture at the structure step (the step
nearest 90% of the starting noise). Rating a clip liked banks it. Then, at that step, the current
latent is turned toward the average banked direction, without adding energy (the size is kept), and
with a cap so it never wipes the current clip. A thin overlay of remembered motion: the best use is
crossing a remembered action onto a new prompt, and it can re-insert a cut your liked clips had.

"Like the nearest liked prompt" picks one real clip's motion instead of the average of all.
Banked directions are the size of the whole latent, so a clip is only matched with banked clips of the
same shape (length and frame size); a run of another shape says so and goes unbiased.
Picture only: sound is never touched.
"""

import torch
import torch.nn.functional as F
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams
from ..score_slider import cond_row

ID = "velocity_bias"
TITLE = "Velocity bias"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Remember the motion I liked",
        "hint": "Nudges the clip's structure toward the motion of clips you liked. A thin overlay, not a copy.",
    },
    "strength": {
        "type": "float", "default": 0.15, "min": 0.0, "max": 3.0, "step": 0.05,
        "label": "How much", "ui": "slider",
        "hint": "0 = only learn. Higher turns the clip further toward the remembered motion.",
        "when": {"enabled": True},
    },
    "nearest": {
        "type": "bool", "default": False,
        "label": "Like the nearest liked prompt",
        "hint": "Copy one liked clip's motion (the closest prompt) instead of the average of all.",
        "when": {"enabled": True},
    },
}

KIND = "velocity"
TARGET = 0.90               # the structure step, as a share of the starting noise
WINDOW = 0.065              # how near a step has to be to count as it (v4)
MAX_ROWS = 16               # a banked direction is a whole latent: keep few
USE_SIM = 0.5               # a prompt this alike counts as "the same kind" (v4 RESCUE_USE_SIM)
STEP_CAP = 0.30             # v4: the rotation is capped at 0.30 x strength of |x|, never over 0.95


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Velocity bias", message)


def structure_step(to):
    """Whether this call is THE structure step: the one nearest 90% of the starting noise, if any is within
    the window. -> bool"""
    where = dit_hooks.current_step(to)
    sched = (to or {}).get("sample_sigmas")
    if where is None or sched is None:
        return False
    top = float(sched[0])
    if top <= 0:
        return False
    ratios = [float(s) / top for s in sched[:-1]]
    nearest = min(range(len(ratios)), key=lambda i: abs(ratios[i] - TARGET))
    return where[0] == nearest and abs(ratios[nearest] - TARGET) <= WINDOW


def reference(rows, shape, sig, nearest):
    """-> (direction | None, how it was chosen) from the liked rows of this shape."""
    liked = [r for r in rows if r["reward"] > 0 and torch.is_tensor(r["rows"].get("v"))]
    fits = [r for r in liked if tuple(r["rows"]["v"].shape) == tuple(shape)]
    if not liked:
        return None, "nothing liked banked yet"
    if not fits:
        return None, f"{len(liked)} liked clip(s) banked, none at this clip's size"
    if nearest and sig is not None:
        best, best_cos = None, USE_SIM
        for r in fits:
            s = r["rows"].get("sig")
            if torch.is_tensor(s) and s.shape == sig.shape:
                cos = float(F.cosine_similarity(s.float(), sig.float(), dim=0))
                if cos >= best_cos:
                    best, best_cos = r, cos
        if best is not None:
            return best["rows"]["v"].float(), f"the nearest of {len(fits)} liked clip(s)"
    return torch.stack([r["rows"]["v"].float() for r in fits]).mean(0), f"the average of {len(fits)} liked clip(s)"


def rotate(x, direction, strength, ratio):
    """`x` turned toward `direction`, size kept (v4 `_apply_velocity_bias`): the push is capped at
    STEP_CAP x strength of |x| (never over 0.95), fades with the step's noise ratio, and the result is
    renormalised to |x|."""
    decay = max(0.0, min(1.0, ratio / TARGET))
    eff = max(0.0, min(3.0, float(strength))) * decay
    if eff <= 0.0:
        return x
    direction = direction.to(device=x.device, dtype=x.dtype)
    delta = direction * eff
    x_norm = x.detach().float().norm().clamp_min(1e-8)
    cap = x_norm * min(0.95, STEP_CAP * eff)
    d_norm = delta.detach().float().norm().clamp_min(1e-8)
    if d_norm > cap:
        delta = delta * (cap / d_norm).to(x.dtype)
    biased = x + delta
    return biased * (x_norm / biased.detach().float().norm().clamp_min(1e-8)).to(x.dtype)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = max(0.0, min(3.0, float(values.get("strength", 0.15))))
    nearest = bool(values.get("nearest"))
    live = {"rows": [], "hit": False, "seen": {}, "multi": False}

    def fresh():
        live["hit"], live["multi"] = False, False
        live["seen"].clear()
        live["rows"] = taste.rows(KIND)
        liked = sum(1 for r in live["rows"] if r["reward"] > 0)
        log.once(f"{ID}:state", log.INFO, "FunPack Velocity bias",
                 f"key {taste.key!r}: {liked} liked clip(s) banked" if liked else
                 f"key {taste.key!r}: learning, nothing liked banked yet")

    # Only liked clips are banked (nothing here learns from a disliked one); clips of different sizes
    # are meant to coexist, so a key holding two resolutions still exports and imports.
    captured = taste.collect(patcher, key, KIND, keep=MAX_ROWS, fresh=fresh, only="liked", mixed=True)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        if dit_hooks.probing(to):
            return executor(x, t, *args, **kwargs)
        if (to or {}).get("context_window") is not None:
            _say("off this run: it cannot follow clips cut into context windows (each window is a different size)")
            return executor(x, t, *args, **kwargs)
        where = dit_hooks.current_step(to)
        if where is not None and where[0] >= where[1] - 1 and not live["hit"] and not live["multi"]:
            _say("did nothing this run: no step of this schedule is near 90% of the starting noise, "
                 "so there was no structure step to bias or learn from")
        if not structure_step(to):
            return executor(x, t, *args, **kwargs)
        kinds = tuple(int(v) for v in ((to or {}).get("cond_or_uncond") or ()))
        c = named.get("c_crossattn")
        row = cond_row(c, to) if torch.is_tensor(c) and c.dim() == 3 else 0
        if row is None:                                    # only the negative prompt in this call
            return executor(x, t, *args, **kwargs)
        if live["seen"].get(kinds) == where[0] or live["multi"]:
            if not live["multi"]:
                live["multi"] = True
                _say("off for the rest of this run: the model is called more than once per step "
                     "(a second-order sampler); use euler-style sampling")
            return executor(x, t, *args, **kwargs)
        live["seen"][kinds] = where[0]
        sig = None
        if torch.is_tensor(c) and c.dim() == 3:
            words = registry.current().ask("text_rows", named, int(c.shape[1]))
            r = c[row] if words is None else c[row][words.to(c.device)]
            sig = r.float().mean(0).detach().cpu()
        split_in = streams.video_of(x, named)
        if split_in is None:
            _say("off this run: could not find the picture in this model's latent")
            return executor(x, t, *args, **kwargs)
        live["hit"] = True
        video_in, rebuild = split_in
        sigma = float(t.max())
        ratio = sigma / float(to["sample_sigmas"][0])
        direction, how = reference(live["rows"], video_in[row].shape, sig, nearest) if strength > 0 else (None, "")
        run_x = x
        if direction is not None:
            run_x = rebuild(rotate(video_in, direction.unsqueeze(0), strength, ratio))
            log.once(f"{ID}:applied:{how}", log.INFO, "FunPack Velocity bias", f"applied: {how}")
        elif strength > 0:
            _say("not applied this run: " + reference(live["rows"], video_in[row].shape, sig, nearest)[1])
        out = executor(run_x, t, *args, **kwargs)
        split_out, split_run = streams.video_of(out, named), streams.video_of(run_x, named)
        if split_out is not None and split_run is not None:
            d = ((split_run[0].float() - split_out[0].float()) / max(sigma, 1e-6))[row]     # where the model moves the picture
            captured["v"] = d.detach().half().cpu()
            if sig is not None:
                captured["sig"] = sig
        return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if strength <= 0.0:
        return "strength 0: learning only"
    return (f"strength {strength:g}, at the structure step{', nearest liked prompt' if nearest else ''}; "
            f"learns from liked clips (read fresh every run)")


PROVIDES = {"modifier": install}
