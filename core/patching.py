"""Tagging what we install, so it can be removed again.

ComfyUI offers two shapes of hook. Some are keyed -- `add_wrapper_with_key` takes
a name and removing it is exact. Others are plain lists on `model_options`, where
`set_model_sampler_pre_cfg_function` appends an anonymous callable and nothing
records who put it there.

For the second kind, install-then-forget means a model run through the loader
twice carries two copies and applies the effect at double strength, reporting
once. That is the accumulation this project has already paid for: v4 registered
hooks on shared blocks and only a restart cleared them.

So everything we install is tagged, and everything tagged is stripped before the
next install. The walk below is generic -- it looks for our tag rather than for a
list of known hook names, because a list of hook names is a thing to keep current
and the day it falls behind is the day a hook stops being removable.
"""

from typing import Any

TAG = "_funpack_key"


def tag(fn, key: str):
    """Mark a callable as ours, so `strip` can find it later."""
    try:
        setattr(fn, TAG, key)
    except AttributeError:
        # A builtin or a slotted object cannot carry the mark, and something
        # unremovable must not be installed silently.
        raise TypeError(
            f"{fn!r} cannot be tagged, so it could never be removed again. "
            f"Install a plain function or a closure.")
    return fn


def _ours(value: Any, prefix: str) -> bool:
    key = getattr(value, TAG, None)
    return isinstance(key, str) and key.startswith(prefix)


def strip(patcher, prefix: str) -> int:
    """Remove every tagged hook under `prefix`, and nothing else.

    Covers both shapes: keyed wrappers and callbacks, and the anonymous function
    lists on model_options. Returns how many were removed, so a caller can say
    that it replaced something rather than leaving it to be inferred.
    """
    removed = 0

    for holder in (getattr(patcher, "wrappers", None), getattr(patcher, "callbacks", None)):
        if not isinstance(holder, dict):
            continue
        for by_key in holder.values():
            if not isinstance(by_key, dict):
                continue
            for key in [k for k in by_key if isinstance(k, str) and k.startswith(prefix)]:
                by_key.pop(key, None)
                removed += 1

    options = getattr(patcher, "model_options", None)
    if isinstance(options, dict):
        removed += _strip_options(options, prefix)
    return removed


def _strip_options(options: dict, prefix: str) -> int:
    """Walk model_options for tagged callables, in lists or on their own."""
    removed = 0
    for name, value in list(options.items()):
        if isinstance(value, list):
            keep = [item for item in value if not _ours(item, prefix)]
            if len(keep) != len(value):
                removed += len(value) - len(keep)
                options[name] = keep
        elif callable(value) and _ours(value, prefix):
            options.pop(name, None)
            removed += 1
        elif isinstance(value, dict):
            removed += _strip_options(value, prefix)
    return removed


# --- keeping a module from killing a run ----------------------------------
#
# Catching a module that fails to INSTALL is easy and was already done. The one
# that matters is a module that installs fine and raises on step 3 of 30: the
# generation dies after the GPU time is spent, and on a rental that is the whole
# cost of the run. A modifier is an opinion about the picture. It is never worth
# the picture.
#
# So a failing hook is dropped for the rest of the run and the run continues
# without it. Loudly -- a dropped modifier is the difference between "I asked for
# this" and "this happened", so it is said once, with the traceback, and carried
# in the run's status.

class Dropped:
    """What was disabled during a run, and why. One per run, not per process."""

    def __init__(self):
        self.reasons: dict = {}

    def record(self, key: str, exc: BaseException) -> bool:
        """True the first time this key fails, so callers report once."""
        if key in self.reasons:
            return False
        self.reasons[key] = f"{type(exc).__name__}: {exc}"
        return True

    def __contains__(self, key) -> bool:
        return key in self.reasons

    def __bool__(self) -> bool:
        return bool(self.reasons)

    def items(self):
        return self.reasons.items()


def guard(fn, key: str, neutral, dropped: Dropped):
    """`fn`, but unable to end a run.

    `neutral(*args, **kwargs)` produces the result the caller would have got had
    the hook never been installed -- it differs per hook shape, which is why it
    is supplied rather than guessed. After the first failure the hook is skipped
    outright: a modifier that raises once will raise every step, and thirty
    tracebacks say nothing the first one did not.
    """
    import traceback

    def guarded(*args, **kwargs):
        if key in dropped:
            return neutral(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                 # noqa: BLE001 -- the whole point
            if dropped.record(key, exc):
                from . import log
                log.note(
                    f"{key} failed during sampling and is now OFF for the rest of "
                    f"this run. The run continues without it.\n"
                    + "".join(traceback.format_exception(exc)).rstrip())
            return neutral(*args, **kwargs)

    # Carried over so a guarded hook is still ours to strip.
    marker = getattr(fn, TAG, None)
    if marker is not None:
        guarded.__dict__[TAG] = marker
    return guarded


# The neutral result for each hook ComfyUI offers, by the method that installs
# it. Core knows ComfyUI's own shapes here -- not FunPack's features -- because
# "what this hook returns when it does nothing" is a fact about ComfyUI.
def _neutral_wrapper(executor, *args, **kwargs):
    return executor(*args, **kwargs)


NEUTRAL = {
    "add_wrapper_with_key": _neutral_wrapper,
    "add_callback_with_key": lambda *a, **k: None,
    "set_model_sampler_pre_cfg_function": lambda args: args["conds_out"],
    "set_model_sampler_post_cfg_function": lambda args: args["denoised"],
    "set_model_sampler_cfg_function": lambda args: args["input"] - (
        args["uncond_denoised"] + (args["cond_denoised"] - args["uncond_denoised"]) * args["cond_scale"]),
    "set_model_denoise_mask_function": lambda sigma, denoise_mask, extra_options=None: denoise_mask,
}

# Which positional argument holds the callable, for methods that take more than one.
CALLABLE_AT = {"add_wrapper_with_key": 2, "add_callback_with_key": 2}


class GuardedPatcher:
    """A ModelPatcher that guards every hook installed through it.

    A module is handed one of these instead of the real patcher, so it does not
    opt in to being safe -- a module that could forget to is a module that can
    break the foundation. Everything else forwards untouched.
    """

    def __init__(self, patcher, key: str, dropped: Dropped, guarding: bool = True):
        object.__setattr__(self, "_patcher", patcher)
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_dropped", dropped)
        object.__setattr__(self, "_guarding", guarding)
        object.__setattr__(self, "unguarded", [])

    def __getattr__(self, name):
        target = getattr(self._patcher, name)
        if not callable(target) or not name.startswith(("set_model_", "add_wrapper", "add_callback")):
            return target

        if not self._guarding:
            # Full control: install exactly what the module wrote, and let it
            # fail where it fails. Someone asking for the raw behaviour wants
            # the real traceback at the real step, not a tidy recovery.
            return target

        neutral = NEUTRAL.get(name)
        if neutral is None:
            # Honest about the limit: an unknown hook shape has no known neutral
            # result, so it is installed as-is and named. Silently pretending it
            # was guarded would be worse than saying it was not.
            self.unguarded.append(name)
            return target

        index = CALLABLE_AT.get(name, 0)

        def install(*args, **kwargs):
            args = list(args)
            if len(args) > index and callable(args[index]):
                args[index] = guard(args[index], self._key, neutral, self._dropped)
            return target(*args, **kwargs)

        return install

    def __setattr__(self, name, value):
        setattr(self._patcher, name, value)
