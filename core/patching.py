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
