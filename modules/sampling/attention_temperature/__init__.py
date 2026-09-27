"""Randomizer: loosen or tighten how firmly chosen blocks weigh the prompt.

Scales the query before attention at the named blocks, which is softmax
temperature without ever building the score matrix: Q/T . K == (Q . K)/T. Above
1 the model is less sure which tokens matter and wanders; below 1 it commits
harder to what the prompt already made confident. Either way it can only
re-weigh what is in context, so unlike added noise it stays inside the prompt.

Validated on H3 by eye (2026-09-20), with two conditions the user found: keep it
moderate, and keep it off blocks a steering module is already pushing on.
Known-good: 0.75x (sharper) at blocks 2-10, or blocks 29-46.

Not rating-driven, on purpose: a plain dial. Coupling one hard-to-read mechanism
to another's data would make neither debuggable.
"""

from ..._core import dit_hooks, log

ID = "attention_temperature"
TITLE = "Randomizer"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "proven"
REQUIRES = ["dit_block_hooks"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Randomizer",
        "hint": "Makes the model more adventurous or more literal with your prompt.",
    },
    "amount": {
        "type": "float", "default": -0.25, "min": -0.9, "max": 3.0, "step": 0.05,
        "label": "Amount", "ui": "slider",
        "hint": "Below zero sticks closer to the prompt, above zero wanders. Keep it small.",
        "when": {"enabled": True},
    },
    "blocks": {
        "type": "text", "default": "2-10",
        "label": "Blocks",
        "hint": "e.g. 2-10 or 29-46. Keep away from blocks a steering feature uses.",
        "when": {"enabled": True},
    },
}

# Carried on the per-block copy of transformer_options, so an attention call knows
# it belongs to a named block without any shared flag that could leak.
_SCOPE = "funpack_attention_temperature"


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    amount = float(values.get("amount", 0.0))
    if amount == 0.0:
        return None
    blocks, problems = dit_hooks.parse_blocks(values.get("blocks"), dit_hooks.block_count(patcher))
    for problem in problems:
        log.once(f"{ID}:{problem}", log.ALERT, "FunPack Randomizer", f"ignored {problem}")
    if not blocks:
        raise ValueError(f"no usable blocks in {values.get('blocks')!r}, so nothing would change")

    # The widget floor keeps this above zero; the clamp is for API callers.
    temperature = max(1.0 + amount, 0.05)

    def scoped(args, extra):
        to = dict(args.get("transformer_options") or {})
        to[_SCOPE] = temperature
        return extra["original_block"]({**args, "transformer_options": to})

    for index in blocks:
        dit_hooks.add_block_hook(patcher, key, index, scoped)

    def override(func, q, k, v, *args, **kwargs):
        t = (kwargs.get("transformer_options") or {}).get(_SCOPE)
        return func(q if t is None else q / t, k, v, *args, **kwargs)

    dit_hooks.add_attention_override(patcher, key, override)
    direction = "sharper" if amount < 0 else "looser"
    return f"{temperature:g}x ({direction}) at blocks {_ranges(blocks)}"


def _ranges(blocks):
    out, start = [], None
    for i, b in enumerate(blocks):
        if start is None:
            start = b
        if i + 1 == len(blocks) or blocks[i + 1] != b + 1:
            out.append(str(start) if start == b else f"{start}-{b}")
            start = None
    return ",".join(out)


PROVIDES = {"modifier": install}
