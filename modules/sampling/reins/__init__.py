"""Taste steering: push chosen blocks' picture rows toward what you liked.

REINS-style ("Pulling The REINS", arXiv:2606.17257), training-free: every run
records the average picture row at every block on its last step. Once a block
has 2+ liked and 2+ disliked runs on the taste key, liked-minus-disliked is
added back into that block's picture rows, scaled by their own size.

Captured at EVERY block, steers only the named ones: changing which blocks
steer never starts the learning over. Captured before the push, so what is
learned is the model's own behaviour, not this run's strength.

Picture rows only. Sound follows through joint attention, never by being
pushed directly (the LTXAV self-consistency failure was exactly that).
"""

import torch

from ..._core import dit_hooks, log, registry

ID = "reins"
TITLE = "Taste steering"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste steering",
        "hint": "Learns from your ratings and nudges the picture toward what you liked.",
    },
    "strength": {
        "type": "float", "default": 0.01, "min": 0.0, "max": 2.0, "step": 0.01,
        "label": "Strength", "ui": "slider",
        "hint": "0 = only learn. Too high breaks the picture without warning; start low.",
        "when": {"enabled": True},
    },
    "blocks": {
        "type": "text", "default": "43-44",
        "label": "Blocks",
        "hint": "Where it steers, e.g. 43-44 or 0. Every block keeps learning either way.",
        "when": {"enabled": True},
    },
}

KIND = "reins"


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste steering", message)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    n = dit_hooks.block_count(patcher)
    steer, problems = dit_hooks.parse_blocks(values.get("blocks"), n)
    for problem in problems:
        _say(f"ignored {problem}")
    strength = float(values.get("strength", 0.0))

    directions, waiting = {}, []
    for b in steer:
        d, liked, disliked = taste.direction(KIND, b)
        if d is None:
            waiting.append(f"{b} ({liked}/{disliked})")
        else:
            directions[b] = (d, liked, disliked)

    captured = taste.collect(patcher, key, KIND)
    spans = {}

    def video_span(args, out):
        seq = int(out.shape[0])
        if seq not in spans:
            spans[seq] = dit_hooks.row_span(dit_hooks.target_rows(
                args.get("mod_segments"), seq, out.device, "video"))
        return spans[seq]

    def make(block, direction):
        cast = {}

        def hook(args, extra):
            out = extra["original_block"](args)["img"]
            span = video_span(args, out)
            if span is None:
                return {"img": out}
            rows = out[span[0]:span[1]]
            if dit_hooks.last_step(args.get("transformer_options")):
                captured[block] = rows.mean(0, dtype=torch.float32).detach()
            if direction is not None and strength > 0.0:
                if direction.numel() != out.shape[-1]:
                    _say(f"block {block}'s learned direction is from a different model; "
                         "learning only there")
                    return {"img": out}
                d = cast.get(out.dtype)
                if d is None:
                    d = cast[out.dtype] = direction.to(device=out.device, dtype=out.dtype)
                norm = torch.linalg.vector_norm(rows, dim=-1, dtype=torch.float32).mean()
                # No host sync: a bad norm makes the push zero instead of a branch.
                scale = torch.where(torch.isfinite(norm) & (norm > 0), norm * strength,
                                    torch.zeros_like(norm))
                rows += d * scale.to(out.dtype)
            return {"img": out}
        return hook

    for b in range(n):
        d = directions.get(b)
        dit_hooks.add_block_hook(patcher, key, b, make(b, None if d is None else d[0]))

    # Changes what the hooked blocks allocate: see dit_hooks.without_compiler.
    dit_hooks.without_compiler(patcher, key)
    parts = []
    if directions and strength > 0.0:
        parts.append("steering " + ", ".join(f"{b} ({l}/{dl})" for b, (_d, l, dl) in directions.items())
                     + f" at {strength:g}")
    elif directions:
        parts.append("strength 0: learning only")
    if waiting:
        parts.append(f"learning, needs 2 liked + 2 disliked at {', '.join(waiting)}")
    return "; ".join(parts) or "learning at every block"


PROVIDES = {"modifier": install}
