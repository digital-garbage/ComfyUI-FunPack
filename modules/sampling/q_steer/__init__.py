"""Taste attention: tilt what chosen blocks' picture rows look FOR, toward what you liked.

Same learning as Taste steering (liked-minus-disliked on the taste key), but on
the attention queries instead of the block output. Only the query can move
anything: a constant added to every key is cancelled by softmax, and one added
to every value is just Taste steering again.

Learned and applied where the rows are NOT yet position-rotated (the model
module supplies the rotation): an average of rotated rows mostly cancels, and a
push added after rotation points somewhere different at every position --
steering by WHERE, not WHAT. v4's first version made exactly that mistake.

Picture rows only. Captured at every block, steers only the named ones.
"""

import torch

from ..._core import dit_hooks, log, registry

ID = "q_steer"
TITLE = "Taste attention"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Taste attention",
        "hint": "Learns from your ratings and shifts what the picture pays attention to.",
    },
    "strength": {
        "type": "float", "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.05,
        "label": "Strength", "ui": "slider",
        "hint": "0 = only learn.",
        "when": {"enabled": True},
    },
    "blocks": {
        "type": "text", "default": "43-44",
        "label": "Blocks",
        "hint": "Where it steers, e.g. 43-44 or 0,1,47-49. Every block keeps learning either way.",
        "when": {"enabled": True},
    },
}

KIND = "q_steer"
_SCOPE = "funpack_q_steer"


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Taste attention", message)


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
    layout = {}                                  # seq_len -> (lo, hi, rotate) | None

    def scoped(block):
        def hook(args, extra):
            to = dict(args.get("transformer_options") or {})
            to[_SCOPE] = (block, args.get("mod_segments"), args.get("rope_freqs"))
            return extra["original_block"]({**args, "transformer_options": to})
        return hook

    for b in range(n):
        dit_hooks.add_block_hook(patcher, key, b, scoped(b))

    def resolve(segs, rope, seq, device):
        if seq not in layout:
            span = dit_hooks.row_span(dit_hooks.target_rows(segs, seq, device, "video"))
            rotate = registry.current().ask("query_rotation", rope, seq)
            layout[seq] = None if span is None or rotate is None else (*span, rotate)
            if layout[seq] is None:
                _say("could not find the picture rows or their rotation; "
                     "not learning or steering this run")
        return layout[seq]

    def override(func, q, k, v, *args, **kwargs):
        to = kwargs.get("transformer_options") or {}
        scope = to.get(_SCOPE)
        if scope is None or q.dim() != 4 or q.shape[0] != 1:
            return func(q, k, v, *args, **kwargs)
        block, segs, rope = scope
        seq = int(q.shape[2])
        found = resolve(segs, rope, seq, q.device)
        if found is None:
            return func(q, k, v, *args, **kwargs)
        lo, hi, rotate = found
        rows = q[0, :, lo:hi]                                     # [H, n, D]
        heads, count, dim = rows.shape
        if dit_hooks.last_step(to):
            natural = rotate(rows, slice(lo, hi), inverse=True)
            captured[block] = natural.transpose(0, 1).reshape(count, -1).mean(0, dtype=torch.float32)
        hit = directions.get(block)
        if hit is None or strength <= 0.0:
            return func(q, k, v, *args, **kwargs)
        direction = hit[0]
        if direction.numel() != heads * dim:
            _say(f"block {block}'s learned direction is from a different model; learning only there")
            return func(q, k, v, *args, **kwargs)
        # Rotation keeps length, so the rotated rows' size is the natural one's.
        norm = torch.linalg.vector_norm(rows, dim=-1, dtype=torch.float32).mean()
        scale = torch.where(torch.isfinite(norm) & (norm > 0), norm * strength, torch.zeros_like(norm))
        push = (direction.to(device=q.device, dtype=q.dtype).view(heads, 1, dim)
                * scale.to(q.dtype)).expand(heads, count, dim)
        q = q.clone()
        q[0, :, lo:hi] += rotate(push, slice(lo, hi))
        return func(q, k, v, *args, **kwargs)

    dit_hooks.add_attention_override(patcher, key, override)
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
