"""Block repeat: run chosen blocks a second time, inside one sampler step.

`x' = x + f(x)`, then `x'' = x' + f(x')`. No VAE round trip, no re-noise, no
extra step: one more block forward per repeat, ~2% of a step per block on a
50-block model.

What it does, from the v4 sweep (research_block_repeat_sweep): a repeat is a
TRAJECTORY RE-ROLL -- a different take on the same prompt -- not a sharpening
pass. The band 20-41 behaved best. The blocks after a repeated one were trained
on its normal output, so pushing further (more passes, wider spans) drifts out
of distribution.

Two shapes:

* **each** -- 31,31,32,32..40,40. Every block gets its own doubled output.
* **span** -- 31..40,31..40. The whole span loops, so every block still gets
  input from its normal predecessor and there is ONE seam (40 -> 31) instead of
  one per block. Span mode runs the span's blocks directly, so a hook another
  module put on a block INSIDE the span does not run on the looped passes.

"Video only" keeps the extra pass for video rows and gives text and audio the
single pass, because repeating the audio rows is the lead suspect for the
unprompted dialogue band 31-40 invents. Attention has already mixed the doubled
audio into the video before the restore; it protects what you hear, not what
the video attended to.
"""

import torch

from ..._core import dit_hooks, log

ID = "block_repeat"
TITLE = "Block repeat"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Block repeat",
        "hint": "Re-rolls part of the model's thinking for a different take on the same prompt.",
    },
    "blocks": {
        "type": "text", "default": "31-40",
        "label": "Blocks",
        "hint": "e.g. 31-40, or 20,35. Somewhere in 20-41 works best.",
        "when": {"enabled": True},
    },
    "shape": {
        "type": "enum", "default": "each", "options": [
            {"value": "each", "label": "Each block"},
            {"value": "span", "label": "Whole span"},
        ],
        "label": "Repeat", "ui": "segmented",
        "hint": "each: every block twice in place. span: the whole range loops once.",
        "when": {"enabled": True},
    },
    "times": {
        "type": "int", "default": 1, "min": 1, "max": 4,
        "label": "Extra passes",
        "hint": "1 = each block runs twice. More drifts further.",
        "when": {"enabled": True},
    },
    "video_only": {
        "type": "bool", "default": False,
        "label": "Video only",
        "hint": "Keeps the sound out of it. Try this if people start talking unprompted.",
        "when": {"enabled": True},
    },
    "steps": {
        "type": "int", "default": 0, "min": -50, "max": 50,
        "label": "Which steps",
        "hint": "0 = every step. 3 = only the last 3. -3 = only the first 3.",
        "when": {"enabled": True},
    },
}


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Block repeat", message)


def _in_window(args, steps):
    if steps == 0:
        return True
    where = dit_hooks.current_step(args.get("transformer_options"))
    if where is None:
        # Fails OPEN -- repeat every step -- and says so, rather than quietly
        # never repeating at all.
        _say("could not tell which step this is, so it repeats on every step this run")
        return True
    index, total = where
    return index >= total - steps if steps > 0 else index < -steps


def _video_only(args, doubled, single):
    """Doubled video rows, single-pass everything else."""
    if single.shape != doubled.shape:
        return doubled
    mask = dit_hooks.target_rows(args.get("mod_segments"), int(doubled.shape[0]),
                                 doubled.device, "video")
    if mask is None:
        _say("could not find the video rows, so 'video only' repeated everything this run")
        return doubled
    return torch.where(mask.view(-1, *([1] * (doubled.dim() - 1))), doubled, single)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    n = dit_hooks.block_count(patcher)
    blocks, problems = dit_hooks.parse_blocks(values.get("blocks"), n)
    for problem in problems:
        _say(f"ignored {problem}")
    if not blocks:
        raise ValueError(f"no usable blocks in {values.get('blocks')!r}, so nothing would repeat")
    times = int(values.get("times", 1))
    steps = int(values.get("steps", 0))
    video_only = bool(values.get("video_only"))
    span = values.get("shape") == "span"

    if span:
        lo, hi = blocks[0], blocks[-1]
        if blocks != list(range(lo, hi + 1)):
            raise ValueError(f"span needs a range like 31-40, got {values.get('blocks')!r}")
        real = patcher.model.diffusion_model.blocks

        def run_span(args, h):
            for i in range(lo, hi + 1):
                h = real[i](h, args["t_emb"], args["mod_segments"], args["rope_freqs"],
                            transformer_options=args.get("transformer_options"))
            return h

        def head(args, extra):
            once = run_span(args, args["img"])
            if not _in_window(args, steps):
                return {"img": once}
            # H3's blocks add their residual INTO the input tensor, so without a
            # copy the next pass overwrites `once` and "video only" restores
            # audio from an already-repeated stream (v4 shipped that bug).
            h = once.clone() if video_only else once
            for _ in range(times):
                h = run_span(args, h)
            return {"img": _video_only(args, h, once) if video_only else h}

        def skip(args, extra):
            return {"img": args["img"]}          # the head already ran this block

        dit_hooks.add_block_hook(patcher, key, lo, head)
        for i in range(lo + 1, hi + 1):
            dit_hooks.add_block_hook(patcher, key, i, skip)
        what = f"blocks {lo}-{hi} loop {times + 1}x as one span"
    else:
        def repeat(args, extra):
            once = extra["original_block"](args)["img"]
            if not _in_window(args, steps):
                return {"img": once}
            # H3's blocks add their residual INTO the input tensor, so without a
            # copy the next pass overwrites `once` and "video only" restores
            # audio from an already-repeated stream (v4 shipped that bug).
            h = once.clone() if video_only else once
            for _ in range(times):
                # A fresh dict: the caller's args must not see the intermediate.
                h = extra["original_block"]({**args, "img": h})["img"]
            return {"img": _video_only(args, h, once) if video_only else h}

        for i in blocks:
            dit_hooks.add_block_hook(patcher, key, i, repeat)
        what = f"blocks {','.join(map(str, blocks))} run {times + 1}x each"

    # Replaces what the hooked blocks allocate: see dit_hooks.without_compiler.
    dit_hooks.without_compiler(patcher, key)
    if video_only:
        what += ", video only"
    if steps:
        what += f", {'last' if steps > 0 else 'first'} {abs(steps)} step(s)"
    return what


PROVIDES = {"modifier": install}
