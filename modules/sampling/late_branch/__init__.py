"""Late-branch guidance: push each step's picture away from a deliberately weakened copy.

STG-style skip guidance made cheap enough for CFG 1: the weak copy shares every block
before the branch with the normal one, so it only re-runs the tail. Branching at 43 of
H3's 50 blocks costs ~7 blocks a step (~15%), not a second forward.

    guided = normal + w * (normal - weak)      picture only; sound can still react

The normal pass saves the stream as it enters the branch block. The weak pass skips every
block before it and the branch block itself, restarts from that saved stream and runs the
tail. The push rides into the NEXT step's input (core/input_steer.py); the last step runs
no weak copy and the model's own answer is always what comes back.

w is learned from ratings (core/rated_dial.py) or typed. Unvalidated on a GPU; v4 had it
off in the run where the user saw motion improve, so it is ruled out as the cause of that.

Costs one tail re-run per cond step. Switches ComfyUI's model compiler off for the run
(the hooks allocate differently across calls).
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, input_steer, log, rated_dial, registry, streams

ID = "late_branch"
TITLE = "Late-branch guidance"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent", "dit_block_hooks"]
# Innermost APPLY_MODEL: every other wrapper sees one guided prediction, not two calls.
AFTER = ["score_slider"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Late-branch guidance",
        "hint": "Pushes the picture away from a weakened copy of itself. Costs ~15% more per step.",
    },
    "mode": {
        "type": "enum", "default": "learned", "options": [
            {"value": "learned", "label": "Learn from ratings"},
            {"value": "manual", "label": "My value"},
        ],
        "label": "Strength", "ui": "segmented",
        "hint": "Learn: each rating nudges it toward what you liked.",
        "when": {"enabled": True},
    },
    "strength": {
        "type": "float", "default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05,
        "label": "Strength", "ui": "slider",
        "hint": "0 does nothing; above 1 is strong.",
        "when": {"enabled": True, "mode": "manual"},
    },
    "block": {
        "type": "int", "default": 43, "min": 1, "max": 49,
        "label": "Branch block",
        "hint": "Later = cheaper and subtler. Must be a block of the model.",
        "when": {"enabled": True},
    },
}

KIND = "late_branch"
WEAK = dit_hooks.WEAK_BRANCH
DIAL = rated_dial.Dial(start=0.5, lo=0.0, hi=1.5, explore=0.15)


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunPack Late-branch guidance", message)


def _weak(args):
    return dit_hooks.weak_branch(args.get("transformer_options"))


def mix(normal, weak, w):
    return normal + w * (normal - weak)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    n = dit_hooks.block_count(patcher)
    branch = int(values.get("block", 43))
    if not 0 < branch < n:
        _say(f"off: block {branch} is not a branch point of this model (1-{n - 1})")
        return None
    manual = values.get("mode") == "manual"
    taste = None if manual else registry.current().ask("taste_store", patcher)
    if not manual and taste is None:
        _say("off: learning needs a Taste key -- set one, or pick 'My value'")
        return None
    live = {"w": float(values.get("strength", 0.5)), "want": False, "centre": None}
    saved = {}
    steer = input_steer.Steer("Late-branch guidance")
    steer.attach(patcher, key)

    def fresh():
        saved.clear()
        if manual:
            return log.once(f"{ID}:state", log.INFO, "FunPack Late-branch guidance",
                            f"manual strength {live['w']:.2f}, branch at block {branch}")
        live["w"], centre, held = DIAL.pick(rated_dial.history(taste.rows(KIND)))
        log.once(f"{ID}:state", log.INFO, "FunPack Late-branch guidance",
                 f"key {taste.key!r}: learned {centre:.2f} from {held} rating(s), trying "
                 f"{live['w']:.2f}, branch at block {branch}")

    captured = taste.collect(patcher, key, KIND, fresh=fresh) if not manual else {}
    if manual:
        def start(executor, *args, **kwargs):
            fresh()
            return executor(*args, **kwargs)

        patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, start)

    def shared(args, extra):
        if _weak(args):
            return {"img": args["img"]}                  # the saved stream replaces this block
        return extra["original_block"](args)

    def at_branch(args, extra):
        if _weak(args):
            # A copy: the tail adds its residuals into its input in place.
            return {"img": saved["h"].clone()}
        if live["want"]:
            saved["h"] = args["img"].clone()             # BEFORE the block writes into it
        return extra["original_block"](args)

    for b in range(branch):
        dit_hooks.add_block_hook(patcher, key, b, shared)
    dit_hooks.add_block_hook(patcher, key, branch, at_branch)
    dit_hooks.without_compiler(patcher, key)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        step = steer.begin(x, t, to)
        cond_only = all(int(i) == 0 for i in (to.get("cond_or_uncond") or [0]))
        branching = (step.steering and not step.final and cond_only and live["w"] > 0.0
                     and not dit_hooks.probing(to))
        live["want"] = branching
        try:
            normal = executor(step.x, t, *args, **kwargs)
        finally:
            live["want"] = False
        if not branching:
            if step.steering and not step.final and not cond_only:
                _say("a negative-prompt call (CFG above 1) is left unguided: guidance is for "
                     "the picture the prompt asks for")
            return normal
        if "h" not in saved:
            _say("Inactive | the branch block never ran, so there is no weak copy")
            return normal
        weak_args, weak_kwargs = streams.with_options(
            args, kwargs, {**to, WEAK: True, dit_hooks.PROBE: True})
        try:
            weak = executor(step.x, t, *weak_args, **weak_kwargs)
        finally:
            saved.pop("h", None)
        split_n, split_w = streams.video_of(normal, named), streams.video_of(weak, named)
        if split_n is None or split_w is None:
            _say("off this run: could not find the picture in this model's latent")
            return normal
        video, rebuild = split_n
        guided = step.keep(normal, rebuild(mix(video, split_w[0], live["w"]).to(video.dtype)))
        if not manual and steer.effect() is not None:
            captured["v"] = torch.tensor(live["w"])
            captured["e"] = torch.tensor(steer.effect())
        return guided

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return (("manual" if manual else "learned") + f" strength, weak copy left out at block {branch}, "
            "picture only, carried into the next step")


PROVIDES = {"modifier": install}
