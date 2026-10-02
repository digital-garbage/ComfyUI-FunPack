"""Decisiveness: a temperature for what the model draws (Temporal Score Rescaling).

Xu et al., arXiv 2510.01184. Each step's prediction is split into "the picture"
and "the noise it thinks is left"; scaling the noise part by

    r = (eta * s^2 + 1) / (eta * s^2 / k + 1),   eta = ((1 - sigma) / sigma)^2

samples from a sharper (k > 1: more committed, less varied) or broader (k < 1)
version of what the model learned. k = 1 is exactly off. Near pure noise r -> 1,
so it cannot disturb the first steps; `s` sets how early it takes hold.

Learned: every run tries a k a little off the key's current value, and each
rating pulls the value toward the k that was liked or away from the one that
was disliked. Manual: the k you type. Picture only; no extra model call. The
edit rides into the NEXT step's input (core/input_steer.py), never the answer.

Unvalidated on a GPU. v4's own check: on simple@4 a k of 0.9 pushes ~2% of a
step's noise, which the run reports as "barely acts" and does not learn from.
"""

import math
import random

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import input_steer, log, registry, streams

ID = "decisiveness"
TITLE = "Decisiveness"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Decisiveness",
        "hint": "Makes the model commit harder (or looser) to what it draws.",
    },
    "mode": {
        "type": "enum", "default": "learned", "options": [
            {"value": "learned", "label": "Learn from ratings"},
            {"value": "manual", "label": "My value"},
        ],
        "label": "Value", "ui": "segmented",
        "hint": "Learn: each rating nudges it toward what you liked.",
        "when": {"enabled": True},
    },
    "k": {
        "type": "float", "default": 1.0, "min": 0.5, "max": 1.6, "step": 0.05,
        "label": "k", "ui": "slider",
        "hint": "Above 1 commits harder, below 1 loosens. 1 is off.",
        "when": {"enabled": True, "mode": "manual"},
    },
}

KIND = "decisiveness"
# The paper's image setting (SD3). ponytail: fixed; learn it too if k alone plateaus.
S = 3.0
LOGK_MAX = 0.3          # k within ~0.74..1.35
EXPLORE = 0.05
LR = 0.5
# Below this share of a step's input noise, a push cannot be felt (simple@4, k 0.9 is ~2%).
# ponytail: judged by eye on one schedule, not measured; tune if a rental says otherwise.
INERT_SHARE = 0.03


def _say(message, level=log.ALERT):
    log.once(f"{ID}:{message}", level, "FunPack Decisiveness", message)


def factor(sigma, k, s=S):
    sigma = min(max(float(sigma), 1e-6), 1.0)
    eta = ((1.0 - sigma) / sigma) ** 2
    return (eta * s * s + 1.0) / (eta * s * s / k + 1.0)


def rescale_x0(x, x0, sigma, k):
    """A rectified-flow x0 prediction at `sigma` with its noise part scaled by r."""
    a = 1.0 - float(sigma)
    if a < 1e-3 or float(sigma) < 1e-4 or k == 1.0:
        return x0
    r = factor(sigma, k)
    eps = (x - a * x0) / float(sigma)
    return (x - float(sigma) * r * eps) / a


def reach(sigmas, k, s=S):
    """The largest share of a step's input noise the push makes up when each push rides
    into the NEXT step's input: sigma*|1-r|/(1-sigma), scaled by the landing step's
    (1 - sigma) and set against its noise sigma. The last step is never edited."""
    vals = [float(v) for v in sigmas]
    best = 0.0
    for i in range(len(vals) - 2):
        s0, s1 = vals[i], vals[i + 1]
        if s0 < 1e-4 or s1 < 1e-4 or 1.0 - s0 < 1e-3:
            continue
        push = s0 * abs(1.0 - factor(s0, k, s)) / (1.0 - s0) * (1.0 - s1)
        best = max(best, push / s1)
    return best


def learned_logk(history):
    """Sign-ES over the rated runs, oldest first: toward a liked log k, away from a disliked."""
    m = 0.0
    for logk, reward in history:
        sign = (reward > 0) - (reward < 0)
        m += LR * sign * (logk - m)
        m = min(max(m, -LOGK_MAX), LOGK_MAX)
    return m


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    manual = values.get("mode") == "manual"
    taste = None if manual else registry.current().ask("taste_store", patcher)
    if not manual and taste is None:
        _say("off: learning needs a Taste key -- set one, or pick 'My value'")
        return None
    live = {"k": 1.0, "acts": False, "checked": False, "steps": 0}
    steer = input_steer.Steer("Decisiveness")
    steer.attach(patcher, key)

    def fresh():
        live.update(checked=False, acts=False, steps=0, rows=None if manual else taste.rows(KIND, blind_to="image"))
        live["k"] = float(values.get("k", 1.0)) if manual else 1.0
        if manual:
            log.once(f"{ID}:state", log.INFO, "FunPack Decisiveness", f"manual k {live['k']:.3f}")

    def choose(steps):
        """Learned k for a schedule of this length. Ratings from another length are left out:
        the same k pushes a very different share of a 4-step and a 12-step run."""
        history = [(float(r["rows"]["logk"]), float(r["reward"])) for r in live["rows"]
                   if "logk" in r["rows"] and int(r["rows"].get("steps", steps)) == steps]
        centre = learned_logk(history)
        live["k"] = math.exp(min(max(centre + random.gauss(0.0, EXPLORE), -LOGK_MAX), LOGK_MAX))
        log.once(f"{ID}:state", log.INFO, "FunPack Decisiveness",
                 f"key {taste.key!r}: learned k {math.exp(centre):.3f} from {len(history)} "
                 f"rating(s) at {steps} steps, trying {live['k']:.3f}")

    captured = (taste.collect(patcher, key, KIND, fresh=fresh) if not manual else {})
    if manual:
        def start(executor, *args, **kwargs):
            fresh()
            return executor(*args, **kwargs)

        patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, start)

    def learn(k):
        """Only a k whose push reached the model teaches. Checked on every call, the last
        included: on a few-step schedule the biggest push lands on the final step."""
        if steer.felt():
            captured["logk"] = torch.tensor(math.log(k))
            captured["steps"] = torch.tensor(live["steps"])
        elif steer.multi:
            captured.clear()

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        step = steer.begin(x, t, to)
        out = executor(step.x, t, *args, **kwargs)
        if not live["checked"] and to and to.get("sample_sigmas") is not None:
            live["checked"] = True
            live["steps"] = len(to["sample_sigmas"]) - 1
            if not manual:
                choose(live["steps"])
            k = live["k"]
            if k == 1.0:
                return out
            share = reach(to["sample_sigmas"].flatten().tolist(), k)
            live["acts"] = share >= INERT_SHARE
            if not live["acts"]:
                _say(f"Inactive | barely acts on this schedule (k {k:.3f} pushes {share:.1%} of "
                     f"a step's noise, under {INERT_SHARE:.0%}); more steps or a larger k would "
                     "let it be felt" + ("" if manual else "; this run teaches nothing"))
        k = live["k"]
        if not manual:
            learn(k)
        if not live["acts"] or step.final or k == 1.0:
            return out
        split_x, split_out = streams.video_of(step.x, named), streams.video_of(out, named)
        if split_x is None or split_out is None:
            _say("off this run: could not find the picture in this model's latent")
            return out
        video, rebuild = split_out
        sigma = float(t.max())
        return step.keep(out, rebuild(rescale_x0(split_x[0], video, sigma, k).to(video.dtype)))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return ("manual k" if manual else "learned k") + ", picture only, carried into the next step"


PROVIDES = {"modifier": install}
