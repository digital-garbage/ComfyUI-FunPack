"""Early taste guidance: the same nudge as Taste guidance, reaching the FIRST half too.

Motion and layout are settled in the first half of the steps, which the late-
half gate every other learning feature shares never reaches. This keeps one
judge per quarter of the schedule, each trained on what the prediction looked
like during ITS quarter, each steering only inside its own quarter.

v4's probe measured the early half carrying ~88% of the late half's rating
signal across 51 rated runs -- the direction was unanimous, nothing alone was
significant. Unvalidated on a GPU: the honest test is the same seed off and on.

A quarter with too few ratings just doesn't act; the run says which do.
"""

from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams

ID = "trajectory_guidance"
TITLE = "Early taste guidance"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "experimental"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Early taste guidance",
        "hint": "After 10+ ratings, nudges motion and layout too, not just the finish.",
    },
    "strength": {
        "type": "float", "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
        "label": "Strength", "ui": "slider",
        "hint": "0 = only learn. Bites harder than Taste guidance at the same number; start at 0.02.",
        "when": {"enabled": True},
    },
}

KIND = "x0_quarters"
QUARTERS = 4


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Early taste guidance", message)


def quarter(transformer_options):
    where = dit_hooks.current_step(transformer_options)
    if where is None:
        return None
    index, total = where
    return min(QUARTERS - 1, index * QUARTERS // max(1, total))


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = float(values.get("strength", 0.02))
    live = {"judges": {}, "sums": {}}

    def fresh():
        live["sums"] = {}
        live["judges"] = {q: taste.judge(KIND, f"q{q}") for q in range(QUARTERS)} if strength > 0 else {}
        ready = [str(q + 1) for q, j in live["judges"].items() if j is not None]
        liked, disliked = taste.counts(KIND)
        log.once(f"{ID}:state", log.INFO, "FunPack Early taste guidance",
                 f"key {taste.key!r}: " + (f"steering quarter(s) {', '.join(ready)}" if ready else
                                            f"learning ({liked} liked / {disliked} disliked; "
                                            "each quarter steers from 10 rated clips)"))

    captured = taste.collect(patcher, key, KIND, fresh=fresh)

    def apply_model(executor, x, t, *args, **kwargs):
        out = executor(x, t, *args, **kwargs)
        named = streams.model_args(args, kwargs)
        if dit_hooks.probing(named.get("transformer_options")):
            return out                           # a discarded candidate: learn and steer nothing
        q = quarter(named.get("transformer_options"))
        split = streams.video_of(out, named)
        if q is None or split is None:
            _say("off this run: could not place this step on the schedule or find the picture")
            return out
        video, rebuild = split
        if video.shape[0] != 1:
            _say("off this run: batched predictions (CFG above 1) can't be steered honestly")
            return out
        # Each quarter's fingerprint is the mean over its steps, kept current so
        # the capture is whole whenever the run ends.
        d = taste.describe(video.detach())
        total, count = live["sums"].get(q, (0, 0))
        live["sums"][q] = (total + d, count + 1)
        captured[f"q{q}"] = (total + d) / (count + 1)
        judge = live["judges"].get(q)
        if judge is None or strength <= 0.0:
            return out
        return rebuild(judge.nudge(video, strength))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return (f"strength {strength:g}, every step by quarter; learns every run, "
            "each quarter steers from 10 rated clips (read fresh every run)")


PROVIDES = {"modifier": install}
