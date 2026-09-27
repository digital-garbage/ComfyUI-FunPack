"""First-step seed search: try a few seeds for one step, keep the one your taste likes.

Before the real run, each candidate seed gets exactly ONE denoising step, its
predicted picture is scored by a judge trained on how your rated clips looked
at their first step, and the run continues with the best seed. The first step
already carries most of what a rating reacts to (v4's probe: ~88% of the late
half's signal).

User verdict (v4): a win on simple prompts; no help on prompts that depend on
word trickery, which the first step can't read yet.

Costs one extra model pass per candidate. Candidate steps are marked as probes,
so no learning feature records a step the clip never took.
"""

import torch
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, log, registry, streams

ID = "explore_first_step"
TITLE = "First-step seed search"
MOUNT = "generation.sampling"
STAGE = "sampling"
CATEGORY = "sampling"
STATUS = "proven"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Seed search",
        "hint": "After 10+ ratings, tries a few seeds for one step each and keeps the one your taste likes.",
    },
    "candidates": {
        "type": "int", "default": 3, "min": 2, "max": 8,
        "label": "Seeds to try",
        "hint": "Each costs one extra step. Your seed is always one of them.",
        "when": {"enabled": True},
    },
}

KIND = "first_step"
NAME = "first"


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Seed search", message)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    count = int(values.get("candidates", 3))
    live = {}

    def fresh():
        live["judge"] = taste.judge(KIND, NAME)
        liked, disliked = taste.counts(KIND, NAME)
        if live["judge"] is None:
            log.once(f"{ID}:state", log.INFO, "FunPack Seed search",
                     f"key {taste.key!r}: learning ({liked} liked / {disliked} disliked; "
                     "searches from 10 rated clips with both kinds)")

    captured = taste.collect(patcher, key, KIND, fresh=fresh)

    # What the real run's first step looked like: what the judge learns from.
    def apply_model(executor, x, t, *args, **kwargs):
        out = executor(x, t, *args, **kwargs)
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        where = dit_hooks.current_step(to)
        if dit_hooks.probing(to) or where is None or where[0] != 0:
            return out
        split = streams.video_of(out, named)
        if split is not None and split[0].shape[0] == 1:
            captured[NAME] = taste.describe(split[0].detach())
        return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)

    def sampler_sample(executor, model_wrap, sigmas, extra_args, callback, noise,
                       latent_image=None, denoise_mask=None, disable_pbar=False):
        run = lambda n: executor(model_wrap, sigmas, extra_args, callback, n,
                                 latent_image, denoise_mask, disable_pbar)
        judge = live.get("judge")
        sampler = getattr(executor, "class_obj", None)
        if judge is None or noise.shape[0] != 1 or not hasattr(sampler, "max_denoise"):
            return run(noise)
        best = pick(judge, model_wrap, sigmas, extra_args, noise, latent_image, denoise_mask,
                    sampler.max_denoise(model_wrap, sigmas))
        return run(best)

    def pick(judge, model_wrap, sigmas, extra_args, noise, latent_image, denoise_mask, max_denoise):
        from comfy.samplers import KSamplerX0Inpaint
        options = dict(extra_args.get("model_options") or {})
        options["transformer_options"] = {**options.get("transformer_options", {}),
                                          dit_hooks.PROBE: True}
        named = {"latent_shapes": getattr(model_wrap.inner_model, "latent_shapes", None)}
        seed = int(extra_args.get("seed") or 0)
        sigma = sigmas[0] * noise.new_ones([noise.shape[0]])
        scored = []
        for k in range(count):
            if k == 0:
                candidate = noise                                    # the seed you asked for
            else:
                gen = torch.Generator().manual_seed(seed + 7919 * k)
                candidate = torch.randn(noise.shape, generator=gen).to(noise)
            model_k = KSamplerX0Inpaint(model_wrap, sigmas)
            model_k.latent_image, model_k.noise = latent_image, candidate
            x = model_wrap.inner_model.model_sampling.noise_scaling(
                sigmas[0], candidate, latent_image, max_denoise)
            x0 = model_k(x, sigma, denoise_mask=denoise_mask, model_options=options,
                         seed=extra_args.get("seed"))
            split = streams.video_of(x0, named)
            if split is None:
                _say("off this run: could not find the picture to score")
                return noise
            scored.append((judge.score(split[0]), k, candidate))
        score, k, best = max(scored, key=lambda s: s[0])
        log.info("FunPack Seed search",
                 ("kept your seed" if k == 0 else f"switched to candidate {k + 1}")
                 + f" of {count} (scores {', '.join(f'{s:.3f}' for s, _k, _c in scored)})")
        return best

    patcher.add_wrapper_with_key(WrappersMP.SAMPLER_SAMPLE, key, sampler_sample)
    return f"{count} seeds, one step each; learns every run, searches from 10 rated clips"


PROVIDES = {"modifier": install}
