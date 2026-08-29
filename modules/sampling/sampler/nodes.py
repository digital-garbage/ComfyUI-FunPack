"""The sampler. It samples, and owns nothing else.

v4's Chain Sampler became a hub: 8183 lines, roughly seventy optional inputs, a
`_sample_chunk` that grew six ALG arguments, and a hardcoded list of knobs that
are silently inert on one model family. Every technique that needed the loop was
added to it, so it accumulated all of them.

This one picks an algorithm and a schedule from what is available and runs them.
Three things keep it from becoming a hub again:

* **It does not know what a modifier is.** Model-side ones ride on the model and
  it never learns they exist. Sampler-side ones reach it through ONE call site,
  `chain.process(ctx, latent)`, which cannot grow because the step travels in a
  context object rather than in the signature.
* **It declares what it can host** (`ACCEPTS`) and core hands it only the
  modifiers that fit. It never names one.
* **Choices are announced, not listed.** The algorithms and schedules are read
  from ComfyUI's own registries, so a new one appears without an edit here.
"""

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import torch
from comfy_api.latest import io

from ..._core import chain as chain_mod, log, patching, registry as registry_mod, run as run_mod
from ..._core import relations as relations_mod, schema as schema_mod, traits as traits_mod

# The hook points this sampler offers. A modifier asking for anything else is
# absent and said, rather than half-running.
ACCEPTS = ("latent",)


def sampler_names():
    """Read from ComfyUI, so a sampler it gains appears here without an edit."""
    return list(comfy.samplers.SAMPLER_NAMES)


def scheduler_names():
    return list(comfy.samplers.SCHEDULER_NAMES)


class FunPackSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        samplers, schedulers = sampler_names(), scheduler_names()
        return io.Schema(
            node_id="FunPackSampler",
            display_name="FunPack Sampler",
            category="FunPack/Sampling",
            description="Sample a latent with a chosen algorithm and schedule.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=1000),
                io.Float.Input("cfg", default=7.0, min=0.0, max=100.0, step=0.1,
                               tooltip="1.0 turns guidance off, which some models want."),
                io.Combo.Input("sampler_name", options=samplers,
                               default="euler" if "euler" in samplers else samplers[0]),
                io.Combo.Input("scheduler", options=schedulers,
                               default="normal" if "normal" in schedulers else schedulers[0]),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               optional=True,
                               tooltip="Below 1 starts partway through the schedule."),
                io.Custom("FUNPACK_SETTINGS").Input("settings", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent, seed: int, steps: int, cfg: float,
                sampler_name: str, scheduler: str, denoise: float = 1.0,
                settings=None) -> io.NodeOutput:
        if settings is not None and not isinstance(settings, dict):
            raise RuntimeError(
                f"FunPack Sampler: settings must be an object keyed by module id, "
                f"got {type(settings).__name__}.")

        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

        # A generation begins here, every time. Taking the id only when none
        # exists left every run after the first reporting as the first, because
        # the node that used to start them is cached away by ComfyUI.
        run = run_mod.start()

        # The record travels on the model, which ComfyUI may hand us again from
        # its cache -- so it is emptied here rather than trusted to be fresh.
        # Otherwise a modifier dropped in one generation stays dropped in every
        # later one, with nothing said.
        dropped = getattr(model, "funpack_dropped", None) or patching.Dropped()
        dropped.clear()

        latent = latent.copy()
        # The latent says what ratios it was BUILT at; this is where a model that
        # wants different ones gets it rescaled. Skipping it silently samples a
        # latent of the wrong size -- and our own empty-latent node reports those
        # ratios expecting somebody to read them.
        samples_in = comfy.sample.fix_empty_latent_channels(
            model, latent["samples"],
            latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
        latent["samples"] = samples_in

        sigmas = cls._sigmas(model, scheduler, steps, denoise)
        sampler = comfy.samplers.sampler_object(sampler_name)

        chain, notes = cls._chain(model, settings, dropped)
        if chain:
            sampler = cls._with_chain(sampler, chain, sigmas, run)

        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        # Progress and previews, from ComfyUI's own machinery rather than ours:
        # it already picks a preview method, already falls back when the decoder
        # is missing, and already reports steps to whatever is watching.
        callback = latent_preview.prepare_callback(model, max(0, len(sigmas) - 1))

        try:
            samples = guider.sample(
                Noise_RandomNoise(seed).generate_noise(latent),
                samples_in, sampler, sigmas,
                denoise_mask=latent.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=seed,
            )
        except comfy.model_management.InterruptProcessingException:
            # Said, then re-raised. Cancelling is a normal thing to do and the
            # log should show it happening rather than a run simply stopping --
            # but it is not ours to absorb: ComfyUI ends the queue item.
            log.alert("FunPack Sampler", f"{run} was cancelled part way through")
            raise
        # Back to the device every other sampler returns on. Leaving it on the
        # GPU works until something downstream is on the CPU, and then it is a
        # device mismatch a long way from here.
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        # Consumed above; carrying them onward would rescale a second time.
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        headline = (f"{run}: {sampler_name} / {scheduler}, {len(sigmas) - 1} step(s), "
                    f"cfg {cfg}")
        log.info("FunPack Sampler", headline)
        return io.NodeOutput(out, "\n".join([headline, *notes]))

    # --- the parts, kept separate so each is testable on its own -----------

    @classmethod
    def _sigmas(cls, model, scheduler: str, steps: int, denoise: float):
        """The schedule, including the partial-denoise case.

        Computed over the FULL step count and then trimmed, which is what makes
        denoise 0.5 mean "the second half of a 20-step schedule" rather than "a
        10-step schedule" -- the two are not the same curve.
        """
        if denoise is None or denoise >= 1.0:
            total = steps
        else:
            if denoise <= 0.0:
                return torch.FloatTensor([])
            total = int(steps / denoise)

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total)
        return sigmas[-(steps + 1):]

    @classmethod
    def _chain(cls, model, settings, dropped):
        """Whoever asked for a hook point this sampler offers."""
        registry = registry_mod.current()
        specs = list(registry.specs.values())
        available = traits_mod.traits_of(model, specs)

        offering = [s for s in specs if s.provides.get("sampler_modifier")]
        compatible, incompatible = traits_mod.split(offering, available)
        ordered, rejected = relations_mod.order(compatible)

        guards_off = any(answer(settings or {}) for _spec, answer
                         in registry.providers("guards_off"))
        values, problems = {}, []
        for spec in ordered:
            clean, said = schema_mod.check_values(
                spec, (settings or {}).get(spec.id), keep_bad=guards_off)
            values[spec.id] = clean
            problems.extend(said)
        schema_mod.refuse_or_warn(problems, "FunPack Sampler", guards_off=guards_off)

        chain, notes = chain_mod.build(ordered, values, ACCEPTS, dropped)
        for spec in incompatible:
            notes.append(f"{spec.id}: needs {', '.join(traits_mod.missing_for(spec, available))}")
        for spec, why in rejected:
            notes.append(f"{spec.id}: {why}")
        return chain, notes

    @classmethod
    def _with_chain(cls, sampler, chain, sigmas, run):
        """The one call site, wrapped around whatever algorithm was chosen.

        The algorithm is not modified and not replaced: its own loop runs, and
        the chain sees each step through the denoiser it already calls.
        """
        inner = getattr(sampler, "sampler_function", None)
        if inner is None:
            log.alert("FunPack Sampler",
                      f"{type(sampler).__name__} has no loop to attach to, so "
                      f"{', '.join(chain.ids)} will not run this time")
            return sampler

        total = max(1, len(sigmas) - 1)

        def with_modifiers(denoiser, x, step_sigmas, extra_args=None, callback=None,
                           disable=None, **options):
            seen = {"index": 0}

            class Watched:
                """Transparent, except that every model call is a step."""

                def __getattr__(self, name):
                    return getattr(denoiser, name)

                def __call__(self, latent, sigma, **kwargs):
                    ctx = chain_mod.Step(
                        index=seen["index"], sigma=sigma, sigmas=step_sigmas,
                        total=total, run=run,
                        anchor=getattr(denoiser, "latent_image", None))
                    latent = chain.process(ctx, latent)
                    seen["index"] += 1
                    return denoiser(latent, sigma, **kwargs)

            return inner(Watched(), x, step_sigmas, extra_args=extra_args,
                         callback=callback, disable=disable, **options)

        with_modifiers.__name__ = getattr(inner, "__name__", "sampler") + "_funpack"
        return comfy.samplers.KSAMPLER(
            with_modifiers,
            extra_options=getattr(sampler, "extra_options", {}),
            inpaint_options=getattr(sampler, "inpaint_options", {}))
