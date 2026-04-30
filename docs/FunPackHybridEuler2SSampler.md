# FunPack Hybrid Euler 2S Sampler

This node outputs a ComfyUI `SAMPLER` object that can be connected directly into `CustomSamplerAdvanced` / `SamplerCustomAdvanced`.

## Purpose

It is designed as a quality/speed compromise between:

- `euler_ancestral`: fast, reliable, one model evaluation per step
- deterministic 2S-style ODE refinement: higher quality, but more expensive because it needs extra evaluations

This sampler keeps classic Euler ancestral for the early "structure building" stage where motion, anatomy and main composition are forming. On the late denoise steps it switches to a deterministic Euler / DPM-Solver++(2S) ODE refinement path where fine detail and cleanup usually matter most.

It can also run an optional paper-style Restart pass during that late quality stage:

- re-noise the current latent up to a higher sigma
- replay a short late-step sigma interval with the deterministic refinement path
- repeat that replay multiple times if requested

## Parameters

**eta**: Ancestral stochasticity. `1.0` keeps normal ancestral behavior.

**s_noise**: Noise multiplier for ancestral noise injection.

**high_quality_pct**: Fraction of the *late* denoise steps that switch from Euler ancestral into deterministic quality refinement.  
Example: `0.35` means only the last 35% of steps use the ODE refinement path.

**correction_blend**: Blend between late-step Euler ODE and late-step 2S correction.

- `0.0` = pure late-step Euler ODE
- `1.0` = full late-step DPM++(2S)-style correction

**restart_steps**: How many previous late-phase sigma steps are revisited during Restart replay.

**restart_repeats**: How many Restart loops to run. `0` disables Restart.

**restart_trigger_pct**: Sampling progress point where Restart is triggered. This value is clamped so Restart only happens inside the late quality phase.

**restart_noise**: Noise strength used when re-noising the latent up to the restart interval's higher sigma.

## Recommended starting values

- `eta = 1.0`
- `s_noise = 1.0`
- `high_quality_pct = 0.30` to `0.40`
- `correction_blend = 1.0`
- `restart_repeats = 0` for baseline testing

For a first Restart test, try:

- `high_quality_pct = 0.35`
- `correction_blend = 1.0`
- `restart_steps = 3`
- `restart_repeats = 1`
- `restart_trigger_pct = 0.85`
- `restart_noise = 1.0`

## Expected behavior

Compared to plain `euler_ancestral`, this sampler should usually:

- preserve more late-stage detail
- clean up texture and edge quality better
- keep the early motion/anatomy formation more lively
- cost less than running a heavier deterministic solver for the whole schedule

With Restart enabled, it may further improve late cleanup and detail consistency, but it will increase runtime and may be too aggressive for some video workflows if overused.

## Limitation

This node improves the **sampler-side quality/speed tradeoff**. It does **not** reduce the underlying cost of CFG++ guidance itself. If the goal becomes specifically "CFG++ quality at lower guidance cost", that likely requires a custom `GUIDER`, not just a custom `SAMPLER`.
