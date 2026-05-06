# FunPack Hybrid Euler 2S Sampler

This node outputs a ComfyUI `SAMPLER` and can also take optional `SIGMAS` input and output modified `SIGMAS` for use with `CustomSamplerAdvanced` / `SamplerCustomAdvanced`.

## Purpose

It is designed as a quality/speed compromise between:

- `euler_ancestral`: fast, reliable, one model evaluation per step
- deterministic 2S-style ODE refinement: higher quality, but more expensive because it needs extra evaluations

This sampler keeps classic Euler ancestral for the early "structure building" stage where motion, anatomy and main composition are forming. On the late denoise steps it switches to a deterministic Euler / DPM-Solver++(2S) ODE refinement path where fine detail and cleanup usually matter most.

It can also apply optional early/mid **motion pulses** for single-clip image-to-video workflows. These pulses add monotonic noise kicks at selected normal denoise steps instead of inserting upward sigma jumps. This is intended to push LTX2.3 away from stale frame-1 reference stiffness while avoiding the audio damage caused by Restart replay.

Restart replay has been removed because upward sigma replay can break LTX audio generation.

For hard viewpoint or scene-context resets, use `FunPack Context Transition Windows` before sampling. That node changes which frames the model can see in each denoise evaluation; this sampler only changes the denoise trajectory.

## Recommended wiring

Use this node with `SamplerCustomAdvanced` or `CustomSamplerAdvanced`:

- connect your base scheduler output into this node's optional `sigmas` input
- connect this node's `sampler` output into the sampler input
- connect this node's `sigmas` output into the sigmas input

This node is not intended for `KSampler`.

## Parameters

**eta**: Ancestral stochasticity. `1.0` keeps normal ancestral behavior.

**s_noise**: Noise multiplier for ancestral noise injection.

**high_quality_pct**: Fraction of the *late* denoise steps that switch from Euler ancestral into deterministic quality refinement.  
Example: `0.35` means only the last 35% of steps use the ODE refinement path.

**correction_blend**: Blend between late-step Euler ODE and late-step 2S correction.

- `0.0` = pure late-step Euler ODE
- `1.0` = full late-step DPM++(2S)-style correction

**motion_pulse_mode**: Anti-stiffness motion pulse preset:

- `off`: preserve legacy sampler behavior.
- `balanced`: one moderate early/mid pulse.
- `aggressive`: at least two stronger early/mid pulses for stale image-to-video generations.
- `custom`: use the transition count, spacing, and strength exactly as configured.

**motion_pulse_start_pct**: Sampling progress point where the first motion pulse is applied.

**motion_pulse_count**: Number of requested early/mid motion pulses. Pulses that would land in the late quality phase are skipped.

**motion_pulse_spacing_pct**: Progress spacing between motion pulses.

**motion_pulse_strength**: Strength of the monotonic noise kick. Higher values push harder against stale image references, with more drift risk.

**sigmas**: Optional incoming sigma schedule. If connected, the node returns the same monotonic schedule plus sampler-side metadata for motion pulses. If not connected, the sampler computes pulse positions from the runtime schedule.

## Recommended starting values

- `eta = 1.0`
- `s_noise = 1.0`
- `high_quality_pct = 0.30` to `0.40`
- `correction_blend = 1.0`
- `motion_pulse_mode = off` for baseline testing

For an aggressive LTX2.3 image-to-video motion test, try:

- `high_quality_pct = 0.35`
- `correction_blend = 1.0`
- `motion_pulse_mode = aggressive`
- `motion_pulse_start_pct = 0.30`
- `motion_pulse_count = 2`
- `motion_pulse_spacing_pct = 0.22`
- `motion_pulse_strength = 0.85`

Keep the prompt/refiner conditioning explicit about both action and camera change, for example `orbiting camera`, `dolly in`, `zoom out`, `new side angle`, `turning`, or `dynamic pose change`.

## Expected behavior

Compared to plain `euler_ancestral`, this sampler should usually:

- preserve more late-stage detail
- clean up texture and edge quality better
- keep the early motion/anatomy formation more lively
- cost less than running a heavier deterministic solver for the whole schedule

With motion pulses enabled, it should more strongly encourage action, camera movement, and viewpoint changes inside a single clip. Stronger settings can increase subject drift or visual instability.

The outgoing `SIGMAS` remain monotonic. Motion pulses happen inside the sampler at selected denoise steps rather than by expanding the schedule.

If you need the model to stop carrying the previous segment's temporal context forward, add `FunPack Context Transition Windows`. It is the stronger transition tool and should be preferred for multi-view or "scene reset" testing.

## Limitation

This node improves the **sampler-side quality/speed tradeoff**. It does **not** reduce the underlying cost of CFG++ guidance itself. If the goal becomes specifically "CFG++ quality at lower guidance cost", that likely requires a custom `GUIDER`, not just a custom `SAMPLER`.
