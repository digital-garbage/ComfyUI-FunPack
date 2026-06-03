# FunPack Hybrid Euler 2S Sampler

This node outputs a ComfyUI `SAMPLER` and can also take optional `SIGMAS` input and output modified `SIGMAS` for use with `SamplerCustomAdvanced` / `CustomSamplerAdvanced`.

## Purpose

It is designed as a quality/speed compromise between:

- `euler_ancestral`: fast, reliable, one model evaluation per step
- deterministic 2S-style ODE refinement: higher quality, but more expensive because it needs extra evaluations

This sampler keeps classic Euler ancestral for the early "structure building" stage where motion, anatomy and main composition are forming. On the late denoise steps it switches to a deterministic Euler / DPM-Solver++(2S) ODE refinement path where fine detail and cleanup usually matter most.

It can also apply optional early/mid **motion pulses** for single-clip image-to-video workflows. These pulses add monotonic noise kicks at selected normal denoise steps instead of inserting upward sigma jumps. This is intended to push LTX2.3 away from stale frame-1 reference stiffness while avoiding the audio damage caused by Restart replay.

On rectified-flow models (CONST, e.g. LTXAV) the sampler runs an RF-correct port of the same feature set — a Heun corrector replaces the 2S step — rather than falling back to plain `euler_ancestral`. Velocity bias and reactive rescue therefore work on LTXV/LTXAV, not just the discrete-step models. On LTXAV, ancestral noise and trajectory steering are applied to the video latent only, never the audio latent, to avoid joint-attention audio corruption.

## Recommended wiring

Use this node with `SamplerCustomAdvanced` or `CustomSamplerAdvanced`:

- connect your base scheduler output into this node's optional `sigmas` input
- connect this node's `sampler` output into the sampler input
- connect this node's `sigmas` output into the sigmas input

## Parameters

**eta**: Ancestral stochasticity at the start of sampling. `1.0` keeps normal ancestral behaviour.

**eta_final**: Target eta value that ancestral noise decays toward. When set below `eta`, noise strength decays linearly toward this value as **schedule progress** advances (decoupled from the quality-phase boundary as of 2.7.7, so decay tracks the sampling timeline rather than a quality threshold). Lower values give a cleaner hand-off into deterministic refinement. Set equal to `eta` to disable decay.

**s_noise**: Noise multiplier for ancestral noise injection.

**high_quality_pct**: Fraction of the *late* denoise steps that switch from Euler ancestral into deterministic quality refinement. Example: `0.35` means the last 35% of steps use the ODE refinement path.

**correction_blend**: Blend between late-step Euler ODE and late-step 2S correction for the second half of quality-phase steps.

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

**velocity_bias_mode**: Experimental early-velocity steering mode. Captures/applies averaged early model velocity around normalized sigma `0.9`/`0.8`. `off` preserves legacy behavior.

**velocity_bias_strength**: Strength of the remembered velocity (action) injected at the structure sigma. `0` disables. `~0.15` = subtle spice; `0.3`–`1.0` = clear action crossover; `2`–`3` approaches full action replacement (capped so the current generation isn't wiped). This is a **creative / action-injection tool, not a consistency tool** — it also carries an emergent scene-cut prior (cuts survive the trajectory mean and can reappear unprompted). The bias is applied magnitude-preservingly with sigma-decay so it injects motion without washing out detail. Range `0.0`–`3.0`.

**velocity_bias_source**: How velocity bias and rescue pick a good direction. `mean` = prompt-blind global average (legacy). `nearest` = single best-matching prompt cluster, which preserves one real good generation's detail instead of a washed-out average (less softening). Affects both apply and rescue.

**velocity_refinement_key**: Memory key used to capture/apply early velocity bias and rescue trajectories. Wire this to Studio's refinement key so memory groups by the same key the Refiner uses. Trajectory banks persist to disk.

**rescue_mode**: Reactive in-flight rescue, rating-gated. Steers each eligible step toward trajectories you rated good and away from ones you rated `Awful` (matched to the current prompt). Learns automatically from ratings while on — no separate capture step. It is a no-op until you've rated a few generations for this prompt/key (a positive rating builds the target, an `Awful` builds what to avoid).

**rescue_threshold**: Fires when the step has diverged from the good trajectory by more than this (`1 - cosine`) **or** aligned with a bad trajectory by more than this (cosine). Lower corrects more eagerly; `0.10`–`0.20` typical, raise toward `0.4+` for only severe cases.

**rescue_strength**: How hard to pull toward good / push away from bad when triggered (magnitude preserved, no energy injected). Keep moderate; `0.5` is a strong correction.

**sigmas**: Optional incoming sigma schedule. If connected, the node returns the same schedule plus sampler-side metadata for motion pulses.

## Recommended starting values

- `eta = 1.0`, `eta_final = 1.0`
- `s_noise = 1.0`
- `high_quality_pct = 0.30` to `0.40`
- `correction_blend = 1.0`
- `motion_pulse_mode = off` for baseline testing
- `velocity_bias_mode = off`, `velocity_bias_strength = 0.0`
- `rescue_mode = off` (turn on once you have a few rated generations for the prompt/key)

To smooth the transition into the quality phase, try `eta_final = 0.5`. This decays ancestral noise from `1.0` at the start toward `0.5` as sigma approaches the quality boundary.

For an aggressive LTX2.3 image-to-video motion test:

- `high_quality_pct = 0.35`
- `correction_blend = 1.0`
- `motion_pulse_mode = aggressive`
- `motion_pulse_start_pct = 0.30`
- `motion_pulse_count = 2`
- `motion_pulse_spacing_pct = 0.22`
- `motion_pulse_strength = 0.85`

## Expected behavior

Compared to plain `euler_ancestral`, this sampler should usually:

- preserve more late-stage detail
- clean up texture and edge quality better
- keep the early motion/anatomy formation more lively
- cost less than running a heavier deterministic solver for the whole schedule

The early phase uses a second-order denoised extrapolation (Adams-Bashforth 2-step) that reuses the previous step's denoised estimate to improve the score direction at no extra model-call cost. The quality phase uses progressive correction blending: the first half of quality steps use single-eval Euler ODE, the second half use the full configured 2S correction, concentrating the expensive second model call where sigma is lowest.

Motion pulse state resets after any pulse fires because the pulse modifies the latent and invalidates the previous denoised estimate.

The outgoing `SIGMAS` remain monotonic. Motion pulses happen inside the sampler at selected denoise steps rather than by expanding the schedule.
