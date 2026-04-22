# FunPack Hybrid Euler 2S Sampler

This node outputs a ComfyUI `SAMPLER` object that can be connected directly into `CustomSamplerAdvanced` / `SamplerCustomAdvanced`.

## Purpose

It is designed as a quality/speed compromise between:

- `euler_ancestral`: fast, reliable, one model evaluation per step
- full 2S-style samplers: higher quality, but more expensive because they need an extra evaluation every step

This sampler keeps classic Euler ancestral for most of the schedule, then applies a DPM-Solver++(2S)-style correction only on the late denoise steps where fine detail and cleanup usually matter most.

## Parameters

**eta**: Ancestral stochasticity. `1.0` keeps normal ancestral behavior.

**s_noise**: Noise multiplier for ancestral noise injection.

**high_quality_pct**: Fraction of the *late* denoise steps that receive the extra 2S correction.  
Example: `0.35` means only the last 35% of steps pay the extra compute cost.

**correction_blend**: Blend between pure Euler ancestral and the late-step 2S correction.

- `0.0` = behaves like Euler ancestral
- `1.0` = full late-step correction

## Recommended starting values

- `eta = 1.0`
- `s_noise = 1.0`
- `high_quality_pct = 0.30` to `0.40`
- `correction_blend = 1.0`

For a safer first test, try:

- `high_quality_pct = 0.25`
- `correction_blend = 0.7`

## Expected behavior

Compared to plain `euler_ancestral`, this sampler should usually:

- preserve more late-stage detail
- clean up texture and edge quality better
- cost noticeably less than a full 2S sampler

## Limitation

This node improves the **sampler-side quality/speed tradeoff**. It does **not** reduce the underlying cost of CFG++ guidance itself. If the goal becomes specifically “CFG++ quality at lower guidance cost”, that likely requires a custom `GUIDER`, not just a custom `SAMPLER`.
