# FunPack Shot Prompt Planner

`FunPack Shot Prompt Planner` rewrites one transition-heavy prompt into ordered shot clauses before text encoding. It is meant to sit after a prompt enhancer and before the text encoder / Video Refiner path.

## Why This Exists

Context windows can limit temporal memory, but they do not invent new semantic targets. If the conditioning remains one smooth paragraph, the model may still treat all concepts as one continuous scene.

This node detects transition language such as `then`, `after this`, `cut to`, `scene change`, `transition to`, `next`, and `finally`, then rewrites the text into explicit shot structure:

```text
Shot 1: ...
cinematic cut, new shot, changed camera context.
Shot 2: ...
```

The `structured_prompt` output is the simplest path: connect it to your text encoder, then pass the encoded conditioning through `FunPack Video Refiner`.

The individual `shot1` through `shot10` outputs are for stronger context-window workflows where each shot is encoded separately and combined before a manual context window node with `split_conds_to_windows` enabled.

## Recommended Workflow

Simple automatic path:

```text
Prompt Enhancer -> Shot Prompt Planner structured_prompt -> text encoder -> Video Refiner -> sampler
```

Stronger context-window path:

```text
Prompt Enhancer -> Shot Prompt Planner shot outputs -> text encoders / conditioning combine
MODEL -> Context Windows (Manual) with split_conds_to_windows enabled
combined conditioning + wrapped model -> sampler
```

## Parameters

**transition_enforcement**

- `off`: pass prompt through.
- `auto`: only structure prompts when transition triggers are detected.
- `soft`: always add shot structure and soft new-shot wording.
- `hard`: add hard new-shot wording that asks the model to drop previous scene context.

**shot_count**: `auto` uses detected clauses. A number forces that many shot slots.

**transition_style**: Default transition wording between shots.

**camera_intensity**: Adds subtle, medium, or strong camera/viewpoint changes per shot.

**global_style**: Persistent style or identity text appended to every shot.

**negative_hints**: Positive-side avoidance language for stiff outputs, such as static pose or locked camera.
