# FunPack Context Transition Windows

`FunPack Context Transition Windows` is a model wrapper for forcing video context resets without changing sampler sigmas or adding latent noise. It borrows Comfy's manual context-window mechanism, then splits the window schedule at transition boundaries so the model stops seeing the entire previous segment when denoising the next one.

Use it before sampling:

```text
MODEL -> FunPack Context Transition Windows -> sampler/model input path
```

## Why This Exists

The sampler-level motion pulses are intentionally conservative: they add monotonic noise kicks, but they do not make the model forget earlier frames. This node targets the stronger behavior: “same scene, viewed differently” by reducing temporal context carryover across chosen parts of the clip.

It does not resample audio and does not add upward sigma jumps. Audio/video errors from Restart-style sigma replay should not be reintroduced by this node.

## Recommended LTX2.3 Starting Point

- `transition_mode`: `aggressive`
- `context_length`: smaller than the latent clip length, commonly `8` to `16`
- `context_overlap`: `2` to `4`
- `context_schedule`: `standard_static`
- `fuse_method`: `pyramid`
- `dim`: `2`
- `transition_start_pct`: `0.30`
- `transition_count`: `2`
- `transition_spacing_pct`: `0.22`
- `transition_strength`: `0.85` to `1.0`
- `cond_retain_index_list`: leave empty for transition testing

If `transition_start_pct` is `0.0`, the first boundary is placed immediately after latent frame `0`. That keeps the upstream image-to-video frame-1 anchor intact, but prevents later windows from treating frame `0` as normal continuous context.

## Behavior

- `off`: uses ordinary manual context windows with no transition boundaries.
- `balanced`: creates one context boundary and allows a small overlap bleed.
- `aggressive`: creates at least two context boundaries and blocks most cross-boundary context.
- `custom`: uses the values exactly as provided.

`transition_strength` controls isolation:

- `1.0`: hard segment split; windows do not cross transition boundaries.
- Lower values: allow a small number of overlap frames around the boundary, based on `context_overlap`.

For strong scene/viewpoint changes, use explicit prompt/refiner language such as orbit, dolly, push-in, pull-back, low angle, high angle, profile view, over-shoulder, or reveal. This node gives those concepts temporal room to form; the conditioning still needs to ask for them.
