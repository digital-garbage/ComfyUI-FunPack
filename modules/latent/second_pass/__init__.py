"""Second pass: the between-passes latent operation. A node-only module.

v4 built the second pass INTO its chain sampler: a toggle, a second schedule, a
second sampler, an operation. In v5 a second pass is just graph: sampler ->
FunPack Latent Resample -> a second FunPack Sampler with denoise below 1 (which
re-noises the finished clip to where its schedule starts). Nothing here samples.

What this module adds is the part the graph could not already do:

* **Loading an upscaler ComfyUI does not know.** Model modules announce
  `latent_upscaler` (state dict -> upscaler or None); H3's is the reason.
* **Resampling a video latent between passes**, sharpen (up then back down --
  pass 2 re-denoises the added detail) or upscale (kept larger, pass 2 runs at
  the new size).
* **Keeping the conditioning valid after an upscale.** Model modules announce
  `rescale_conditioning`; H3 packs its anchor pin as rows sized to the grid, so
  without this pass 2 refuses the anchor outright.

v4's user verdict: the H3 second pass WORKS (2026-09-20).
"""

from .nodes import FunPackLatentResample, FunPackLatentUpscalerLoader

ID = "latent_second_pass"
TITLE = "Second pass"
STAGE = "latent"
CATEGORY = "sampling"
STATUS = "proven"

NODES = [FunPackLatentUpscalerLoader, FunPackLatentResample]
