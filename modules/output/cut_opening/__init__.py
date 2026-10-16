"""Cut the opening: let the starting image do its job, then remove it from the clip.

An i2v anchor pinned at full strength transfers identity, style and composition
better than anything that softens it (ALG blurs it, identity tokens approximate
it) -- but it is also literally the first frame you see, so every i2v shot opens
on the exact reference still. Generate normally, then drop N frames off the
front, and the sound by the same amount of time.

Cut on DECODED frames, never on the latent (v4 learned both reasons): a causal
video VAE decodes latent frame 0 as the time origin, so a promoted survivor
decodes wrong; and H3's latent sits on a 5k+2 grid, so most cut counts leave a
length the VAE cannot decode at all. On pixels, N means N.

Nothing is regrown to replace what is cut -- v4 tried, and the invented ending
consistently moved worse. The clip just comes out shorter.
"""

from .nodes import FunPackCutOpening

ID = "output_cut_opening"
TITLE = "Cut opening"
STAGE = "post"
CATEGORY = "post"
STATUS = "proven"

NODES = [FunPackCutOpening]
