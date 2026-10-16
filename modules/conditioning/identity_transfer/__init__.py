"""Identity transfer (Best-FaceID): a native port, not a splice of an external node.

A node-only module. It needs an LTX audio+video model (`ltx_av`): the reference face is injected
into that model's token sequence (see patch.py) and, optionally, an ArcFace embedding goes through a
trained projector into the text context (projector.py). H3 has no equivalent: the projector is
trained against LTX's text context and the overlap tokens against LTX's RoPE.
"""

from .nodes import FunPackIdentityTransfer

ID = "conditioning_identity_transfer"
TITLE = "Identity transfer"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"
REQUIRES = ["ltx_av"]

NODES = [FunPackIdentityTransfer]
