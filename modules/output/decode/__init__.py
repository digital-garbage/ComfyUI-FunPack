"""Decode. A node-only module."""

from .nodes import FunPackDecode

ID = "output_decode"
TITLE = "Decode"
STAGE = "post"
CATEGORY = "post"
STATUS = "proven"

NODES = [FunPackDecode]
