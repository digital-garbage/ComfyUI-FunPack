"""Text encoder loader. A node-only module."""

from .nodes import FunPackCLIPLoader

ID = "loader_clip"
TITLE = "CLIP loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackCLIPLoader]
