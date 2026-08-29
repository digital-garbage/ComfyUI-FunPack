"""LoRA loader. A node-only module."""

from .nodes import FunPackLoraLoader

ID = "loader_lora"
TITLE = "LoRA loader"
STAGE = "load"
STATUS = "proven"

NODES = [FunPackLoraLoader]
