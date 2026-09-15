"""Checkpoint loader. A node-only module."""

from .nodes import FunPackCheckpointLoader

ID = "loader_checkpoint"
TITLE = "Checkpoint loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackCheckpointLoader]
