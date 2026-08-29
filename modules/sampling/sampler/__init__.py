"""The sampler. A node-only module."""

from .nodes import ACCEPTS, FunPackSampler

ID = "sampling_sampler"
TITLE = "Sampler"
STAGE = "sampling"
STATUS = "proven"

NODES = [FunPackSampler]
