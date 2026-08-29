"""Diffusion model loader. A node-only module."""

from .nodes import FunPackDiffusionModelLoader

ID = "loader_diffusion_model"
TITLE = "Diffusion model loader"
STAGE = "load"
STATUS = "proven"

NODES = [FunPackDiffusionModelLoader]
