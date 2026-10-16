"""Diffusion model loader. A node-only module."""

from .nodes import FunPackDiffusionModelLoader

ID = "loader_diffusion_model"
TITLE = "Diffusion model loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackDiffusionModelLoader]


def routes(table, base, web):
    @table.get(base + "/torch")
    async def _torch(_req):
        """Why int8 would run slow on this torch build (None when it would not), for the chip beside Generate."""
        from ..common import slow_int8_build
        return web.json_response({"slow_int8": slow_int8_build()})


PROVIDES = {"routes": routes}
