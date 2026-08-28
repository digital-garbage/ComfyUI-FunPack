"""ComfyUI entry point for FunPack v5.

Core mounts the app and the module loader. Nodes are announced by modules, never
listed here.
"""

from core import log

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# ComfyUI loads everything under this directory into its OWN graph frontend.
# The Cutting Room app lives in app/ and must never be served from here.
WEB_DIRECTORY = "./web"

try:
    from core import routes  # noqa: F401  (registers routes on import)
except Exception as exc:  # a broken route must not take ComfyUI down with it
    log.failed("core.routes", exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
