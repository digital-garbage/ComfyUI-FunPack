"""ComfyUI entry point for FunPack v5.

Core mounts the app and the module loader. Nodes are announced by modules, never
listed here.

There is deliberately NO `NODE_CLASS_MAPPINGS` in this file. ComfyUI's loader
checks for it FIRST and only falls through to `comfy_entrypoint` in an `elif`
(nodes.py), so merely defining an empty dict registers zero nodes and silently
skips the entrypoint entirely -- an empty mapping is a valid V1 answer, so there
is no error to notice. Declaring nothing is what makes the V3 path run.

Imports here are RELATIVE. ComfyUI loads a pack with `spec_from_file_location`
and never puts its directory on `sys.path`, so `from core import ...` finds
nothing -- and were it to work, a top-level name as generic as `core` would be
one collision away from another pack's. The cost is that this file cannot be
imported outside a package context, which is what the guard below is for: test
collectors walk into it, and there is nothing here they need.
"""

# ComfyUI loads everything under this directory into its OWN graph frontend.
# The Cutting Room app lives in app/ and must never be served from here.
WEB_DIRECTORY = "./web"

try:
    from .core import log
except ImportError:                 # imported outside a package; nothing to set up
    log = None
else:
    try:
        from .core import routes    # noqa: F401  (registers routes on import)
    except Exception as exc:        # a broken route must not take ComfyUI down
        log.failed("core.routes", exc)


async def comfy_entrypoint():
    """Hand ComfyUI the nodes that modules announced."""
    from .core import nodes
    return nodes.extension()


__all__ = ["WEB_DIRECTORY", "comfy_entrypoint"]
