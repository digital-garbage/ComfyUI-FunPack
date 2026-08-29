"""Collecting the ComfyUI nodes that modules announced.

Core holds no list of nodes. It walks the registry, takes whatever each module
declared in `NODES`, and hands the result to ComfyUI -- the same inversion the
panels use, applied to node registration.

Nothing here may raise. ComfyUI wraps `comfy_entrypoint` in one try/except and
skips the ENTIRE pack on any exception (nodes.py, the `elif` branch), so a single
bad module must not take every other node down with it. Every failure is caught,
logged once, and turns into that node's absence.
"""

from typing import List, Optional, Tuple

from . import log, registry as registry_mod

# Node ids are global in ComfyUI: `NODE_CLASS_MAPPINGS[schema.node_id]`, with no
# collision check -- a clash silently overwrites whoever registered first. The
# prefix is what keeps FunPack out of everyone else's namespace.
PREFIX = "FunPack"


def _schema_of(node) -> Optional[object]:
    """The node's V3 schema, or None if it cannot produce one."""
    getter = getattr(node, "GET_SCHEMA", None)
    if not callable(getter):
        return None
    return getter()


def collect(registry=None) -> Tuple[List[type], List[Tuple[str, str]]]:
    """(nodes, rejected) -- rejected carries the reason, for the modules dump."""
    reg = registry if registry is not None else registry_mod.current()
    nodes: List[type] = []
    rejected: List[Tuple[str, str]] = []
    claimed = {}                                 # node_id -> module id

    # Sorted so the set of registered nodes does not depend on filesystem order.
    for spec in sorted(reg.specs.values(), key=lambda s: s.id):
        for node in spec.nodes:
            # Named inside the guard, and starting from a name that cannot fail.
            # `getattr(node, "__name__", repr(node))` looks safe and is not:
            # Python evaluates the default eagerly, so a class whose metaclass
            # raises in __repr__ threw BEFORE the try and took every other
            # module's nodes with it -- the one thing this loop exists to stop.
            where = f"{spec.id}.<unnamed node>"
            try:
                name = getattr(node, "__name__", None)
                if isinstance(name, str) and name:
                    where = f"{spec.id}.{name}"
                schema = _schema_of(node)
            except Exception as exc:             # noqa: BLE001 -- absence, not a crash
                log.broke(where, exc, "describing itself")
                rejected.append((where, f"{type(exc).__name__}: {exc}"))
                continue

            if schema is None:
                rejected.append((where, "is not a comfy_api io.ComfyNode (no GET_SCHEMA)"))
                log.warning(where, "is not a ComfyUI V3 node and was not registered")
                continue

            node_id = getattr(schema, "node_id", None)
            if not isinstance(node_id, str) or not node_id.startswith(PREFIX):
                rejected.append((where, f"node_id {node_id!r} does not start with {PREFIX!r}"))
                log.warning(where, f"declares node_id {node_id!r}, which is not prefixed "
                                   f"{PREFIX!r}, and was not registered")
                continue

            if node_id in claimed:
                # ComfyUI would let the second one silently replace the first,
                # and which one wins would depend on import order.
                rejected.append((where, f"duplicate node_id {node_id!r}, already "
                                        f"declared by {claimed[node_id]}"))
                log.warning(where, f"reuses node_id {node_id!r}, already declared by "
                                   f"{claimed[node_id]}, and was not registered")
                continue

            claimed[node_id] = where
            nodes.append(node)

    return nodes, rejected


def extension():
    """A ComfyExtension over whatever announced itself, or None without ComfyUI."""
    try:
        from comfy_api.latest import ComfyExtension
    except Exception as exc:                     # noqa: BLE001 -- not inside ComfyUI
        log.failed("comfy_api.latest", exc)
        return None

    nodes, rejected = collect()

    class FunPackExtension(ComfyExtension):
        async def get_node_list(self) -> List[type]:
            return list(nodes)

    if rejected:
        log.warning("nodes", f"{len(rejected)} node(s) were not registered; "
                    + "; ".join(f"{where}: {why}" for where, why in rejected))
    log.info("nodes", f"registered {len(nodes)} node(s)")
    return FunPackExtension()
