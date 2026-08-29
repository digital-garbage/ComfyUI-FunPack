"""How a module reaches core, in both contexts, decided once.

A module cannot simply `import core`. Inside ComfyUI the pack is imported by file
location, so `core` is only reachable beneath the pack's own package; under the
tests and the dev server the repo root IS on sys.path, so it is top-level. A
static relative import gets one of those right and breaks the other.

Every module therefore imports core THROUGH here -- `from ..._core import traits`
-- and this file holds the only place that has to know about the difference. The
leading underscore keeps it out of the module scan.
"""

try:                                             # root on sys.path: tests, dev server
    from core import log, registry, schema, traits
except ImportError:                              # inside ComfyUI: a subpackage of the pack
    from ..core import log, registry, schema, traits

__all__ = ["log", "registry", "schema", "traits"]
