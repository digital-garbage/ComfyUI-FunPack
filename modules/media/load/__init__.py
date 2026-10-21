"""Media loading.

A node-only module: no settings, no panel, no mount. It exists to contribute one
ComfyUI node, which is why `SETTINGS` is absent rather than empty.
"""

from .nodes import FunPackLoadMedia

ID = "media_load"
TITLE = "Media loader"
STAGE = "load"
CATEGORY = "system"
STATUS = "proven"

NODES = [FunPackLoadMedia]


#: Two dev builds briefly shipped a node each for video and audio; Load Media does both now.
_GONE = ("FunPackLoadMediaVideo", "FunPackLoadMediaAudio")


def upgrade(slots, outputs):
    """Pipelines saved while those nodes existed lose their slots (the app wires references per run) and any wire from them."""
    dead = {s["id"] for s in slots if s.get("node") in _GONE}
    if not dead:
        return slots
    out = []
    for s in slots:
        if s["id"] in dead:
            continue
        inputs = {k: v for k, v in (s.get("inputs") or {}).items()
                  if not (isinstance(v, (list, tuple)) and v and v[0] in dead)}
        out.append({**s, "inputs": inputs})
    return out


PROVIDES = {"upgrade_slots": upgrade}
