"""Save the finished video: MP4 (H.264), on the GPU's encoder when there is one, else on every CPU core.

Core's SaveVideo encodes on one CPU thread; on a rental that was most of the time after decoding.
"""

from .nodes import FunPackSaveVideo

ID = "output_save"
TITLE = "Save video"
STAGE = "post"
CATEGORY = "post"
STATUS = "proven"

NODES = [FunPackSaveVideo]


def _link(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def upgrade(slots, outputs):
    """Pipelines saved before this node hold core's CreateVideo -> SaveVideo: one FunPack Save Video instead.
    A SaveVideo left with no feed takes the one picture source nothing else reads (its sound too).
    `outputs(node)` -> that node's output types. -> the slots, changed or not."""
    slots = [dict(s) for s in slots]
    by_id = {s["id"]: s for s in slots}
    read = {v[0] for s in slots for v in (s.get("inputs") or {}).values() if _link(v)}
    drop = set()
    for s in slots:
        if s["node"] != "SaveVideo":
            continue
        feed = (s.get("inputs") or {}).get("video")
        src = by_id.get(feed[0]) if _link(feed) else None
        new = {"filename_prefix": (s.get("inputs") or {}).get("filename_prefix", "FunPack")}
        roles = s.get("roles")
        if src is not None and src["node"] == "CreateVideo":
            new.update({k: v for k, v in (src.get("inputs") or {}).items() if k in ("images", "audio", "fps")})
            roles = src.get("roles") or roles
            if sum(1 for o in slots for v in (o.get("inputs") or {}).values() if _link(v) and v[0] == src["id"]) == 1:
                drop.add(src["id"])
        elif src is None:
            loose = [o for o in slots if o["id"] not in read and "IMAGE" in outputs(o["node"])]
            if len(loose) != 1:
                continue
            kinds = outputs(loose[0]["node"])
            new["images"] = [loose[0]["id"], kinds.index("IMAGE")]
            if "AUDIO" in kinds:
                new["audio"] = [loose[0]["id"], kinds.index("AUDIO")]
        else:
            continue
        s.update(node="FunPackSaveVideo", inputs=new)
        if roles:
            s["roles"] = roles
    return [s for s in slots if s["id"] not in drop]


PROVIDES = {"upgrade_slots": upgrade}
