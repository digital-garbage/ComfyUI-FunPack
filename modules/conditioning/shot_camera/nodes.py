"""The Shot Camera node: the prompt text in, the prompt text out with cut times, views and camera moves in its [Shot N] blocks."""

import hashlib

from comfy_api.latest import io

from ..._core import control as control_mod, log, registry as registry_mod, schema as schema_mod, shortcuts
from . import engine, memory

ID = "conditioning_shot_camera"


def _values(settings):
    """This module's settings from the pipeline's settings object, checked against its own declaration."""
    spec = registry_mod.current().specs.get(ID)
    if spec is None or control_mod.all_off(settings):          # "disable all enhancements" is a switch for this too
        return {}
    clean, problems = schema_mod.check_values(spec, (settings or {}).get(ID))
    for problem in problems or ():
        log.once(f"shot_camera:{problem}", log.ALERT, "FunPack Shot Camera", f"setting ignored, default used: {problem}")
    return clean


def _pieces():
    """Every enabled shortcut's replacement text, longest first: how a cut tells where one shortcut ends and the next begins."""
    try:
        out = set()
        for sc in shortcuts.listing():
            if sc.enabled:
                out.update(r.strip() for r in sc.replacements if len(r.strip()) >= 15)
        return sorted(out, key=len, reverse=True)
    except Exception:                                          # noqa: BLE001 -- no library: nothing is split, times are still added
        return []


def _prompt_id():
    try:
        from server import PromptServer
        return getattr(PromptServer.instance, "last_prompt_id", None)
    except Exception:                                          # noqa: BLE001
        return None


def rewrite(text, values, seconds=None, pieces=None):
    """-> (new text, what happened, what the run chose for the rating that follows)."""
    seed = f"{engine.content_fingerprint(text)}:{values.get('variation', 0)}"       # the same prompt gives the same shots; 'variation' re-rolls
    said, arms, views = [], [], []
    # A payload from before the toggle had no cut_same_shot key: Shot cut times owned the chance.
    splitting = bool(values["cut_same_shot"]) if "cut_same_shot" in values else bool(values.get("shot_cuts"))
    if (values.get("shot_cuts") or splitting) and engine.SHOT.search(text or ""):
        chance = memory.split_chance(float(values.get("shot_cuts_chance", 0.5))) if splitting else 0.0
        text, info = engine.add_shot_cuts(text, seconds, seed=seed, chance=chance, pieces=_pieces() if pieces is None else pieces)
        arms += info.get("arms", [])
        said.append(f"cuts at {', '.join(info['times'])}" + (f" ({info['before']}→{info['after']} shots)" if info["after"] != info["before"] else "") if info["times"] else f"cuts: {info['why']}")
    if values.get("shot_views") and engine.SHOT.search(text or ""):
        skipped = []
        text, added = engine.add_shot_views(text, seed=seed, chance=float(values.get("shot_views_chance", 0.4)), stats=memory.view_stats(), skipped=skipped)
        views += [{"view": a["view"], "traits": a["traits"]} for a in added]
        said.append("views: " + (", ".join(f"shot {a['shot']} {a['view']}" for a in added) if added else f"none ({'; '.join(skipped) or 'nothing to do'})"))
    if values.get("camera_moves") and engine.SHOT.search(text or ""):
        chance = memory.effective_chance(float(values.get("camera_moves_chance", 0.7)))
        before = text
        text, report = engine.add_camera_moves(text, seed=seed, chance=chance, prior=memory.prior(), arms=memory.arm_stats())
        arms += [a for r in report for a in r.get("arms", [])]
        if report:
            memory.observe(engine.content_fingerprint(before), [w for r in report for w in r.get("lemmas", [])])
        moved = [r for r in report if r["move"]]
        said.append(f"moves: {len(moved)} of {len(report)} shots" + ("" if moved else " (" + "; ".join(f"shot {r['shot']} {r['why']}" for r in report) + ")"))
    details = []
    if values.get("detail_notes") and engine.SHOT.search(text or ""):
        base = float(values.get("detail_notes_chance", 0.5))
        bank = []
        for row in memory.detail_bank():
            row = dict(row)
            row["chance"] = memory.detail_chance(base, row["good"], row["bad"])
            bank.append(row)
        text, info = engine.add_detail_notes(text, seed=seed, chance=base, bank=bank)
        details = info.get("added") or []
        said.append("details: " + (", ".join(f"shot {d['shot']} {d['phrase']}" for d in details) if details else info.get("why") or "none"))
    return text, "; ".join(said), {"views": views, "arms": arms, "details": details,
                                   "text": text if values.get("detail_notes") else None}


class FunPackShotCamera(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackShotCamera",
            display_name="FunPack Shot Camera",
            category="FunPack/Conditioning",
            description="Cut times, views and camera moves for the [Shot N] blocks of an H3 prompt, chosen by rules (no language model). Each is off until switched on in Settings ▸ Engine.",
            inputs=[io.String.Input("text", multiline=True, default=""),
                    io.Custom("FUNPACK_SETTINGS").Input("settings", optional=True),
                    io.Int.Input("length", default=124, min=1, max=100000, tooltip="Frames of the video: with the frame rate, how long the shots have between them."),
                    io.Float.Input("frame_rate", default=25.0, min=1.0, max=240.0)],
            outputs=[io.String.Output(display_name="text"), io.String.Output(display_name="status")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Only the seed changes between takes, so ComfyUI would reuse this node's output and never note the
        # new run, and a rating of that take would find nothing to learn from. Cheap, so it just runs every time.
        return float("nan")

    @classmethod
    def execute(cls, text: str, length: int, frame_rate: float, settings=None) -> io.NodeOutput:
        values = _values(settings)
        if not any(values.get(k) for k in ("camera_moves", "shot_cuts", "shot_views", "cut_same_shot", "detail_notes")) or not engine.SHOT.search(text or ""):
            return io.NodeOutput(text, "unchanged")
        try:
            out, said, chose = rewrite(text, values, seconds=length / frame_rate if frame_rate else None)
        except Exception as exc:                               # noqa: BLE001 -- a rewrite is never worth the run
            log.failed("FunPack Shot Camera", exc)
            return io.NodeOutput(text, f"left as written ({exc})")
        pid = _prompt_id()
        if pid and (chose["views"] or chose["arms"] or chose.get("text") is not None):
            try:
                memory.record_run(pid, chose)
            except Exception as exc:                           # noqa: BLE001 -- only the learning is lost, not the run
                log.failed("FunPack Shot Camera", exc)
        log.info("FunPack Shot Camera", said)
        return io.NodeOutput(out, said)
