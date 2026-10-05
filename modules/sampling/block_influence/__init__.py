"""Block influence: which blocks move the picture, and does it differ on clips you liked?

MEASUREMENT ONLY -- nothing about a generation changes. Each block of a residual
transformer adds `x_next = x + f(x)`. On every step this records how big f(x) is
on the PICTURE rows (raw, and relative to x), and whether each block's push goes
the same way as the block before it. A run's averages wait on the taste key for a
rating, like every other learner.

What it can and cannot say. A big push means the block moved the stream there --
not that the move reached the finished clip; later blocks can undo an early one.
The causal version is ablation, offline. What this gives, free, is the shape: if
every block moves the stream equally (flat) rating-driven per-block weighting has
nothing to aim at. A flat result is a real answer here, not a failure.

Prior art: per-block steering from ratings was built in v4 and removed (block
activity differed in the 4th decimal). That version averaged over steps before
looking; this keeps every step separate until the end.

Switched on from Settings > Refinement & Taste (off by default: it is research
data). The hooks sit on the model whenever a taste key is set and read the
switch at the start of each run, because ComfyUI caches the node that installs
them and a toggle must reach the next generation.
"""

import torch

from ..._core import dit_hooks, log, registry
from . import measure

ID = "block_influence"
TITLE = "Block influence"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "system"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]
USES = ["taste_store"]
# Measured before anything that changes a block's output, so the number is the
# block's OWN push and not another mechanism's injection.
BEFORE = ["reins", "block_repeat", "shadow_negative", "q_steer", "attention_temperature"]

KIND = measure.KIND


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Block influence", message)


def install(patcher, values, key):
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        measure.problem = "no Taste key is set, so nothing is measured or kept: set a Taste key"
        if measure.enabled():
            _say("recording is on but " + measure.problem)
        return None                              # no key: nothing to pair a profile with
    n = dit_hooks.block_count(patcher)
    if n == 0:
        measure.problem = "this model's blocks can't be read, so nothing is measured"
        _say("off: " + measure.problem)
        return None
    measure.problem = None
    live = {"on": False, "tally": measure.Tally(n)}

    def hook_for(block):
        def hook(args, extra):
            to = args.get("transformer_options")
            if not live["on"] or dit_hooks.probing(to):
                return extra["original_block"](args)
            return live["tally"].measure(block, args, extra["original_block"])
        return hook

    for b in range(n):
        dit_hooks.add_block_hook(patcher, key, b, hook_for(b))

    from comfy.patcher_extension import WrappersMP

    def outer(executor, *args, **kwargs):
        live["on"] = measure.enabled()
        live["tally"] = measure.Tally(n)
        out = executor(*args, **kwargs)
        if live["on"]:
            rows = live["tally"].rows()
            measure.problem = None
            if rows is None:
                measure.problem = (f"the last recording run (key {taste.key!r}) measured nothing: no "
                                   "picture rows found in any block")
                log.warning("FunPack Block influence",
                            "recording is on but nothing was measured this run (no picture rows "
                            "found in any block); this run teaches nothing")
            else:
                taste.capture(KIND, rows)
                log.once(f"{ID}:run", log.INFO, "FunPack Block influence",
                         f"Active | measured {live['tally'].blocks_seen()} of {n} blocks on key "
                         f"{taste.key!r}; rate the clip to keep it")
        else:
            measure.problem = None
            log.once(f"{ID}:off", log.INFO, "FunPack Block influence",
                     "Inactive | recording is off (Settings > Refinement & Taste turns it on)")
        return out

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)
    return "measures per-block movement when recording is on (read fresh every run)"


def routes(table, base, web):
    def key_of(req, body=None):
        raw = (body or {}).get("key") if body is not None else req.rel_url.query.get("key")
        return str(raw or "default").strip() or "default"

    @table.get(base + "/status")
    async def _status(req):
        try:
            return web.json_response(measure.state(key_of(req)))
        except ValueError as exc:
            return web.json_response({"why": str(exc)}, status=400)

    @table.post(base + "/enabled")
    async def _enabled(req):
        body = await req.json()
        try:
            state = measure.state(key_of(req, body))     # refuse BEFORE flipping the switch
        except ValueError as exc:
            return web.json_response({"why": str(exc)}, status=400)
        measure.set_enabled(body.get("enabled") is True)
        return web.json_response({**state, "enabled": measure.enabled()})

    @table.post(base + "/clear")
    async def _clear(req):
        body = await req.json()
        try:
            key = key_of(req, body)                      # exactly the key named: never a fallback
            measure.clear(key)
            return web.json_response(measure.state(key, fallback=False))
        except ValueError as exc:
            return web.json_response({"why": str(exc)}, status=400)

    @table.get(base + "/export")
    async def _export(req):
        try:
            path = measure.path_of(key_of(req))
        except ValueError as exc:
            return web.json_response({"why": str(exc)}, status=400)
        if not path.exists():
            return web.json_response({"why": f"nothing recorded yet for {key_of(req)!r}"}, status=404)
        return web.Response(body=path.read_bytes(), content_type="application/octet-stream",
                            headers={"Content-Disposition":
                                     f'attachment; filename="{key_of(req)}.block_influence.pt"'})


PROVIDES = {"modifier": install, "routes": routes}
