"""Phrase probe: where does the model read a bracketed phrase, and where does it matter?

MEASUREMENT ONLY -- nothing about a generation changes. Switched on from Settings >
Refinement & Taste (off by default: it is research data, and the response map costs one
extra model forward per phrase per call).

The phrases are the ones Prompt Markup placed -- weighted `(word:1.5)` and timed
`[phrase@2-4]` -- which it publishes into the model's transformer_options. That is read
at run time, because this installs BEFORE the markup node runs in the pipeline.

v4's first result (blocks 15-20 read the text, 29-34 and block 43 commit it, blocks 0-6 are
text-blind; two different shortcuts gave the same map) came from a run with steering on.
Rerun it clean on a rental. Unvalidated in v5.
"""

import torch

from ..._core import dit_hooks, log, streams
from . import measure

ID = "phrase_probe"
TITLE = "Phrase probe"
MOUNT = "generation.sampling"
STAGE = "post"
CATEGORY = "system"
STATUS = "experimental"
REQUIRES = ["dit_block_hooks"]
# Innermost on the model, so every call it sees is a real forward: the extra forwards of
# late-branch guidance and the taste slider (flagged as probes) pass through it untouched
# instead of being measured as the picture.
AFTER = ["video_detail", "late_branch", "score_slider", "sharpen"]


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack Phrase probe", message)


def install(patcher, values, key):
    if not measure.enabled():
        measure.problem = None
        return None                              # off: no wrappers, no cost, nothing to disturb
    n = dit_hooks.block_count(patcher)
    if n == 0:
        measure.problem = "this model's blocks can't be read, so nothing is measured"
        if measure.enabled():
            _say("recording is on but " + measure.problem)
        return None
    measure.problem = None
    live = {"on": False, "state": None}

    def hook_for(block):
        def hook(args, extra):
            st = live["state"]
            if not live["on"] or st is None:
                return extra["original_block"](args)
            return measure.block_hook(st, block)(args, extra)
        return hook

    for b in range(n):
        dit_hooks.add_block_hook(patcher, key, b, hook_for(b))

    def attention(func, *a, **kw):
        st = live["state"]
        if not live["on"] or st is None:
            return func(*a, **kw)
        return measure.attention_override(st)(func, *a, **kw)

    dit_hooks.add_attention_override(patcher, key, attention)
    # Masked forwards allocate differently from the visible one: see dit_hooks.without_compiler.
    dit_hooks.without_compiler(patcher, key)

    from comfy.patcher_extension import WrappersMP

    def outer(executor, *args, **kwargs):
        live["on"] = measure.enabled()
        live["state"] = None
        try:
            return executor(*args, **kwargs)
        finally:
            st = live["state"]
            if live["on"] and st is not None:
                res = measure.result(st)
                measure.save(res)
                for p in measure.peaks(res, top=3):
                    rt = ", ".join(f"b{b} {v * 100:.1f}%" for b, v in p["read_top"])
                    gt = ", ".join(f"b{b} +{v:.3f}" for b, v in p["response_growth_top"])
                    log.once(f"{ID}:run:{p['phrase']}", log.INFO, "FunPack Phrase probe",
                             f"Active | phrase {p['phrase']} ({p['tokens']} tokens): read most at "
                             f"{rt or 'n/a'}; response grows most at {gt or 'n/a'}")
            elif live["on"]:
                measure.problem = ("recording is on but this run had no phrases to measure: put a "
                                   "(word:1.5) or [phrase@2-4] in the prompt")
                log.once(f"{ID}:none", log.INFO, "FunPack Phrase probe", "Inactive | " + measure.problem)
            else:
                measure.problem = None

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options") or {}
        if not live["on"] or dit_hooks.probing(to):
            return executor(x, t, *args, **kwargs)
        st = live["state"]
        if st is None:
            phrases = measure.phrases_of(to.get("funpack_markup_phrases"))
            if not phrases:
                return executor(x, t, *args, **kwargs)
            st = live["state"] = measure.State(phrases, to["funpack_markup_phrases"]["cond_len"])
            log.once(f"{ID}:cost:{len(phrases)}", log.INFO, "FunPack Phrase probe",
                     f"{len(phrases)} phrase(s): the reading map is free, the response map costs "
                     f"{len(phrases)} extra forward(s) per model call this run")
        st.mode, st.block = -1, 0
        try:
            out = executor(x, t, *args, **kwargs)
        finally:
            st.mode = None
        st.calls += 1
        probe_args, probe_kwargs = streams.with_options(args, kwargs, {**to, dit_hooks.PROBE: True})
        for i in range(len(st.phrases)):
            st.mode, st.block = i, 0
            try:
                executor(x, t, *probe_args, **probe_kwargs)
            except Exception as exc:                  # noqa: BLE001 -- the probe's own pass never costs the step
                _say(f"a masked pass failed ({type(exc).__name__}: {exc}); this phrase's response is not measured")
            finally:
                st.mode = None
        st.visible = {}
        return out

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    return "measures where each marked phrase is read and where it matters (off unless recording is switched on)"


def routes(table, base, web):
    @table.get(base + "/status")
    async def _status(_req):
        return web.json_response({"enabled": measure.enabled(), "problem": measure.problem,
                                  "latest": measure.latest(), "peaks": measure.peaks(measure.latest())})

    @table.post(base + "/enabled")
    async def _enabled(req):
        try:
            body = await req.json()
        except Exception:                             # noqa: BLE001
            body = {}
        on = measure.set_enabled((body or {}).get("enabled") is True)
        return web.json_response({"enabled": on, "problem": measure.problem, "latest": measure.latest(),
                                  "peaks": measure.peaks(measure.latest())})

    @table.post(base + "/clear")
    async def _clear(_req):
        measure.clear()
        return web.json_response({"enabled": measure.enabled(), "problem": measure.problem, "latest": None,
                                  "peaks": []})

    @table.get(base + "/export")
    async def _export(_req):
        res = measure.latest()
        if res is None:
            return web.json_response({"why": "nothing recorded yet"}, status=404)
        return web.json_response(res, headers={"Content-Disposition":
                                               'attachment; filename="phrase_probe.json"'})


def cache_key():
    # The node that installs this is cached by ComfyUI: the switch must be part of what re-runs it.
    return "on" if measure.enabled() else "off"


PROVIDES = {"modifier": install, "routes": routes, "cache_key": cache_key}
