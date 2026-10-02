"""Prompt enhancer: a language model rewrites the finished prompt before it is
encoded. A node-only module, with a small readout route for the Composer.

Sits after shortcut expansion and `$variables` (the app resolves both before the
text reaches the pipeline), so the model is handed exactly the text that would
otherwise have been encoded. Off by default. Every failure hands the ORIGINAL
prompt on and says so in the log: a rewrite that did not happen must never empty
the prompt.

One generation per scene -- v5 runs a scene per generation, so a multi-scene
project is never rewritten as one blob and re-split.

Whether a given text encoder writes sensible prose is an open question per model
(H3's conditioning checkpoint has no trained decoder head; Gemma3/Qwen3 do). v4's
result: confirmed working on Gemma3-12B. Unvalidated on a rental in v5.
"""

from .nodes import RUNS, FunPackEnhancePrompt

ID = "conditioning_prompt_enhancer"
TITLE = "Prompt enhancer"
STAGE = "conditioning"
CATEGORY = "conditioning"
STATUS = "experimental"

NODES = [FunPackEnhancePrompt]


def routes(table, base, web):
    @table.get(base + "/defaults")
    async def _defaults(_req):
        """What an empty field means, so the Composer can show it."""
        from . import enhance
        return web.json_response({"instructions": enhance.SYSTEM_PROMPT,
                                  "reference_intro": enhance.REFERENCE_INTRO})

    @table.get(base + "/runs")
    async def _runs(req):
        """The last runs' rewrites, newest last. `?prompt_id=` narrows to one run."""
        want = req.rel_url.query.get("prompt_id")
        runs = [r for r in RUNS if not want or r["prompt_id"] == want]
        return web.json_response({"runs": runs})


PROVIDES = {"routes": routes}
