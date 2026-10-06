"""The Enhance Prompt node: an LLM rewrites the finished prompt before it is encoded."""

import collections
import json
import re
import time

from comfy_api.latest import io

from ..._core import log
from . import enhance as en

#: The last runs' rewrites, newest last: what the Composer's readout shows. In
#: memory on purpose -- it is the last run's answer, not state to keep.
RUNS = collections.deque(maxlen=20)


_MARKUP = re.compile(r"\([^()]*:\s*[\d.]+\)|\[[^\]]*[@|][^\]]*\]")


def _prompt_id():
    try:
        from server import PromptServer
        return str(PromptServer.instance.last_prompt_id or "")
    except Exception:                                               # noqa: BLE001
        return ""


class FunPackEnhancePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackEnhancePrompt",
            display_name="FunPack Enhance Prompt",
            category="FunPack/Conditioning",
            description="Rewrite the finished prompt with a language model before it is encoded. "
                        "Off, or on any failure, the prompt passes through unchanged.",
            inputs=[
                io.Clip.Input("clip", tooltip="A text encoder that can generate text (Gemma, Qwen...)."),
                io.String.Input("text", multiline=True, default=""),
                io.Boolean.Input("enabled", default=False,
                                 tooltip="Off: the prompt passes through and the model is not run."),
                io.String.Input("instructions", multiline=True, default="", optional=True,
                                tooltip="What the model is told to do. Empty uses the built-in instructions."),
                io.Int.Input("max_length", default=400, min=32, max=4096, step=8, optional=True,
                             tooltip="The longest rewrite, in tokens."),
                io.Boolean.Input("greedy", default=False, optional=True,
                                 tooltip="Always the likeliest word. Hides the sampling dials below."),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05, optional=True),
                io.Float.Input("top_p", default=0.92, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("top_k", default=50, min=1, max=1000, optional=True,
                             tooltip="top_p and min_p only apply while top_k is above 0."),
                io.Float.Input("min_p", default=0.05, min=0.0, max=1.0, step=0.01, optional=True),
                io.Float.Input("repetition_penalty", default=1.3, min=1.0, max=2.0, step=0.05, optional=True),
                io.Float.Input("presence_penalty", default=0.0, min=0.0, max=2.0, step=0.05, optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, optional=True,
                             tooltip="0 = a fresh rewrite every run. Any other number repeats the same rewrite."),
                io.Boolean.Input("thinking", default=False, optional=True,
                                 tooltip="Let a thinking model reason first. Its reasoning eats the token budget."),
                io.Image.Input("image", optional=True, tooltip="Shown to the model alongside the prompt."),
                io.Boolean.Input("use_image", default=True, optional=True,
                                 tooltip="Send the picture (when one is connected) to the model."),
                io.String.Input("shortcuts", multiline=True, default="", optional=True,
                                tooltip="Shortcut names to hand the model as reference, one per line."),
                io.String.Input("lorebooks", multiline=True, default="", optional=True,
                                tooltip="Paths to lorebook or shortcuts JSON files, one per line."),
                io.String.Input("reference_intro", multiline=True, default="", optional=True,
                                tooltip="The line above the reference entries. Empty uses the built-in one."),
                io.String.Input("chat", multiline=True, default="", optional=True,
                                tooltip="Your comments on earlier rewrites (Composer > Chat)."),
            ],
            outputs=[io.String.Output(display_name="enhanced"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def fingerprint_inputs(cls, seed=0, enabled=False, **_):
        # Seed 0 means "a new rewrite every run", and ComfyUI would otherwise hand
        # back the cached one for identical inputs.
        return time.time() if enabled and not seed else ""

    @classmethod
    def execute(cls, clip, text, enabled=False, instructions="", max_length=400, greedy=False,
                temperature=0.7, top_p=0.92, top_k=50, min_p=0.05, repetition_penalty=1.3,
                presence_penalty=0.0, seed=0, thinking=False, image=None, use_image=True, shortcuts="",
                lorebooks="", reference_intro="", chat="") -> io.NodeOutput:
        if not enabled:
            # still the final prompt, and the Composer's Enhance tab shows what was encoded either way
            RUNS.append({"prompt_id": _prompt_id(), "status": "off: the prompt ran as typed", "before": text, "after": text,
                         "thinking": "", "sent": "", "ok": True})
            return io.NodeOutput(text, "off")
        if _MARKUP.search(text or ""):
            log.once("prompt_enhancer:markup", log.ALERT, "FunPack Prompt enhancer",
                     "the prompt has (word:1.5) / [a@2-4] / [a|b] markup, which the model will "
                     "read as plain text: with the enhancer on, markup does not survive a rewrite")
        lines = lambda s: [x for x in str(s or "").splitlines() if x.strip()]    # noqa: E731
        groups = en.sources(lines(shortcuts), lines(lorebooks))
        try:
            rounds = json.loads(chat) if str(chat or "").strip() else []
        except ValueError:
            rounds = []
            log.warning("FunPack Prompt enhancer", "the Chat comments are not readable; ignored")
        if not isinstance(rounds, list):
            rounds = []
        out, status, info = en.enhance(
            clip, text, system=instructions,
            reference_text=en.reference(text, groups, reference_intro), chat=en.chat_block(rounds),
            seed=seed, image=image if use_image else None, thinking=thinking, max_length=max_length,
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
            repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
            do_sample=not greedy)
        if not info["ok"]:
            # The prompt went through UNCHANGED and the person must know why.
            log.warning("FunPack Prompt enhancer", f"{status}; the original prompt is used")
        RUNS.append({"prompt_id": _prompt_id(), "status": status, **info})
        return io.NodeOutput(out, status)
