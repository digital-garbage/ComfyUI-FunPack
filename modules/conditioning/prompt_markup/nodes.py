"""The two markup nodes. See the module docstring for the shape."""

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ..._core import dit_hooks, log, patching, registry as registry_mod
from . import markup as mk

MARKUP = io.Custom("FUNPACK_MARKUP")
KEY = "funpack_markup.prompt"


def _ask(capability, *args):
    """The first model module with an answer, or None."""
    for _spec, provider in registry_mod.current().providers(capability):
        got = provider(*args)
        if got is not None:
            return got
    return None


def _say(message):
    log.once(f"prompt_markup:{message}", log.ALERT, "FunPack Prompt Markup", message)


class FunPackPromptMarkup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackPromptMarkup",
            display_name="FunPack Prompt Markup",
            category="FunPack/Conditioning",
            description="Take (word:1.5), [phrase@2-4] and [a|b] out of a prompt, so the "
                        "encoder sees clean text and the markup can be applied properly.",
            inputs=[io.String.Input("text", multiline=True, default="")],
            outputs=[io.String.Output(display_name="clean"),
                     MARKUP.Output(display_name="markup"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, text: str) -> io.NodeOutput:
        parsed = mk.parse(text or "")
        n = {k: len(parsed[k]) for k in ("weighted", "timed", "blended")}
        status = (", ".join(f"{v} {k}" for k, v in n.items() if v) or "no markup")
        return io.NodeOutput(parsed["clean"], parsed, status)


def _blend(cond, rows, clip, tokenizer, clean, offsets, blended):
    """(1-s)*A + s*B over the prompt rows, each span measured against the
    ORIGINAL A so several blends add instead of compounding."""
    start, end = rows
    a_rows = cond[:, start:end, :]
    delta, done = torch.zeros_like(a_rows), 0
    for c0, c1, alt, strength in blended:
        if not mk.token_spans(offsets, [(c0, c1)]):
            continue
        alt_text = clean[:c0] + alt + clean[c1:]
        encoded = clip.encode_from_tokens_scheduled(clip.tokenize(alt_text))
        b_cond, b_meta = encoded[0][0], encoded[0][1]
        n_b = len(tokenizer(alt_text, add_special_tokens=False,
                            return_offsets_mapping=True)["offset_mapping"])
        b_rows = _ask("prompt_rows", b_meta, int(b_cond.shape[1]), n_b)
        if b_rows is None:
            continue
        b = b_cond[:, b_rows[0]:b_rows[1], :].to(a_rows)
        if b.shape[1] != a_rows.shape[1]:
            # Linear along the token axis: most rows already correspond (only
            # the swapped phrase differs); a smoothing, not a word alignment.
            b = F.interpolate(b.transpose(1, 2), size=a_rows.shape[1], mode="linear",
                              align_corners=True).transpose(1, 2)
        delta += float(strength) * (b - a_rows)
        done += 1
    if not done:
        return cond, 0
    out = cond.clone()
    out[:, start:end, :] += delta
    return out, done


class FunPackApplyPromptMarkup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FunPackApplyPromptMarkup",
            display_name="FunPack Apply Prompt Markup",
            category="FunPack/Conditioning",
            description="Apply the markup FunPack Prompt Markup took out: weights and "
                        "timed phrases on the model, blends on the conditioning.",
            inputs=[io.Model.Input("model"), io.Clip.Input("clip"),
                    io.Conditioning.Input("positive"), io.Latent.Input("latent"),
                    MARKUP.Input("markup")],
            outputs=[io.Model.Output(display_name="model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.String.Output(display_name="status")],
        )

    @classmethod
    def execute(cls, model, clip, positive, latent, markup) -> io.NodeOutput:
        patched = model.clone()
        patching.strip(patched, "funpack_markup.")
        weighted, timed, blended = (markup or {}).get("weighted"), (markup or {}).get("timed"), \
            (markup or {}).get("blended")
        if not (weighted or timed or blended):
            return io.NodeOutput(patched, positive, "no markup")

        def refuse(why):
            _say(f"markup NOT applied: {why}. The prompt still went in, without it.")
            return io.NodeOutput(patched, positive, f"not applied: {why}")

        tokenizer = _ask("text_tokenizer", clip)
        if tokenizer is None:
            return refuse("this text encoder is not one a model module can place words in")
        clean = markup["clean"]
        offsets = list(tokenizer(clean, add_special_tokens=False,
                                 return_offsets_mapping=True)["offset_mapping"])
        cond, meta = positive[0][0], positive[0][1]
        cond_len = int(cond.shape[1])
        rows = _ask("prompt_rows", meta, cond_len, len(offsets))
        if rows is None:
            return refuse("the encoded prompt does not match this text -- was the clean "
                          "text from Prompt Markup what got encoded?")
        base = rows[0]
        said = []

        if blended:
            new_cond, n = _blend(cond, rows, clip, tokenizer, clean, offsets, blended)
            if n:
                positive = [[new_cond, meta], *positive[1:]]
                said.append(f"{n} blend(s)")

        w_spans = mk.token_spans(offsets, weighted or [])
        t_spans = []
        for tok0, tok1, w, t0, t1 in mk.token_spans(offsets, timed or []):
            window = _ask("video_time_rows", latent["samples"], t0, t1)
            if window is None:
                _say(f"a timed phrase's window {t0:g}-{t1:g}s misses this clip; ignored")
                continue
            t_spans.append((tok0, tok1, w, window))
        if w_spans or t_spans:
            cls._install(patched, w_spans, t_spans, base, cond_len)
            if w_spans:
                said.append(f"{len(w_spans)} weight(s)")
            if t_spans:
                said.append(f"{len(t_spans)} timed phrase(s)")
        return io.NodeOutput(patched, positive, ", ".join(said) or "nothing placeable")

    @staticmethod
    def _install(patcher, w_spans, t_spans, base, cond_len):
        cache = {}

        def build(seq_len, device, dtype):
            timed, video_start = [], seq_len
            for tok0, tok1, w, (n_video, r0, r1) in t_spans:
                video_start = seq_len - n_video
                if video_start < cond_len:
                    _say("this sequence is too short to hold the video the timed "
                         "phrases were placed on; they are off for this call")
                    return mk.plan(w_spans, [], base, seq_len, seq_len, device, dtype)
                timed.append((tok0, tok1, w, video_start + r0, video_start + r1))
            return mk.plan(w_spans, timed, base, seq_len, video_start, device, dtype)

        def override(func, q, k, v, heads, *args, mask=None, **kwargs):
            def run(m, qq=q):
                return func(qq, k, v, heads, *args, mask=m, **kwargs)

            # Only the packed self-attention: the prompt plus the streams after it.
            # A sequence exactly as long as the prompt is the text-only token
            # refiner, which must never see a video window's mask.
            if k.ndim != 4 or q.shape[2] != k.shape[2] or k.shape[2] <= cond_len:
                return run(mask)
            try:
                key = (int(k.shape[2]), k.device, k.dtype)
                if key not in cache:
                    cache[key] = build(*key)
                plan = cache[key]
                if plan is None:
                    return run(mask)
                if not isinstance(plan, list):
                    return run(plan if mask is None else mask + plan)
                outs = [run(b if mask is None else mask + b, q[:, :, a:z].contiguous())
                        for a, z, b in plan]
                return torch.cat(outs, dim=2 if outs[0].ndim == 4 else 1)
            except Exception as exc:             # noqa: BLE001 -- markup never costs the step
                _say(f"markup failed inside attention and was dropped ({exc})")
                return run(mask)

        dit_hooks.add_attention_override(patcher, KEY, override)
