"""The prompt enhancer's logic: what the model is asked, and how its answer is
cleaned. Pure text in, text out -- the node (nodes.py) only wires a CLIP to it.

Every failure path returns the ORIGINAL prompt. An enhancer that empties the
prompt would render a blank video and blame the model.
"""

from __future__ import annotations

import json
import re
import time

from ..._core import log, shortcuts as shortcuts_mod

SYSTEM_PROMPT = """You expand short video prompts into detailed ones for a text-to-video model.

Rules:
- Keep every element the user asked for. Never drop, replace or contradict one.
- Add concrete visual detail where the prompt is vague: lighting, materials, textures, clothing, setting, time of day.
- Describe motion in the present progressive ("is walking", "is turning").
- Add a matching soundscape: ambient sound, and any sound the described actions would make.
- If the user quoted speech, reproduce it exactly. Never invent speech that was not asked for.
- Do not invent camera moves or scene cuts unless the user asked for them.
- Do not add characters the user did not mention.

Output: one paragraph of plain prose. No preamble, no headings, no markdown, no quotes around the whole answer. Output only the prompt itself."""

CHAT_SYSTEM = """REVISION: this time the user is giving feedback on your earlier rewrite.
The message has labelled parts: ORIGINAL PROMPT (what the user wrote), YOUR LATEST REWRITE (your last answer, if shown) and USER FEEDBACK (their comments, oldest first).
- Start from YOUR LATEST REWRITE (or the ORIGINAL PROMPT if none is shown) and change it as the feedback asks. Apply every comment; when two disagree, the later one wins.
- Keep everything the feedback does not ask to change.
- The feedback is instructions to you, not prompt text: never quote it, mention it or append it.
Output only the revised prompt, nothing else."""

REFERENCE_INTRO = ("Reference entries. Use an entry only where the prompt "
                   "calls for it, in your own words; ignore the rest.")

CHAT_LABELS = ("ORIGINAL PROMPT:", "YOUR LATEST REWRITE:", "USER FEEDBACK")
CHARS_PER_TOKEN = 4          # turns the user's TOKEN limit into a character bound
REPEAT_LIMIT = 3             # how often a sentence may repeat before it is a stuck model

# Filler that says nothing about which shortcut a prompt means.
_STOPWORDS = frozenset("""
the and with from into onto over under for but not are was were been being has have had
its his her hers their theirs them they she him you your yours our ours this that these
those there here then than very just only also some any all each every both few more most
other such own same too can will would could should may might must does did doing done
while when where which who whom whose what why how about above after again against
before below between during out off once further through until upon via per
""".split())


def vocab(text) -> set[str]:
    """Meaningful words (letters only, 3+ long, not filler), lowercased."""
    return {w for w in re.findall(r"[^\W\d_]{3,}", str(text or "").lower()) if w not in _STOPWORDS}


# --- reference ---------------------------------------------------------------

def sources(shortcut_names=None, paths=None, say=None) -> list[list[dict]]:
    """Reference groups: the picked shortcuts as one group, then one per file (a
    SillyTavern lorebook with `entries`, or a FunPack shortcuts file). Entry:
    {block, words|keys, vocab, whole, constant}. A missing shortcut or an unreadable
    file is skipped and said so -- through `say` when given."""
    say = say or (lambda m: log.warning("FunPack Prompt enhancer", m))

    def sc_entry(sc):
        reps = [r for r in sc.replacements if r.strip()]
        lines = [f"[Shortcut] {sc.name}"]
        lines += reps if len(reps) == 1 else [f"Variant {i + 1}: {r}" for i, r in enumerate(reps)]
        # triggers are expanded away by now, so the content counts as a mention too
        return {"block": "\n".join(lines), "words": [sc.name, *sc.triggers, *reps],
                "vocab": vocab(" ".join([sc.name, *reps]))}

    groups = []
    library = {s.name.lower(): s for s in shortcuts_mod.listing()}
    picked = []
    for name in (n.strip() for n in (shortcut_names or [])):
        if not name:
            continue
        if name.lower() in library:
            picked.append(sc_entry(library[name.lower()]))
        else:
            say(f"reference shortcut {name!r} is not in the library; skipped")
    if picked:
        groups.append(picked)
    for path in (str(p).strip() for p in (paths or [])):
        if not path:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("not a lorebook or shortcuts file")
        except (OSError, ValueError) as exc:
            say(f"reference file {path} skipped: {exc}")
            continue
        if "shortcuts" in data:
            raw = data["shortcuts"]
            raw = list(raw.values()) if isinstance(raw, dict) else raw
            items = [shortcuts_mod.Shortcut.from_dict(r) for r in raw if isinstance(r, dict)]
            groups.append([sc_entry(s) for s in items if s.triggers and s.enabled])
            continue
        entries = data.get("entries", [])
        entries = list(entries.values()) if isinstance(entries, dict) else entries
        entries = sorted((e for e in entries if isinstance(e, dict) and str(e.get("content") or "").strip()),
                         key=lambda e: e.get("insertion_order", e.get("order", 0)) or 0)
        groups.append([{"block": f"[Lore] {e.get('comment') or e.get('name') or 'entry'}\n"
                                 f"{str(e['content']).strip()}",
                        "keys": e.get("keys", e.get("key", [])),
                        "whole": e.get("matchWholeWords") is not False,   # ST's default is whole words
                        "constant": bool(e.get("constant"))} for e in entries])
    return [g for g in groups if g]


def reference(text, groups, intro="") -> str:
    """Appended after the prompt the model rewrites. Per group: the entries the prompt
    mentions, or the whole group when it mentions none (the model may need what the
    prompt does not name). Shortcuts are mentioned by name, trigger, content or any
    meaningful word of those; lore by its keywords, a constant entry joining any match."""
    low = str(text or "").lower()

    def said(word, whole=True, lore=False):
        word = str(word).strip()
        if not word:
            return False
        if lore and len(word) > 2 and word.startswith("/") and word.endswith("/"):
            try:
                return re.search(word[1:-1], text or "", re.IGNORECASE) is not None
            except re.error:
                return False
        if not whole:
            return word.lower() in low
        return re.search(r"(?<!\w)" + re.escape(word.lower()) + r"(?!\w)", low) is not None

    typed = vocab(low)

    def mentioned(e):
        if e.get("vocab") and typed & e["vocab"]:
            return True
        if "keys" in e:
            keys = e["keys"].split(",") if isinstance(e["keys"], str) else e["keys"]
            return any(said(k, e.get("whole", True), lore=True) for k in keys or [])
        return any(said(w) for w in e["words"])

    blocks = []
    for group in groups or []:
        hit = [e for e in group if mentioned(e)]
        if hit:
            hit = [e for e in group if e.get("constant") or e in hit]
        blocks += [e["block"] for e in (hit or group)]
    if not blocks:
        return ""
    return "\n\n" + (str(intro or "").strip() or REFERENCE_INTRO) + "\n\n" + "\n\n".join(blocks)


# --- chat --------------------------------------------------------------------

def chat_block(chat, scene=None) -> str:
    """The user's comments on earlier rewrites as the labelled part of the message:
    the LATEST rewrite of this prompt and every comment since Reset, oldest first.
    `chat` is a list of rounds {"rewrites": {"whole"|"<scene index>": text}, "comment"}.
    Only the latest rewrite is shown (a model shown several picks one to repeat);
    every comment stays, because the user expects all of them to still hold."""
    rounds = [r for r in (chat or []) if isinstance(r, dict) and str(r.get("comment") or "").strip()]
    if not rounds:
        return ""
    key = "whole" if scene is None else str(scene)
    latest = ""
    for r in rounds:
        rewrites = r.get("rewrites") if isinstance(r.get("rewrites"), dict) else {}
        prev = rewrites.get(key)
        if prev is None and len(rewrites) == 1:
            prev = next(iter(rewrites.values()))
        if str(prev or "").strip():
            latest = str(prev).strip()
    parts = []
    if latest:
        parts.append("YOUR LATEST REWRITE:\n" + latest)
    parts.append("USER FEEDBACK (oldest first):\n" + "\n".join("- " + str(r["comment"]).strip() for r in rounds))
    return "\n\n" + "\n\n".join(parts)


def cut_chat_echo(text) -> str:
    """A model that treats the conversation as more prompt text repeats its labels:
    a leading label is removed, and everything from the first later label is cut."""
    text = str(text or "").strip()
    labels = "|".join(re.escape(l.rstrip(":")) for l in CHAT_LABELS)
    text = re.sub(rf"(?i)^\s*(?:{labels})[^:\n]*:\s*", "", text)
    hit = re.search(rf"(?i)(?:{labels})", text)
    return (text[:hit.start()] if hit else text).strip()


# --- cleaning ----------------------------------------------------------------

def extract_thinking(raw) -> str:
    """`<think>...</think>` (also an unterminated one), or everything before a lone
    `</think>` when the template opened the block itself."""
    text = str(raw or "")
    parts = re.findall(r"(?is)<think>(.*?)(?:</think>|$)", text)
    if not parts and "</think>" in text.lower():
        parts = [re.split(r"(?i)</think>", text, maxsplit=1)[0]]
    return "\n\n".join(p.strip() for p in parts if p.strip())


def clean(raw) -> str:
    """Strip what a chat model wraps around a prompt it was asked to output bare:
    thinking traces, code fences, "Here is..." preambles, whole-answer quotes.
    Quotes INSIDE the prompt survive -- speech is quoted on purpose."""
    text = str(raw or "")
    text = re.sub(r"(?is).*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*", "", text)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "")
    text = re.sub(r"(?i)^\s*(here(?:'s| is)[^:\n]*:|output\s*:|prompt\s*:)\s*", "", text.strip())
    text = text.strip().strip("`").strip()
    if len(text) > 1 and text[0] in "\"'“" and text[-1] in "\"'”":
        text = text[1:-1].strip()
    return text.strip()


def trim_runaway(text, max_length=None):
    """Cut a prompt that never ended -- by REPETITION first, length second. A model
    with no trained decoder head may never emit a stop token. Length alone cannot tell
    "long and detailed" from "stuck", so what is detected is a sentence repeating; the
    length bound comes from the user's own token limit. Cuts land on a sentence."""
    text = str(text or "").strip()
    if not text:
        return text, False
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen: dict[str, int] = {}
    for i, s in enumerate(sentences):
        key = " ".join(s.lower().split())
        if len(key) < 12:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= REPEAT_LIMIT:
            kept = " ".join(sentences[:i]).strip()
            if kept:
                return kept, True
    if max_length is None:
        return text, False
    cap = max(64, int(max_length) * CHARS_PER_TOKEN)
    if len(text) <= cap:
        return text, False
    head = text[:cap]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut > cap // 4:
        return head[:cut + 1].strip(), True
    return head.rsplit(" ", 1)[0].strip(), True


# --- generation --------------------------------------------------------------

def generation_device(clip) -> str:
    """Where the text generation actually ran -- reported, not assumed. A 32B encoder
    sharing a card with the DiT can end up on the CPU: minutes per call, invisible
    from the text alone."""
    for path in (("patcher", "load_device"), ("_device",), ("device",)):
        probe = clip
        for name in path:
            probe = getattr(probe, name, None)
            if probe is None:
                break
        if probe is not None:
            return str(probe)
    return "an unreported device"


def generate(clip, system, user, *, seed=None, image=None, thinking=False, max_length=400,
             temperature=0.7, top_p=0.92, top_k=50, min_p=0.05, repetition_penalty=1.3,
             presence_penalty=0.0, do_sample=True):
    """(raw text, status). Through a CLIP that exposes ComfyUI's generate/decode pair.
    No ComfyUI tokenizer takes a system prompt (they swallow unknown kwargs), so the
    instructions go first in the user turn, inside the model's own chat template --
    which is also what places an image marker."""
    if clip is None or not hasattr(clip, "generate") or not hasattr(clip, "decode"):
        return "", ("unavailable: the connected CLIP does not expose text generation; wire a "
                    "generation-capable text encoder")
    max_length = max(32, int(max_length or 400))
    started = time.time()
    try:
        merged = f"{system}\n\n{user}" if str(system).strip() else str(user)
        try:
            tokens = clip.tokenize(merged, image=image, skip_template=False, min_length=1,
                                   thinking=bool(thinking))      # min_length=1: Gemma pads to 1024 otherwise
        except TypeError:
            tokens = clip.tokenize(merged, image=image)
        kwargs = dict(do_sample=bool(do_sample), max_length=max_length, temperature=float(temperature),
                      top_k=int(top_k), top_p=float(top_p), min_p=float(min_p),
                      repetition_penalty=float(repetition_penalty), no_repeat_ngram_size=5,
                      presence_penalty=float(presence_penalty), seed=seed if seed else None)
        try:
            ids = clip.generate(tokens, **kwargs)
        except TypeError:
            kwargs.pop("no_repeat_ngram_size")
            ids = clip.generate(tokens, **kwargs)
        text = str(clip.decode(ids, skip_special_tokens=True) or "").strip()
        return text, (f"generated {len(text)} chars in {time.time() - started:.1f}s on "
                      f"{generation_device(clip)} (cap {max_length} tokens)")
    except Exception as exc:                                    # noqa: BLE001
        return "", f"generation failed: {exc}"


def enhance(clip, text, *, system="", reference_text="", chat="", **sampling):
    """-> (text, status, info). On any failure the ORIGINAL text comes back."""
    original = str(text or "").strip()
    info = {"before": original, "after": original, "thinking": "", "sent": "", "ok": False}
    if not original:
        return text, "skipped: the prompt is empty", info
    system = str(system or "").strip() or SYSTEM_PROMPT
    user = original + str(reference_text or "")
    if chat:
        # Labelled parts plus a revision rule in the SYSTEM turn: with the conversation merely
        # appended, a model rewrote it as more prompt text and echoed it back.
        system += "\n\n" + CHAT_SYSTEM
        user = "ORIGINAL PROMPT:\n" + user + str(chat)
        info["sent"] = f"SYSTEM:\n{system}\n\nMESSAGE:\n{user}"
    raw, status = generate(clip, system, user, **sampling)
    info["thinking"] = extract_thinking(raw)
    cleaned = clean(raw)
    if chat:
        cleaned = cut_chat_echo(cleaned)
    out, overran = trim_runaway(cleaned, max_length=sampling.get("max_length"))
    if overran:
        status += ("; the model repeated itself instead of stopping and was cut at a sentence "
                   "(a model with a trained chat head does not do this)")
    if not out:
        return original, (status if "failed" in status or "unavailable" in status
                          else "the model returned nothing; prompt unchanged"), info
    info.update(after=out, ok=True)
    return out, f"{status}; prompt {len(original)} -> {len(out)} chars", info
