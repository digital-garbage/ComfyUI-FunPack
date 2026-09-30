"""Add a camera move to each `[Shot N]` block of an H3 prompt that has none, aimed at the
most specific thing in that shot. No language model: a small spaCy tagger (rules-only
fallback when it is not installed) picks the target, a few words pick the move.

`[Shot N]` here is MiniMax's own in-prompt shot, never a timeline scene. `<Subject N>` tags
are the subjects every shot is about, so they are never a target on their own; a part or
belonging of one ("<Subject 1>'s hand") is. A thing named in most shots (or in the text before
the first shot) is a constant, not a target. Sound text (music, score...) is left alone.
"""

import math
import re

SHOT = re.compile(r"\[\s*Shot\s+(\d+)\s*\]", re.I)
SUBJECT = re.compile(r"<\s*Subject\s*(\d+)\s*>", re.I)
# A `<Subject N>` tag confuses the tagger ("Subject1 bites" reads as a noun); a real-looking
# rare name parses like a person. Swapped in before tagging, swapped back after.
NAMES = ("Mariel", "Tobias", "Odette", "Rurik", "Sabine", "Teodor", "Ilsa", "Corwin", "Yvette")
STAND_IN = re.compile(r"\b(" + "|".join(NAMES) + r")\b")
AUDIO = re.compile(r"\b(music|soundtrack|score|melody|instrumental|beat|bassline|synth|"
                   r"song|tempo|audio|ambient sound|sound of)\b", re.I)
CAMERA = re.compile(r"\b(camera|zoom(?:s|ing)?|close-?up|pan(?:s|ning)?|dolly|tracking|"
                    r"rack focus|focus(?:es|ing)? (?:on|at)|tilt(?:s|ing)?|rotate|orbit|"
                    r"push[- ]in|pull[- ]back|wide shot|handheld|crane)\b", re.I)
GENERIC = {"camera", "video", "scene", "shot", "frame", "background", "foreground", "view",
           "image", "screen", "moment", "time", "way", "side", "front", "middle", "one",
           "other", "something", "everything", "thing", "music", "sound", "style", "lighting",
           "light", "atmosphere", "mood", "setting"}
PARTS = {"lip", "mouth", "tongue", "tooth", "teeth", "eye", "eyes", "face", "cheek", "hand",
         "finger", "hair", "neck", "shoulder", "waist", "hip", "leg", "foot", "feet", "arm",
         "skin", "expression", "smile", "gaze", "back", "chin", "brow"}
PLACES = {"room", "street", "kitchen", "bedroom", "bed", "table", "floor", "stage", "hall",
          "garden", "beach", "forest", "city", "car", "office", "bathroom", "field"}
MOVES_DETAIL = ("Close-up on {x}.", "Focus at {x}.", "Zoom in at {x}.")
MOVES_OTHER = ("Zoom in at {x}.", "Move around {x}.", "Rotate the camera to {x}.")
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}
POSSESSIVE_PRONOUNS = {"his", "her", "their", "its", "my", "your", "our"}

_nlp = None
_tried = False


def _spacy():
    """The small English tagger, or None (said once) when it is not installed."""
    global _nlp, _tried
    if not _tried:
        _tried = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:  # noqa: BLE001
            try:
                import funpack_log as _log
            except ImportError:
                from . import funpack_log as _log
            fix = ("spacy not installed, run `pip install spacy`" if isinstance(e, ImportError)
                   else "model missing, run `python -m spacy download en_core_web_sm`")
            _log.failed("FunPackCameraMoves", "spaCy tagger", e, f"simple word rules used; {fix}")
    return _nlp


def _hide(text):
    return SUBJECT.sub(lambda m: NAMES[(int(m.group(1)) - 1) % len(NAMES)], text)


def _show(text):
    return STAND_IN.sub(lambda m: f"<Subject {NAMES.index(m.group(1)) + 1}>", text)


def _is_subject(text):
    return bool(STAND_IN.fullmatch(re.sub(r"^(?:the\s+)", "", text.strip(), flags=re.I)))


def _split_sound(body):
    """-> (picture text, sound text): everything from the first sentence about sound on."""
    for m in re.finditer(r"[^.!?]+[.!?]?\s*", body):
        if AUDIO.search(m.group(0)):
            return body[:m.start()], body[m.start():]
    return body, ""


def _phrase(tok):
    """A chunk head with the words that name it ("Subject1's hand", "red lips")."""
    keep = {tok}
    for c in tok.children:
        if c.dep_ in ("poss", "compound", "amod") and c.i < tok.i:
            keep.add(c)
            if c.dep_ == "poss":
                keep.update(g for g in c.children if g.dep_ == "case")
    toks = sorted(keep, key=lambda t: t.i)
    text = " ".join(t.text for t in toks).strip().replace(" 's", "'s").replace(" ’s", "’s")
    return text, any(c.dep_ == "poss" for c in tok.children)


def candidates(picture):
    """[(lemma, phrase, owned, position, dep)] for the nouns of one shot's picture text."""
    hidden = _hide(picture)
    nlp = _spacy()
    out = []
    if nlp is not None:
        doc = nlp(hidden)
        for chunk in doc.noun_chunks:
            root = chunk.root
            prev = doc[root.i - 1] if root.i else None
            if prev is not None and STAND_IN.fullmatch(prev.text) and prev.dep_ != "poss":
                continue                      # "<Subject 1> laughs": the tagger misread a verb
            if root.pos_ in ("PRON", "NUM") or _is_subject(chunk.text):
                continue
            lemma = root.lemma_.lower()
            if root.dep_ == "pobj" and root.head.lower_ == "of":
                continue                      # "a glass of wine": the glass is the thing
            text, owned = _phrase(root)
            if _is_subject(text) or lemma in GENERIC or STAND_IN.fullmatch(root.text):
                continue
            out.append((lemma, text, owned, chunk.start, root.dep_))
        return out
    for m in re.finditer(rf"(?:({'|'.join(NAMES)})'s|\b(his|her|their|the|a|an))\s+((?:\w+\s+)?\w+)",
                         hidden, re.I):
        words = m.group(3).split()
        lemma = words[-1].lower().rstrip("s") if words[-1].lower() not in PARTS else words[-1].lower()
        if lemma in GENERIC:
            continue
        owner = m.group(1) or (m.group(2) if (m.group(2) or "").lower() in POSSESSIVE_PRONOUNS else "")
        text = (f"{owner}'s " if m.group(1) else f"{owner} " if owner else "") + " ".join(words)
        out.append((lemma, text.strip(), bool(owner), m.start(), "obj"))
    return out


def _score(c):
    lemma, _text, owned, pos, dep = c
    return ((2 if owned or lemma in PARTS else 0) + (1 if dep in ("dobj", "pobj") else 0)
            - pos * 1e-4)


def add_camera_moves(text):
    """-> (new prompt, [per-shot report dicts]). Unchanged when there is no `[Shot N]`."""
    marks = list(SHOT.finditer(text or ""))
    if not marks:
        return text, []
    header = text[:marks[0].start()]
    bodies = [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
              for i, m in enumerate(marks)]
    parts = [_split_sound(b) for b in bodies]
    cands = [candidates(p[0]) for p in parts]
    header_lemmas = {c[0] for c in candidates(header)}
    need = max(2, math.ceil(0.6 * len(bodies))) if len(bodies) > 1 else 10 ** 9
    seen = {}
    for lst in cands:
        for lemma in {c[0] for c in lst}:
            seen[lemma] = seen.get(lemma, 0) + 1
    constant = {l for l, n in seen.items() if n >= need} | header_lemmas
    report, out, last = [], [header], None
    for i, m in enumerate(marks):
        picture, sound = parts[i]
        entry = {"shot": int(m.group(1)), "move": None, "target": None, "why": ""}
        pool = [c for c in cands[i] if c[0] not in constant]
        if CAMERA.search(picture):
            entry["why"] = "already has a camera move"
        elif not pool:
            entry["why"] = "nothing specific to aim at"
        else:
            best = max(pool, key=_score)
            detail = best[2] or best[0] in PARTS
            moves = MOVES_DETAIL if detail else MOVES_OTHER
            move = next((mv for mv in moves if mv != last), moves[0])
            target = _show(best[1])
            if not re.match(r"(?:the|a|an|<Subject|his|her|their)\b", target, re.I) \
                    and "'s " not in target:
                target = "the " + target
            entry.update(move=move.format(x=target), target=target)
            last = move
            core = picture.rstrip()
            gap = picture[len(core):] or (" " if sound else "")
            picture = core + ("" if core.endswith((".", "!", "?")) else ".") + " " \
                + entry["move"] + gap
        out.append(m.group(0) + picture + sound)
        report.append(entry)
    return "".join(out), report
