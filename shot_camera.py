"""Add a camera move to each `[Shot N]` block of an H3 prompt that has none, aimed at the
most specific thing in that shot. No language model: a small spaCy tagger (rules-only
fallback when it is not installed) picks the target, a few words pick the move.

`[Shot N]` here is MiniMax's own in-prompt shot, never a timeline scene. `<Subject N>` tags
are the subjects every shot is about, so they are never a target on their own; a part or
belonging of one ("<Subject 1>'s hand") is. A thing named in most shots (or in the text before
the first shot) is a constant, not a target. Sound text (music, score...) is left alone.
"""

import math
import random
import re

SHOT = re.compile(r"\[\s*Shot\s+(\d+)\s*\]", re.I)
SUBJECT = re.compile(r"<\s*Subject\s*(\d+)\s*>", re.I)
# A `<Subject N>` tag confuses the tagger ("Subject1 bites" reads as a noun); a real-looking
# rare name parses like a person. Swapped in before tagging, swapped back after.
NAMES = ("Mariel", "Tobias", "Odette", "Rurik", "Sabine", "Teodor", "Ilsa", "Corwin", "Yvette")
STAND_IN = re.compile(r"\b(" + "|".join(NAMES) + r")\b")
AUDIO = re.compile(r"\b(music|soundtrack|score|melody|instrumental|beat|bassline|synth|"
                   r"song|tempo|audio|ambient sound|sound of)\b", re.I)
# H3's own vocabulary (VIDEO_PROMPT_WRITING_GUIDE_base_en.md): a shot opening "the camera cuts
# to" is a cut, not a camera move, and must not make the shot look like it already has one.
CUT = re.compile(r"\b(?:the )?(?:camera|shot) (?:cuts|transitions|changes|switches) to\b", re.I)
# Whether a shot ALREADY has a camera move. Words like tilt, roll, pull, push, arc, focus are
# ordinary body movement ("tilts <Subject 1>'s head up and down", "pulls out a knife"), and
# "the camera" is often only where someone is looking, so those count only when the CAMERA
# does them, or when a sentence opens with the bare camera verb ("Pan right to reveal...").
_CAMERA_VERBS = (r"(?:moves?|pushes|pulls|pans|tilts|zooms|tracks|follows|drifts|circles|orbits|"
                 r"rises|lowers|glides|sweeps|racks|arcs|dollies|trucks|rolls|rotates|holds|"
                 r"shakes|stays|remains|sways|slides|cranes|swings|reframes|widens|tightens|"
                 r"starts|begins|continues|lingers|settles|zoom|pan|tilt|push|pull)")
CAMERA = re.compile(
    r"\b(?:zoom(?:s|ed|ing)?\s+(?:in|out)|zoom-(?:in|out)|dolly|rack(?:s|ing)? focus|close-?up|"
    r"pov|handheld|wide shot|wide-angle shot|tracking shot|crane shot|static shot|whip pan|"
    r"push[- ]in|pull[- ]out|camera shake)\b"
    r"|\bcamera(?:'s)?\s+(?:\w+ly\s+)?" + _CAMERA_VERBS + r"\b"
    r"|\bcamera(?:'s)? (?:movement|motion|move)\b"
    r"|(?:^|[.!?]\s+)(?:slowly |quickly |smoothly )?(?:pan|tilt|truck|pedestal|arc|orbit|roll|"
    r"push in|pull out|pull back|zoom|track|dolly|crane)\b",
    re.I)
GENERIC = {"camera", "video", "scene", "shot", "frame", "background", "foreground", "view",
           "image", "screen", "moment", "time", "way", "side", "front", "middle", "one",
           "other", "something", "everything", "thing", "music", "sound", "style", "lighting",
           "light", "atmosphere", "mood", "setting"}
PARTS = {"lip", "mouth", "tongue", "tooth", "teeth", "eye", "eyes", "face", "cheek", "hand",
         "finger", "hair", "neck", "shoulder", "waist", "hip", "leg", "foot", "feet", "arm",
         "skin", "expression", "smile", "gaze", "back", "chin", "brow"}
PLACES = {"room", "street", "kitchen", "bedroom", "bed", "table", "floor", "stage", "hall",
          "garden", "beach", "forest", "city", "car", "office", "bathroom", "field"}
MOVES_DETAIL = ("The camera pushes in toward {x}.", "The camera racks focus to {x}.",
                "The camera zooms in on {x}.")
MOVES_OTHER = ("The camera zooms in on {x}.", "The camera arcs around {x}.",
               "The camera pushes in toward {x}.")
MOVES_TRAVEL = ("The camera pans from {x} to {y}.", "The camera racks focus from {x} to {y}.",
                "The camera moves from {x} to {y}.")
# Chained moves (varied mode): a follow-up to a first move, and a closing pull-back.
MOVES_THEN = ("Then the camera pans to {y}.", "Then the camera racks focus to {y}.",
              "Then the camera pushes in toward {y}.")
MOVES_FINISH = ("Then the camera pulls out.", "Then the camera zooms out.",
               "Then the camera pulls out to a wider view.")
COUNT_WEIGHTS = (6, 3, 1)      # one move / two / three, in a shot that gets any
TRAVEL_SHARE = 0.5             # of one-move shots with two topics, how many travel X -> Y
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


def _name(c):
    """The phrase of candidate `c` as it reads in the prompt, tags restored, with an article."""
    text = _show(c[1])
    if not re.match(r"(?:the|a|an|<Subject|his|her|their)\b", text, re.I) and "'s " not in text:
        text = "the " + text
    return text


def _topics(pool):
    """The distinct things a shot dwells on, in the order the text reaches them. Only things
    with a role beyond being the doer (owned parts, objects of a verb or preposition)."""
    seen, out = set(), []
    for c in sorted(pool, key=lambda c: c[3]):
        if _score(c) > 0.5 and c[0] not in seen:
            seen.add(c[0])
            out.append(c)
    return out


def _pick(options, last, rng):
    """One of `options`, never the one the previous shot opened with when others exist."""
    pool = [o for o in options if o != last] or list(options)
    return rng.choice(pool) if rng else pool[0]


def _plan(pool, last, rng):
    """-> (moves text, target description, opening template). Fixed and minimal without
    `rng` (one move, X -> Y when the shot has two topics); with it, how many moves, whether
    they travel and which words are all drawn, so no shot pattern repeats by construction."""
    topics = _topics(pool)
    best = max(pool, key=_score)
    if rng is None:
        k, travel = 1, len(topics) >= 2
    else:
        k = rng.choices((1, 2, 3), COUNT_WEIGHTS)[0]
        travel = k == 1 and len(topics) >= 2 and rng.random() < TRAVEL_SHARE
    if travel:
        x, y = _name(topics[0]), _name(topics[-1])
        move = _pick(MOVES_TRAVEL, last, rng)
        return move.format(x=x, y=y), f"{x} -> {y}", move
    first = topics[0] if (k > 1 and topics) else best
    moves = MOVES_DETAIL if (first[2] or first[0] in PARTS) else MOVES_OTHER
    opening = _pick(moves, last, rng)
    target = _name(first)
    sentences = [opening.format(x=target)]
    desc = target
    if k > 1:
        if len(topics) >= 2:
            y = _name(topics[-1])
            sentences.append(_pick(MOVES_THEN, None, rng).format(y=y))
            desc += f" -> {y}"
        else:
            sentences.append(_pick(MOVES_FINISH, None, rng))
    if k > 2 and len(topics) >= 2:
        sentences.append(_pick(MOVES_FINISH, None, rng))
    return " ".join(sentences), desc, opening


def add_camera_moves(text, seed=None, chance=1.0):
    """-> (new prompt, [per-shot report dicts]). Unchanged when there is no `[Shot N]`.

    `seed=None`: one plain move per shot (deterministic). With a `seed`, each shot is left
    alone with probability 1 - `chance`, and otherwise gets one to three moves, drawn per
    shot from the seed, so the same prompt and seed always give the same text."""
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
        rng = random.Random(f"{seed}:{i}") if seed is not None else None
        pool = [c for c in cands[i] if c[0] not in constant]
        if CAMERA.search(CUT.sub("", picture)):
            entry["why"] = "already has a camera move"
        elif not pool:
            entry["why"] = "nothing specific to aim at"
        elif rng is not None and rng.random() >= chance:
            entry["why"] = "left as written by chance"
        else:
            moves, desc, opening = _plan(pool, last, rng)
            entry.update(move=moves, target=desc)
            last = opening
            core = picture.rstrip()
            gap = picture[len(core):] or (" " if sound else "")
            picture = core + ("" if core.endswith((".", "!", "?")) else ".") + " " \
                + moves + gap
        out.append(m.group(0) + picture + sound)
        report.append(entry)
    return "".join(out), report
