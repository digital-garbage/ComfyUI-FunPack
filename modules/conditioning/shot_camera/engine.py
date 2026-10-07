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

from ..._core import log

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
    r"handheld|wide shot|wide-angle shot|tracking shot|crane shot|static shot|whip pan|"
    r"camera shake)\b"
    r"|\bcamera(?:'s)?\s+(?:\w+ly\s+)?" + _CAMERA_VERBS + r"\b"
    r"|\bcamera(?:'s)? (?:movement|motion|move)\b"
    r"|(?:^|[.!?]\s+)(?:slowly |quickly |smoothly )?(?:pan|tilt|truck|pedestal|arc|orbit|roll|"
    r"zoom|track|dolly|crane)\b",
    re.I)
GENERIC = {"camera", "video", "scene", "shot", "frame", "background", "foreground", "view",
           "image", "screen", "moment", "time", "way", "side", "front", "middle", "one",
           "other", "something", "everything", "thing", "music", "sound", "style", "lighting",
           "light", "atmosphere", "mood", "setting"}
PARTS = {"lip", "mouth", "tongue", "tooth", "teeth", "eye", "eyes", "face", "cheek", "hand",
         "finger", "hair", "neck", "shoulder", "waist", "hip", "leg", "foot", "feet", "arm",
         "skin", "expression", "smile", "gaze", "back", "chin", "brow"}
# A liked detail may name one of these. Hair, face, skin and the like are appearance: the old
# refiner refused to auto-inject those, and a like of a clip cannot tell the person from the act.
DETAIL_PARTS = PARTS - {"hair", "face", "skin", "expression", "smile", "gaze", "back"}
# Adjectives that locate a part ("lower lip") are not a detail worth repeating everywhere.
_NOT_A_DETAIL = {"left", "right", "other", "same", "whole", "own", "both", "each", "another",
                 "first", "next", "last", "upper", "lower", "front", "back"}
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
MOVES_FINISH = ("Then the camera zooms out.", "Then the camera pulls back.",
               "Then the camera pulls back to a wider view.")
COUNT_WEIGHTS = (6, 3, 1)      # one move / two / three, in a shot that gets any
TRAVEL_SHARE = 0.25            # of one-move shots with two topics, how many travel X -> Y
HOLD_SHARE = 0.25              # of the other one-move shots, how many stay still on one thing
MOVES_HOLD = ("The camera holds steady on {x}.", "The camera stays still, focused on {x}.",
              "The camera stays fixed, focused on {x}.")
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}
POSSESSIVE_PRONOUNS = {"his", "her", "their", "its", "my", "your", "our"}

_nlp = None
_tried = False


def _say(what, error, effect):
    """One log line, once, when something falls back."""
    log.once(f"shot_camera:{what}", log.ALERT, "FunPack Shot Camera", f"{what}: {type(error).__name__}: {error} -- {effect}")


def _spacy():
    """The small English tagger, or None (said once) when it is not installed."""
    global _nlp, _tried
    if not _tried:
        _tried = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except Exception as e:  # noqa: BLE001
            fix = ("spacy not installed, run `pip install spacy`" if isinstance(e, ImportError)
                   else "model missing, run `python -m spacy download en_core_web_sm`")
            _say("spaCy tagger", e, f"simple word rules used; {fix}")
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


# Nouns that only mean something as part of a bigger thing: "the base" of what?
RELATIONAL = {"base", "tip", "top", "bottom", "side", "edge", "end", "middle", "part", "front",
              "back", "surface", "inside", "outside", "center", "centre", "head", "bit", "area",
              "spot", "line", "corner", "half"}


def _words(tok, with_det=False):
    keep = {tok}
    for c in tok.children:
        if c.i < tok.i and (c.dep_ in ("poss", "compound", "amod")
                            or (with_det and c.dep_ == "det")):
            keep.add(c)
            if c.dep_ == "poss":
                keep.update(g for g in c.children if g.dep_ == "case")
    toks = sorted(keep, key=lambda t: t.i)
    return " ".join(t.text for t in toks).strip().replace(" 's", "'s").replace(" ’s", "’s")


def _phrase(tok):
    """-> (text, owned, has_of): a chunk head with the words that name it ("Subject1's hand",
    "red lips", "base of the lamp"). An "of ..." that follows belongs to the noun and is kept:
    a target cut off before it ("the base") names nothing."""
    text = _words(tok)
    has_of = False
    for c in tok.children:
        if c.dep_ == "prep" and c.lower_ == "of":
            pobj = next((g for g in c.children if g.dep_ == "pobj"), None)
            if pobj is not None:
                text += " of " + _words(pobj, with_det=True)
                has_of = True
    return text, any(c.dep_ == "poss" for c in tok.children), has_of


def _resolve(lemma, text, owned, has_of, names):
    """The candidate's phrase made whole, or None when it cannot be: a relational noun with
    nothing to be the base/tip/side OF, or a body part with no owner when the shot has more
    than one person ("the skin": whose?). With exactly one person in the shot, they own it."""
    if has_of or owned:
        return text, True
    if lemma in RELATIONAL:
        return None
    if lemma in PARTS:
        if len(names) == 1:
            return f"{next(iter(names))}'s {text}", True
        return None
    return text, owned


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
            text, owned, has_of = _phrase(root)
            if _is_subject(text) or lemma in GENERIC or STAND_IN.fullmatch(root.text):
                continue
            whole = _resolve(lemma, text, owned, has_of, set(STAND_IN.findall(hidden)))
            if whole is None:
                continue
            dep, head = root.dep_, root
            while dep == "conj" and head.head is not head:      # "a banana and a cherry": both objects
                head = head.head
                dep = head.dep_
            out.append((lemma, whole[0], whole[1], chunk.start, dep))
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


def _content_amods(root):
    """Adjectives that describe `root`, ignoring ones that only say which side it is on."""
    return sorted((c for c in root.children
                   if c.dep_ == "amod" and c.is_alpha and c.lemma_.lower() not in _NOT_A_DETAIL),
                  key=lambda t: t.i)


def detail_phrases(picture):
    """[(lemma, phrase)] for a short detail already written on a body part ("detailed lips").

    A bare noun is not a detail. Neither is a place, a person, or hair and face: those are who
    the clip is about, and a like must not paste them onto the next one. With no tagger, one
    adjective in front of a body-part word is the same idea."""
    hidden = _hide(picture or "")
    nlp = _spacy()
    if nlp is None:
        return _detail_phrases_plain(hidden)
    out, seen = [], set()
    for root in (c.root for c in nlp(hidden).noun_chunks):
        lemma = root.lemma_.lower()
        if lemma not in DETAIL_PARTS:
            continue
        amods = _content_amods(root)
        if not amods:
            continue
        phrase = " ".join(c.text.lower() for c in amods) + " " + root.text.lower()
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if not phrase or phrase in seen or len(phrase.split()) > 5:
            continue
        seen.add(phrase)
        out.append((lemma, phrase))
    return out


def _detail_phrases_plain(hidden):
    parts = "|".join(sorted(DETAIL_PARTS, key=len, reverse=True))
    skip = DETERMINERS | POSSESSIVE_PRONOUNS | _NOT_A_DETAIL | {"and", "with", "from", "into", "onto", "over"}
    out, seen = [], set()
    for m in re.finditer(rf"\b([A-Za-z]{{3,}})\s+({parts})s?\b", hidden):
        adj = m.group(1).lower()
        if adj in skip:
            continue
        word = m.group(2).lower()
        lemma = word if word in DETAIL_PARTS else word[:-1]
        phrase = f"{adj} {m.group(2).lower()}"
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append((lemma, phrase))
    return out


def qualified_lemmas(picture):
    """Body-part lemmas this shot already describes with an adjective. A second detail would
    argue with that adjective, so nothing is added onto them."""
    hidden = _hide(picture or "")
    nlp = _spacy()
    if nlp is None:
        return {lemma for lemma, _phrase in _detail_phrases_plain(hidden)}
    out = set()
    for tok in nlp(hidden):
        if tok.lemma_.lower() in DETAIL_PARTS and any(c.dep_ == "amod" and c.is_alpha for c in tok.children):
            out.add(tok.lemma_.lower())
    return out


def detail_phrases_in(text):
    """Every detail phrase in a prompt, one per wording, read per shot so the sound track is skipped."""
    marks = list(SHOT.finditer(text or ""))
    chunks = ([text] if not marks else
              [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))] for i, m in enumerate(marks)])
    out, seen = [], set()
    for chunk in chunks:
        picture, _sound = _split_sound(chunk)
        for lemma, phrase in detail_phrases(picture):
            if phrase not in seen:
                seen.add(phrase)
                out.append((lemma, phrase))
    return out


def _as_sentence(phrase):
    phrase = (phrase or "").strip().rstrip(".")
    return (phrase[:1].upper() + phrase[1:] + ".") if phrase else ""


def add_detail_notes(text, seed=0, chance=0.5, bank=()):
    """-> (prompt, {"added": [{"shot", "lemma", "phrase"}], "why"}).

    A banked detail is added once, as its own sentence at the end of a shot, and only when that
    shot already names the body part, the part is not already described, and the words are not
    already there. One detail per shot: the one ratings have kept most. The original sentence
    is never rewritten, so it cannot be made ungrammatical. `chance` of 0 or 1 is kept as set;
    anything between is the caller's, already tilted per phrase."""
    marks = list(SHOT.finditer(text or ""))
    info = {"added": [], "why": ""}
    if not marks:
        info["why"] = "no shots"
        return text, info
    usable = [b for b in bank or () if isinstance(b, dict) and b.get("lemma") and b.get("phrase") and float(b.get("good") or 0) >= 1]
    if not usable:
        info["why"] = "nothing learned yet"
        return text, info
    out = [text[:marks[0].start()]]
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        picture, sound = _split_sound(text[m.end():end])
        present = {c[0] for c in candidates(picture)}
        taken = qualified_lemmas(picture)
        low = picture.lower()
        options = [b for b in usable
                   if b["lemma"] in present and b["lemma"] not in taken and b["phrase"].lower() not in low]
        options.sort(key=lambda b: (-(float(b["good"]) - float(b["bad"])), -float(b["good"]), b["phrase"]))
        if options and random.Random(f"{seed}:detail:{i}:{options[0]['phrase']}").random() < float(options[0].get("chance", chance)):
            sentence = _as_sentence(options[0]["phrase"])
            picture = picture.rstrip() + " " + sentence + " "
            info["added"].append({"shot": int(m.group(1)), "lemma": options[0]["lemma"], "phrase": options[0]["phrase"]})
        out.append(m.group(0) + picture + sound)
    if not info["added"]:
        info["why"] = "nothing matched"
    return "".join(out), info


def _score(c, prior=None):
    """How much `c` deserves the camera: what it is in its own prompt (an owned part, an object
    of a verb), plus, when given, `prior`'s bonus for words the user's prompts keep returning to."""
    lemma, _text, owned, pos, dep = c
    return ((2 if owned or lemma in PARTS else 0) + (1 if dep in ("dobj", "pobj") else 0)
            + (prior or {}).get(lemma, 0.0) - pos * 1e-4)


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


HABIT_TEMPERATURE = 0.7     # lower = the habit decides more; higher = closer to a coin toss
MENTION_BONUS = 0.5         # per extra mention inside the shot, capped at 1.0: what a shot keeps returning to


def _draw(cands, prior, rng, n):
    """`n` distinct candidates. Only those within a role-step of the best plain score compete (a
    habit can never lift a background word over a real target); among them, with an `rng`, the
    pick is drawn with odds exp(score / T) rather than always taking the top, so a word the user
    has used a hundred times does not win every shot it merely appears in."""
    top = max(_score(c) for c in cands)
    eligible = [c for c in cands if _score(c) >= top - 0.99] or list(cands)
    if rng is None:
        return sorted(eligible, key=lambda c: -_score(c, prior))[:n]
    chosen, pool = [], list(eligible)
    while pool and len(chosen) < n:
        weights = [math.exp(_score(c, prior) / HABIT_TEMPERATURE) for c in pool]
        pick = rng.choices(range(len(pool)), weights)[0]
        chosen.append(pool.pop(pick))
    return chosen


def _tilt(base, key, arms):
    """`base` odds tilted by how the ratings treated arm `key` (0.5-1.5x; 1 when unrated)."""
    g, b = (arms or {}).get(key, (0.0, 0.0))
    return base * (0.5 + (g + 1.0) / (g + b + 2.0))


def _plan(pool, last, rng, prior=None, force=None, mode="auto", arms=None):
    """-> (moves text, target description, opening template, style). Fixed and minimal without
    `rng` (one move, X -> Y when the shot has two topics); with it, how many moves, whether
    they travel and which words are all drawn, so no shot pattern repeats by construction.
    `arms`: {"style:hold": (good, bad), ...} what ratings taught; tilts those draws. `style` is
    the arm that was drawn ("hold", "travel", "k1".."k3"), None when a person decided."""
    topics = _topics(pool)
    best = _draw(pool, prior, rng, 1)[0]
    if prior and len(topics) > 2:                 # two of them, weighted by habit, in text order
        topics = sorted(_draw(topics, prior, rng, 2), key=lambda c: c[3])
    if force:                                     # the user picked what to aim at, in this order
        best, topics = force[0], list(force)
    if mode == "hold":
        hold = _pick(MOVES_HOLD, last, rng)
        target = _name(best)
        return hold.format(x=target), target, hold, None
    if force and len(force) >= 2:
        # "from here to there (and on to...)": the camera visits the picks in the order given.
        move = _pick(MOVES_TRAVEL, last, rng)
        sentences = [move.format(x=_name(force[0]), y=_name(force[1]))]
        sentences += [_pick(MOVES_THEN, None, rng).format(y=_name(c)) for c in force[2:]]
        return " ".join(sentences), " -> ".join(_name(c) for c in force), move, None
    if mode == "move":
        k, travel = 1, False
    elif rng is None:
        k, travel = 1, len(topics) >= 2
    else:
        k = rng.choices((1, 2, 3), [_tilt(w, f"style:k{n}", arms) for n, w in zip((1, 2, 3), COUNT_WEIGHTS)])[0]
        travel = k == 1 and len(topics) >= 2 and rng.random() < _tilt(TRAVEL_SHARE, "style:travel", arms)
    if travel:
        x, y = _name(topics[0]), _name(topics[-1])
        move = _pick(MOVES_TRAVEL, last, rng)
        return move.format(x=x, y=y), f"{x} -> {y}", move, "travel" if rng else None
    if rng is not None and k == 1 and rng.random() < _tilt(HOLD_SHARE, "style:hold", arms):
        hold = _pick(MOVES_HOLD, last, rng)              # some shots want one fixed focus
        target = _name(best)
        return hold.format(x=target), target, hold, "hold"
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
    return " ".join(sentences), desc, opening, (f"k{k}" if rng is not None and mode == "auto" else None)


def _analyse(text):
    """-> (marks, header, parts, pools): the `[Shot N]` markers, the text before the first, each
    shot's (picture, sound) and its candidate targets with the things that are in (nearly) every
    shot, or in the intro, taken out."""
    marks = list(SHOT.finditer(text or ""))
    if not marks:
        return [], "", [], []
    header = text[:marks[0].start()]
    bodies = [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
              for i, m in enumerate(marks)]
    parts = [_split_sound(b) for b in bodies]
    cands = [candidates(own_words_removed(p[0])) for p in parts]
    header_lemmas = {c[0] for c in candidates(header)}
    constant = _constants([p[0] for p in parts], cands, header_lemmas)
    return marks, header, parts, [[c for c in lst if c[0] not in constant] for lst in cands]


def _content_key(picture):
    """Hash of a shot's own words. Cut times, views and camera sentences are not part of it."""
    import hashlib
    body = own_words_removed(picture or "")
    body = re.sub(r"\s+", " ", body).strip().lower()
    return hashlib.md5(body.encode("utf-8", "replace")).hexdigest()[:16]


def shot_key(picture, occurrence=0):
    """A stable name for one take of a shot. The same whatever cut opener, view or camera
    sentence was added, so a choice still finds it after the prompt is regenerated.
    The same shortcut used again is a later take (`:1`, `:2`), so a focus or view chosen
    for the first does not lock the next."""
    base = _content_key(picture)
    return base if not occurrence else f"{base}:{int(occurrence)}"


def _take_keys():
    """A counter: the first copy of a shot's words keeps the plain key, the next get :1, :2."""
    seen = {}

    def key(picture):
        base = _content_key(picture)
        n = seen.get(base, 0)
        seen[base] = n + 1
        return shot_key(picture, n)
    return key


def _constants(pictures, lists, header_lemmas):
    """Lemmas in (nearly) every distinct shot, plus whatever the intro already named.
    Two copies of the same shortcut count once: DANCE DANCE must not strip the only
    things the next take can focus on. 70%: with three different shots, a noun shared
    by two of them is still a topic of those two."""
    groups = {}
    for picture, lst in zip(pictures, lists):
        groups.setdefault(_content_key(picture), set()).update(c[0] for c in lst)
    if len(groups) <= 1:
        return set(header_lemmas)
    need = max(2, math.ceil(0.7 * len(groups)))
    seen = {}
    for lemmas in groups.values():
        for lemma in lemmas:
            seen[lemma] = seen.get(lemma, 0) + 1
    return {lemma for lemma, n in seen.items() if n >= need} | set(header_lemmas)


def focus_options(text, prior=None):
    """What the camera could aim at in each `[Shot N]`, for a person to choose from.
    -> [{"shot", "key", "already", "candidates": [{"lemma", "text", "score"}], "auto": str|None}]
    `auto` is what the rewriter would pick with no say from anyone (its plain, top choice)."""
    marks, _header, parts, pools = _analyse(text)
    out, keys = [], _take_keys()
    for i, m in enumerate(marks):
        picture = parts[i][0]
        pool = pools[i]
        opts, seen = [], set()
        for c in sorted(pool, key=lambda c: -_score(c, prior)):
            if c[0] in seen or _score(c) <= 0.5:
                continue
            seen.add(c[0])
            opts.append({"lemma": c[0], "text": _name(c), "score": round(_score(c, prior), 2)})
        auto = None
        if opts and not CAMERA.search(CUT.sub("", picture)):
            auto = _plan(pool, None, None, prior)[1]
        out.append({"shot": int(m.group(1)), "key": keys(picture),
                    "auto_lemma": opts[0]["lemma"] if opts else None,
                    "already": bool(CAMERA.search(CUT.sub("", picture))),
                    "candidates": opts, "auto": auto})
    return out


def add_camera_moves(text, seed=None, chance=1.0, prior=None, choices=None, arms=None):
    """-> (new prompt, [per-shot report dicts]). Unchanged when there is no `[Shot N]`.

    `seed=None`: one plain move per shot (deterministic). With a `seed`, each shot is left
    alone with probability 1 - `chance`, and otherwise gets one to three moves, drawn per
    shot from the seed, so the same prompt and seed always give the same text.
    `arms`: what ratings taught (see _plan); each report entry's "arms" lists the choices drawn
    on their own for the rating that follows ("move:yes"/"move:no", a style, "word:<lemma>")."""
    marks, header, parts, pools = _analyse(text)
    if not marks:
        return text, []
    report, out, last, keys = [], [header], None, _take_keys()
    for i, m in enumerate(marks):
        picture, sound = parts[i]
        entry = {"shot": int(m.group(1)), "move": None, "target": None, "why": "", "lemmas": [],
                 "key": keys(picture), "arms": []}
        rng = random.Random(f"{seed}:{i}") if seed is not None else None
        pool = pools[i]
        # What the person said about this shot, if anything: "none", or a target and/or a mode.
        said = (choices or {}).get(entry["key"]) or {}
        mode = said.get("mode") or "auto"
        wanted = said.get("lemmas") or ([said["lemma"]] if said.get("lemma") else [])
        force = [c for c in (next((c for c in pool if c[0] == l), None) for l in wanted) if c]
        entry["lemmas"] = [c[0] for c in pool if _score(c) > 0.5]      # the words this shot dwells on
        # A word the shot keeps coming back to is what it is about: a small lift per extra mention.
        mentions = {}
        for c in pool:
            mentions[c[0]] = mentions.get(c[0], 0) + 1
        boost = {l: min(1.0, MENTION_BONUS * (n - 1)) for l, n in mentions.items() if n > 1}
        shot_prior = ({l: (prior or {}).get(l, 0.0) + boost.get(l, 0.0) for l in {*(prior or {}), *boost}}
                      if boost else prior)
        if CAMERA.search(CUT.sub("", picture)):
            entry["why"] = "already has a camera move"
        elif not pool:
            entry["why"] = "nothing specific to aim at"
        elif mode == "none":
            entry["why"] = "no move, as you chose"
        elif rng is not None and mode == "auto" and not force and rng.random() >= chance:
            entry["why"] = "left as written by chance"
            entry["arms"] = ["move:no"]
        else:
            moves, desc, opening, style = _plan(pool, last, rng, shot_prior, force=force, mode=mode, arms=arms)
            entry.update(move=moves, target=desc)
            if rng is not None and mode == "auto" and not force:    # only what nobody dictated is learned
                entry["arms"] = ["move:yes"] + ([f"style:{style}"] if style else [])
                names = desc.split(" -> ")
                entry["arms"] += [f"word:{c[0]}" for c in {c[0]: c for c in pool if _name(c) in names}.values()]
            last = opening
            core = picture.rstrip()
            gap = picture[len(core):] or (" " if sound else "")
            picture = core + ("" if core.endswith((".", "!", "?")) else ".") + " " \
                + moves + gap
        out.append(m.group(0) + picture + sound)
        report.append(entry)
    return "".join(out), report


# ── shot cuts ───────────────────────────────────────────────────────────────────────
# H3 wants every shot after the first to open with its cut time and a cut phrase
# ("[Shot 2] At 00:03.000, the camera cuts to ..."), times increasing and inside the video.
# A shot that holds several shortcuts is cut between them, the same shortcut used again
# included ("[Shot 1] DANCE KISS", "[Shot 1] DANCE DANCE"), so each part can take a time,
# a view and a focus. Never inside one shortcut. Nothing in a prompt knows the video's
# length, so the caller hands it in (seconds).
STAMP = re.compile(r"\b\d\d:\d\d\.\d{3}\b")
CUT_OPENERS = ("the camera cuts to a new angle", "the shot transitions to the next moment",
               "the shot changes to a new view", "the shot switches to what follows")
LEAD_CUT = re.compile(r"^\s*(?:the\s+)?(?:camera|shot)\s+(?:cuts|transitions|changes|switches)\s+to\b",
                      re.I)
MIN_SHOT_SECONDS = 2


def _stamp(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}.000"


def _whole_piece(picture, i, piece):
    """True when `piece` starts at `i` on a word edge, not as a prefix of a longer word."""
    end = i + len(piece)
    before = i == 0 or picture[i - 1].isspace()
    after = end >= len(picture) or picture[end].isspace() or picture[end] in ".!?,;:"
    return before and after and picture.startswith(piece, i)


def _cut_points(picture, pieces):
    """Indexes where a later shortcut starts inside this one shot. The same text used again
    counts, so DANCE DANCE is two takes. Longest shortcut wins, so a cut never lands inside one."""
    ordered = sorted({p.strip() for p in pieces if isinstance(p, str) and p.strip()}, key=len, reverse=True)
    if not ordered or not picture:
        return []
    hits, i = [], 0
    while i < len(picture):
        match = next((p for p in ordered if _whole_piece(picture, i, p)), None) if (
            i == 0 or picture[i - 1].isspace()) else None
        if match:
            hits.append(i)
            i += len(match)
        else:
            i += 1
    return hits[1:]


def _fresh_piece(text):
    """A part cut off the front of a shot: drop a leading 'Then', keep the capital."""
    text = re.sub(r"^\s*(?:then|next|after that|afterwards),?\s+", "", text or "", flags=re.I)
    return (text[:1].upper() + text[1:]) if text else text


def add_shot_cuts(text, seconds, seed=0, chance=0.5, pieces=()):
    """-> (new prompt, info). A shot is cut between the shortcuts inside it, including the same
    shortcut used again; every shot after the first then opens with its cut time, spread evenly
    over `seconds` and rounded to whole seconds. A prompt that already carries cut times is left
    alone. `pieces`: the texts of the user's shortcuts. With none given nothing is split (times
    are still added). info = {"before", "after", "times": [...], "why": str}."""
    marks = list(SHOT.finditer(text or ""))
    info = {"before": len(marks), "after": len(marks), "times": [], "why": "", "arms": []}
    if not marks:
        return text, info
    if not seconds or seconds < MIN_SHOT_SECONDS:
        info["why"] = "the video's length is not known here" if not seconds else "the video is too short"
        return text, info
    bodies = [text[m.end():(marks[i + 1].start() if i + 1 < len(marks) else len(text))]
              for i, m in enumerate(marks)]
    if any(STAMP.search(b[:60]) for b in bodies[1:]):
        info["why"] = "cut times are already written, left as they are"
        return text, info
    header = text[:marks[0].start()]
    parts = [_split_sound(b) for b in bodies]
    blocks = []                                   # [picture text, sound text]
    for i, (picture, sound) in enumerate(parts):
        chosen = []
        for n, at in enumerate(_cut_points(picture, pieces)):
            rng = random.Random(f"{seed}:cut:{i}" if n == 0 else f"{seed}:cut:{i}:{n}")
            take = rng.random() < chance
            info["arms"].append("split:yes" if take else "split:no")
            if take:
                chosen.append(at)
        if not chosen:
            blocks.append([picture, sound])
            continue
        prev = 0
        for at in chosen:
            blocks.append([picture[prev:at].rstrip() + " ", ""])
            prev = at
        blocks.append([_fresh_piece(picture[prev:]), sound])
    if len(blocks) > len(marks) and (len(blocks) > seconds or seconds / len(blocks) < MIN_SHOT_SECONDS):
        blocks = [[p, s] for p, s in parts]         # not enough seconds for the splits
        info["arms"] = []
    # Equal shares of the scene, rounded to whole seconds, no shot shorter than MIN_SHOT_SECONDS
    # (a shortcut's text length says nothing about how long its beat should last: weighting by
    # it gave one-second shots next to five-second ones).
    n = len(blocks)
    gap = MIN_SHOT_SECONDS if n * MIN_SHOT_SECONDS <= seconds else 1
    times = []
    for k in range(1, n):
        t = round(k * seconds / n)
        t = max(t, (times[-1] if times else 0) + gap)          # not too close behind
        t = min(t, int(seconds) - (n - k) * gap)               # leave room for the shots after
        times.append(t)
    if times and (times[0] < 1 or times[-1] >= seconds or any(b <= a for a, b in zip(times, times[1:]))):
        info["why"] = f"{len(blocks)} shots do not fit in {seconds}s at whole seconds"
        info["arms"] = []
        return text, info
    out = [header]
    for k, (picture, sound) in enumerate(blocks):
        label = f"[Shot {k + 1}]"
        body = picture
        if k:
            stamp = f"At {_stamp(times[k - 1])}, "
            rng = random.Random(f"{seed}:opener:{k}")
            if LEAD_CUT.match(body):
                body = " " + stamp + LEAD_CUT.sub(lambda m: m.group(0).strip().lower(), body.lstrip(), 1)
            else:
                body = " " + stamp + rng.choice(CUT_OPENERS) + ". " + body.lstrip()
        else:
            body = body if body.startswith(" ") else " " + body
        out.append(label + body + sound)
    info.update(after=len(blocks), times=[_stamp(t) for t in times])
    return "".join(out), info


# ── views ───────────────────────────────────────────────────────────────────────────
# Chosen, not detected: a prompt seldom says which view it wants. The user's own trusted
# wording, verbatim. Shot 1 is skipped: it may sit on a reference image or a pinned first
# frame, which a new view would contradict; every later shot starts on a new angle anyway.
VIEWS = ("POV view", "POV view from above", "POV view from below", "Side view",
         "View from above", "View from below", "View from behind", "Front view")
# A view the shot already asks for. "from behind" / "from above" alone are also body actions
# ("enters from behind", "drips from above"), so they count only when a camera word goes with
# them ("seen from behind", "shot from above", "camera from below").
VIEW_STATED = re.compile(r"\b(pov|(?:side|front|rear|back|top|overhead|bird'?s[- ]eye)[- ]view|view from|"
                         r"(?:seen|shot|filmed|viewed|captured|camera|angle|footage)\b[^.!?,;]{0,20}?"
                         r"\bfrom (?:above|below|behind))\b", re.I)
OPENER = re.compile(r"^\s*At \d\d:\d\d\.\d{3},[^.!?]*[.!?]\s*")


# What a shot's own words say about which views can show it. A view that cannot show what the
# shot describes (the back of someone whose face is the point, the front of someone walking
# away) only confuses the model, so it is never offered. Traits also key what the ratings
# teach: a view that works for a face shot may not work for a hands shot.
VIEW_TRAITS = {
    "face": re.compile(r"\b(faces?|eyes?|eye contact|mouths?|lips|smil\w*|expressions?|gaze|"
                       r"cheeks?|facing|looks? (?:at|into))\b", re.I),
    "back": re.compile(r"\b(back(?! and| to| into| off| up| down)|rear|spine|shoulder blades?|nape|"
                       r"walks? away|turned away|from behind)\b", re.I),
    "hands": re.compile(r"\b(hands?|fingers?|you|your|reach\w*|hold\w*|touch\w*)\b", re.I),
}
_FRONT_ONLY = ("POV view", "POV view from above", "POV view from below", "Front view")


def view_traits(picture):
    body = own_words_removed(picture or "")
    return sorted(t for t, rx in VIEW_TRAITS.items() if rx.search(body))


def allowed_views(traits):
    """The views that can show a shot with these traits: a face excludes the view from behind;
    a back (or someone leaving) excludes every view from the front, POV included."""
    out = list(VIEWS)
    if "face" in traits:
        out = [v for v in out if v != "View from behind"]
    if "back" in traits:
        out = [v for v in out if v not in _FRONT_ONLY]
    return out


def _view_weight(view, traits, stats):
    """How much a view is liked for a shot like this: the smoothed good-rate of the view, and
    of the view on each of this shot's traits, averaged. 0.5 when nothing was rated yet."""
    rates = []
    for key in [view] + [f"{view}@{t}" for t in (traits or ["none"])]:
        g, b = (stats or {}).get(key, (0.0, 0.0))
        if g + b > 0:
            rates.append((g + 1.0) / (g + b + 2.0))
    return 0.15 + (sum(rates) / len(rates) if rates else 0.5)


def shot_texts(text):
    """{shot number: its picture text, without what this module wrote into it}: what the
    Reactive focus review shows so a choice is about a shot the person can read."""
    out = {}
    marks = list(SHOT.finditer(text or ""))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        picture, _sound = _split_sound(text[m.end():end])
        out[int(m.group(1))] = " ".join(own_words_removed(picture).split())
    return out


def view_options(text, stats=None):
    """What each `[Shot N]` after the first could open with, for a person to choose from.
    -> [{"shot", "key", "traits", "already", "candidates": [{"view", "score"}], "auto"}]"""
    marks = list(SHOT.finditer(text or ""))
    out, keys = [], _take_keys()
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        picture, _sound = _split_sound(text[m.end():end])
        key = keys(picture)                  # count shot 1 too, even though it is not offered
        if not i and len(marks) > 1:         # see add_shot_views: a lone shot may take a view
            continue
        traits = view_traits(picture)
        cands = sorted(((v, _view_weight(v, traits, stats)) for v in allowed_views(traits)),
                       key=lambda c: -c[1])
        stated = VIEW_STATED.search(CUT.sub("", picture))
        out.append({"shot": int(m.group(1)), "key": key, "traits": traits,
                    "already": bool(stated), "stated": stated.group(0) if stated else "",
                    "candidates": [{"view": v, "score": round(w, 2)} for v, w in cands],
                    "auto": cands[0][0] if cands else None})
    return out


def add_shot_views(text, seed=0, chance=0.4, stats=None, choices=None, skipped=None):
    """-> (new prompt, [{"shot", "view", "traits"}]). Shots 2+ (or the only shot) that state no view get one of the
    views that can show them as a sentence of its own (after the cut opener, if there is one),
    never the same as the shot before, drawn per shot from the seed and weighted by `stats`
    (what ratings taught). `choices` {shot_key: {"mode": "auto"|"none"|"pick", "view": str}}:
    a person's pick is always used, "none" leaves the shot alone. The same shortcut used again
    is a different key, so one pick does not lock the next take. `skipped`, if a list, gets one
    "shot N: reason" per shot that got no view."""
    marks = list(SHOT.finditer(text or ""))
    if not marks:
        return text, []
    # Shot 1 of several may sit on a reference image or a pinned first frame, which a view would
    # contradict, so views open shots 2+. A prompt with ONE shot has nothing else to open.
    lone = len(marks) == 1
    out, added, last, keys = [text[:marks[0].start()]], [], None, _take_keys()
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        rng = random.Random(f"{seed}:view:{i}")
        picture, sound = _split_sound(body)
        key = keys(picture)
        view = None
        stated = VIEW_STATED.search(CUT.sub("", picture)) if (i or lone) else None
        if stated and skipped is not None:
            skipped.append(f"shot {m.group(1)}: already states “{stated.group(0)}”")
        if (i or lone) and not stated:
            traits = view_traits(picture)
            said = (choices or {}).get(key) or {}
            allowed = allowed_views(traits)
            if said.get("mode") == "none":
                pass
            elif said.get("views") or said.get("view"):
                # "fine from these": a pick is always used; with several, one of them is drawn.
                wanted = [v for v in (said.get("views") or [said.get("view")]) if v in allowed]
                pool = [v for v in wanted if v != last] or wanted
                if pool:
                    view = rng.choices(pool, weights=[_view_weight(v, traits, stats) for v in pool])[0]
            elif rng.random() < chance:
                pool = [v for v in allowed if v != last] or allowed
                if pool:
                    view = rng.choices(pool, weights=[_view_weight(v, traits, stats) for v in pool])[0]
            if not view and skipped is not None:
                skipped.append(f"shot {m.group(1)}: " + ("no view, as you chose" if said.get("mode") == "none"
                               else "left as written by chance"))
        if view:
            head = OPENER.match(picture)
            cut = head.end() if head else len(picture) - len(picture.lstrip())
            picture = picture[:cut].rstrip() + (" " if cut else " ") + view + ". " + picture[cut:].lstrip()
            body = picture + sound
            added.append({"shot": int(m.group(1)), "view": view, "traits": traits})
            last = view
        out.append(m.group(0) + body)
    return "".join(out), added


_VIEW_SENTENCE = re.compile(r"^\s*(?:" + "|".join(re.escape(v) for v in VIEWS) + r")\.\s*", re.I)


def own_words_removed(picture):
    """`picture` without what this module wrote into it (its own cut openers and views) and
    without the bare cut words of a user-written opener ("At 00:03.500, the camera cuts to"),
    whose content after them is kept: those words are not the shot's content and must never
    become a target or a topic."""
    picture = _OUR_OPENER.sub("", picture, count=1)
    picture = _CUT_WORDS.sub("", picture, count=1)
    return _VIEW_SENTENCE.sub("", picture, count=1)


_OUR_OPENER = re.compile(r"^\s*At \d\d:\d\d\.\d{3},\s*(?:" + "|".join(re.escape(o) for o in CUT_OPENERS)
                         + r")\.\s*", re.I)
_CUT_WORDS = re.compile(r"^\s*(?:At \d\d:\d\d\.\d{3},\s*)?(?:the\s+)?(?:camera|shot)\s+"
                        r"(?:cuts|transitions|changes|switches)\s+to\b\s*", re.I)


def content_fingerprint(text):
    """A hash of a prompt's own content: the same whatever cut times, openers, views or shot
    labels this module added on that run, so regenerating one prompt is one prompt."""
    import hashlib
    body = re.sub(r"\[\s*Shot\s+\d+\s*\]", " ", text or "", flags=re.I)
    body = re.sub(r"At \d\d:\d\d\.\d{3},\s*(?:" + "|".join(re.escape(o) for o in CUT_OPENERS) + r")\.",
                  " ", body, flags=re.I)
    body = re.sub(r"\s+", " ", body)
    body = re.sub(r"(?:^|(?<=[.!?]\s))(?:" + "|".join(re.escape(v) for v in VIEWS) + r")\.", " ", body)
    body = re.sub(r"\s+", " ", body).strip().lower()
    return hashlib.md5(body.encode("utf-8", "replace")).hexdigest()
