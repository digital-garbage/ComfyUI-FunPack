"""Multi-refinement-key support: shortcut->key binding, per-scene attribution, fallback."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRoutes:
    # Mirror every verb server.py registers (this fake is installed into sys.modules
    # as `server`, so a real server.py import elsewhere registers against it).
    def get(self, _p):
        return lambda f: f

    def post(self, _p):
        return lambda f: f

    def put(self, _p):
        return lambda f: f

    def delete(self, _p):
        return lambda f: f

    def patch(self, _p):
        return lambda f: f


sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir="", get_user_directory=lambda: ""))
sys.modules.setdefault(
    "server",
    types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=_FakeRoutes()))),
)

import templates
import conditioning
from templates import apply_prompt_shortcuts, normalize_shortcut_item
from movie_editor.backend import bridge


def _db(shortcuts):
    """Build a normalized shortcut DB from {name: (triggers, replacements, key)}."""
    out = {"version": 1, "source": "test", "shortcuts": {}}
    for name, (triggers, reps, key) in shortcuts.items():
        item = normalize_shortcut_item({
            "name": name, "triggers": triggers, "replacements": reps, "refinement_key": key,
        })
        out["shortcuts"][templates.shortcut_key(name)] = item
    return out


def _norm(s):
    import re as _re
    return _re.sub(r"\s+", " ", str(s or "")).strip()


def test_verbatim_split_when_trigger_is_also_a_shortcut(monkeypatch):
    """Regression: a transition trigger that is ALSO a shortcut (qcut/cut) was expanded away
    before transition detection, losing/misplacing the split -> 'scene 2 removed, scene 1 shows
    scene 2's prompt'. The verbatim splitter must also scan the original text and catch it."""
    db = _db({
        "char": (["CHAR:"], ["A cinematic film."], ""),
        "man": (["man"], ["a tall man"], ""),
        "qcut": (["qcut"], ["fast paced editing"], "cuts"),  # word 'qcut' disappears on expand
        "cut": (["cut"], ["Scene cut."], ""),
        "10": (["10"], ["an explosion"], "key10"),
        "20": (["20"], ["a chase"], "key20"),
        "30": (["30"], ["a kiss"], "key30"),
    })
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"qcut": {"placement": "start", "visual_effect": None},
                                 "cut": {"placement": "silent", "visual_effect": None}})
    v = conditioning.split_timeline_verbatim("CHAR: a woman man qcut 10 cut 20 30")
    assert len(v["scenes"]) == 2
    assert _norm(v["anchor"]) == "CHAR: a woman man"
    # action 10 in scene 1, actions 20 & 30 in scene 2 (cut is silent -> trails scene 1)
    assert "10" in v["scenes"][0]["text"] and "20" in v["scenes"][1]["text"] and "30" in v["scenes"][1]["text"]
    # lossless round-trip
    rejoined = " ".join(x for x in [v["anchor"]] + [s["text"] for s in v["scenes"]] if x)
    assert _norm(rejoined) == _norm("CHAR: a woman man qcut 10 cut 20 30")


def test_verbatim_split_shortcut_driven_split_still_works(monkeypatch):
    """A shortcut whose trigger is NOT a transition but whose EXPANSION resolves to one must
    still create a boundary (the supported shortcut-driven split)."""
    db = _db({"break": (["<break>"], ["Scene cut."], "")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "silent", "visual_effect": None}})
    v = conditioning.split_timeline_verbatim("intro shot <break> the night falls")
    assert len(v["scenes"]) == 1
    assert "<break>" in v["anchor"]


def test_split_scenes_splits_on_expansion_keyed_transition(monkeypatch):
    """Regression: split_scenes (the canonical generation/preview/key splitter) must cut when a
    shortcut's EXPANSION is a transition phrase (e.g. 'cut'->'Scene cut.'), not only when the
    shortcut's TRIGGER word is itself in the transition DB. It previously matched only the raw
    trigger, so an expansion-keyed split marker produced ONE scene here while the verbatim editor
    split produced TWO — and the disagreement collapsed the timeline's middle scenes."""
    db = _db({
        "cut": (["cut"], ["Scene cut."], ""),
        "qcut": (["qcut"], ["The video rapidly cuts, showing the next view."], ""),
    })
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {
        "scene cut.": {"placement": "silent", "visual_effect": None},
        "the video rapidly cuts, showing the next view.": {"placement": "start", "visual_effect": None},
    })
    prompt = "a dog runs cut a cat sleeps cut a bird flies"
    s = conditioning.split_scenes(prompt)
    assert len(s["scenes"]) == 2           # silent 'cut' x2 -> anchor + 2 scenes
    assert s["anchor_expanded"] == "a dog runs"
    # The canonical split must agree with the verbatim editor split (no collapse).
    v = conditioning.split_timeline_verbatim(prompt)
    assert len(v["scenes"]) == len(s["scenes"])
    # Non-silent 'qcut' too.
    assert len(conditioning.split_scenes("a dog runs qcut a cat sleeps qcut a bird flies")["scenes"]) == 2


def test_postfix_shortcuts_expand_and_key_folds_into_every_scene(monkeypatch):
    """A shortcut in the postfix must be EXPANDED at generation (not passed verbatim) and its
    refinement key must fold into every scene — exactly like the anchor. The editor stores the
    raw postfix; expansion happens here on the generation hit."""
    db = _db({
        "cine": (["cine"], ["cinematic, shallow depth of field"], "look"),
    })
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})

    out = conditioning.split_scenes_from_segments(
        "a knight", ["in a forest", "at a castle"], postfix="cine")
    # The shortcut is expanded, not left as the literal trigger 'cine'.
    assert out["postfix_expanded"] == "cinematic, shallow depth of field"
    assert out["postfix"] == "cine"  # raw trigger preserved for round-trip
    # The postfix's refinement key folds into EVERY scene (like the anchor's).
    for scene in out["scenes"]:
        assert "look" in scene["keys"]


def test_expand_prompt_fragment_matches_generation_postfix(monkeypatch):
    """The preview's postfix text must be expanded exactly like generation appends it — same
    shortcut expansion the chain sampler runs via split_scenes_from_segments."""
    db = _db({"cine": (["cine"], ["cinematic, shallow depth of field"], "look")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})

    preview = bridge.expand_prompt_fragment("cine")
    generation = conditioning.split_scenes_from_segments(
        "", ["x"], postfix="cine")["postfix_expanded"]
    assert preview == "cinematic, shallow depth of field"
    assert preview == generation                       # preview mirrors generation exactly
    assert bridge.expand_prompt_fragment("") == ""     # disabled/empty postfix shows nothing


def test_split_scenes_no_false_cut_from_generic_label_in_expansion(monkeypatch):
    """A content shortcut whose expansion merely CONTAINS 'scene 2' must NOT cut — only a
    user-defined custom transition phrase does (the generic label is never matched on expansions)."""
    db = _db({"desc": (["desc"], ["a wide shot like scene 2 of a film"], "")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    assert len(conditioning.split_scenes("a dog runs desc here")["scenes"]) == 1


def test_verbatim_split_still_splits_on_literal_cuts(monkeypatch):
    """The fix must not suppress real, user-typed scene cuts."""
    db = _db({"closeup": (["<closeup>"], ["a tight close-up"], "closeupkey")})  # no cut in replacement
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    v = conditioning.split_timeline_verbatim("wide intro. scene 1 he walks <closeup>. scene 2 the door opens")
    assert len(v["scenes"]) == 2


def test_shortcut_replacement_keeps_commas():
    """A replacement is a prose phrase: commas must NOT split it into variants (regression for
    'qcut' replacement 'The video rapidly cuts, showing the next view.' becoming two variants,
    which never matched the comma-form transition trigger and reverted on every save)."""
    phrase = "The video rapidly cuts, showing the next view."
    assert templates._shortcut_replacements([phrase]) == [phrase]      # list path (editors)
    assert templates._shortcut_replacements(phrase) == [phrase]        # string fallback
    # newlines still separate variants; commas inside each survive
    assert templates._shortcut_replacements("a, b\nc, d") == ["a, b", "c, d"]
    item = normalize_shortcut_item({"name": "rapidcut", "triggers": ["qcut"],
                                    "replacements": [phrase], "refinement_key": "cuts"})
    assert item["replacements"] == [phrase]


def test_normalize_shortcut_item_carries_key():
    item = normalize_shortcut_item({"name": "x", "triggers": ["alpha"], "replacements": ["A"],
                                    "refinement_key": "  charA  "})
    assert item["refinement_key"] == "charA"
    # "-None-" / blank normalize to default (empty)
    item2 = normalize_shortcut_item({"name": "y", "triggers": ["beta"], "replacements": ["B"],
                                     "refinement_key": "-None-"})
    assert item2["refinement_key"] == ""


def test_apply_prompt_shortcuts_surfaces_refinement_key():
    db = _db({"a": (["alpha"], ["the alpha thing"], "keyA")})
    expanded, applied = apply_prompt_shortcuts("an alpha here", seed=1, shortcut_db=db)
    assert "the alpha thing" in expanded
    assert applied and applied[0]["refinement_key"] == "keyA"


def test_expand_with_map_tags_pieces(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    expanded, pieces = conditioning._expand_with_map("x alpha y beta z")
    keyed = {pc["refinement_key"] for pc in pieces if pc.get("is_shortcut")}
    assert keyed == {"keyA", "keyB"}


def test_studio_scene_refinement_keys_canonical(monkeypatch):
    """Attribution is read off the canonical split — one set per scene, no scene_count, no
    divergence fallback (count reconciliation now lives in the generation guard, not here)."""
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    studio = conditioning.FunPackVideoRefinerV2()
    raw = "wide shot cut alpha here cut beta here"
    assert studio._v2_scene_refinement_keys(raw) == [{"keyA"}, {"keyB"}]


def test_studio_scene_refinement_keys_no_bindings(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    # no keyed shortcuts -> every scene is the default key (all empty sets)
    out = studio._v2_scene_refinement_keys("alpha scene 2 here")
    assert out and all(s == set() for s in out)


def test_resolver_matches_studio(monkeypatch):
    """The Movie Editor preview resolver and the Studio generation path are one function."""
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    studio = conditioning.FunPackVideoRefinerV2()
    raw = "wide shot cut alpha here cut beta here"
    assert conditioning.resolve_scene_refinement_keys(raw) == studio._v2_scene_refinement_keys(raw)


def test_reset_prompt_keys_union_excludes_primary(monkeypatch):
    """Session reset wipes the project key (caller) PLUS every non-default key in the prompt."""
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB"),
              "c": (["gamma"], ["CCC"], "keyC")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    reset_calls, saved = [], []
    monkeypatch.setattr(studio, "_v2_load_state",
                        lambda key, reset_session=False: (reset_calls.append((key, reset_session)) or ({}, "fresh")))
    monkeypatch.setattr(studio, "_v2_save_state", lambda data, key: saved.append(key))
    # keyA fires in positive, keyC fires in intent; primary "default" is excluded.
    done = studio._v2_reset_prompt_keys("default", "alpha shot", "gamma intent")
    assert done == ["keyA", "keyC"]
    assert all(rs is True for _, rs in reset_calls)
    assert sorted(k for k, _ in reset_calls) == ["keyA", "keyC"]
    assert sorted(saved) == ["keyA", "keyC"]


def test_reset_prompt_keys_skips_when_primary_only(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "keyA")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    monkeypatch.setattr(studio, "_v2_load_state", lambda *a, **k: ({}, "fresh"))
    monkeypatch.setattr(studio, "_v2_save_state", lambda *a, **k: None)
    # primary key is keyA itself -> nothing extra to reset
    assert studio._v2_reset_prompt_keys("keyA", "alpha shot") == []
    # no keyed shortcuts at all -> nothing extra
    assert studio._v2_reset_prompt_keys("default", "plain prompt") == []


def test_split_scenes_canonical_model(monkeypatch):
    """The one canonical split reports per-scene raw + expanded text + keys; anchor keys fold into
    every scene; only real transition triggers (here 'cut') split; the reset pool is the union."""
    db = _db({"anc": (["zeta"], ["ZZZ"], "anchorKey"),
              "a": (["alpha"], ["AAA"], "keyA"),
              "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    out = conditioning.split_scenes("zeta intro cut alpha here cut beta here")
    scenes = out["scenes"]
    assert len(scenes) == 2
    assert scenes[0]["keys"] == {"anchorKey", "keyA"}
    assert scenes[1]["keys"] == {"anchorKey", "keyB"}
    assert "alpha" in scenes[0]["raw"]       # verbatim text preserved
    assert "AAA" in scenes[0]["expanded"]    # expanded text available for generation
    assert conditioning.refinement_key_pool_for(
        "zeta intro cut alpha here cut beta here") == {"anchorKey", "keyA", "keyB"}


def test_generic_scene_label_splits_but_is_never_encoded(monkeypatch):
    """The generic `scene <N>` label is a pure split DELIMITER (the Movie Editor injects it to
    force a boundary for a scene with no leading transition). It MUST cut, but MUST NOT appear in
    any scene's encoded/verbatim text — otherwise 'scene 1', 'scene 2', … pollute the conditioning
    and the first scene reads as a label-only (anchor-only) scene. Regression for that report."""
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: _db({}))
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = conditioning.split_scenes(
        "scene 1 a red car drives down a street scene 2 the car stops at a cafe scene 3 a woman exits")
    scenes = out["scenes"]
    assert len(scenes) == 3
    assert [s["expanded"] for s in scenes] == [
        "a red car drives down a street", "the car stops at a cafe", "a woman exits"]
    # the literal markers must be gone from BOTH expanded (generation) and raw (editing/attribution)
    for s in scenes:
        assert "scene 1" not in s["expanded"].lower()
        assert "scene 2" not in s["expanded"].lower()
        assert "scene 3" not in s["expanded"].lower()
        assert "scene 1" not in s["raw"].lower()
    # a REAL custom trigger still keeps its configured placement (only generic labels go silent):
    # 'cut' (placement start) leads — and stays in — the new scene's text.
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    real = conditioning.split_scenes("a dog runs cut a cat sleeps")
    assert real["anchor_expanded"] == "a dog runs"
    assert len(real["scenes"]) == 1
    assert "cut" in real["scenes"][0]["expanded"]


def test_split_scenes_from_segments_uses_explicit_boundaries(monkeypatch):
    """The editor hands Studio its OWN scene list, so boundaries never need an in-text delimiter.
    One segment = exactly one scene; anchor keys fold into every scene; no 'scene N' anywhere."""
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: _db({}))
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = conditioning.split_scenes_from_segments(
        "", ["a red car drives down a street", "the car stops at a cafe", "a woman exits"])
    assert [s["expanded"] for s in out["scenes"]] == [
        "a red car drives down a street", "the car stops at a cafe", "a woman exits"]
    for s in out["scenes"]:
        assert "scene 1" not in s["expanded"].lower() and "scene 2" not in s["expanded"].lower()


def test_split_scenes_from_segments_preserves_empty_scene(monkeypatch):
    """A text-less i2v scene must KEEP its slot (count = editor scene count), not be dropped —
    otherwise per-scene anchors/seeds shift by one. It becomes an empty (anchor-only) scene."""
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: _db({}))
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = conditioning.split_scenes_from_segments("", ["", "a red car", "the cafe"])
    assert len(out["scenes"]) == 3
    assert out["scenes"][0]["expanded"] == ""
    assert [s["expanded"] for s in out["scenes"][1:]] == ["a red car", "the cafe"]


def test_split_scenes_from_segments_folds_anchor_and_keys(monkeypatch):
    """Anchor text + anchor keys fold into every scene, exactly like split_scenes()."""
    db = _db({"anc": (["zeta"], ["ZZZ"], "anchorKey"), "a": (["alpha"], ["AAA"], "keyA")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = conditioning.split_scenes_from_segments("zeta", ["alpha here", "plain scene"])
    assert out["anchor_expanded"] == "ZZZ"
    assert out["scenes"][0]["keys"] == {"anchorKey", "keyA"}
    assert out["scenes"][1]["keys"] == {"anchorKey"}
    assert "AAA" in out["scenes"][0]["expanded"]


def test_split_scenes_single_scene(monkeypatch):
    db = _db({"hero": (["hero"], ["a knight"], "mykey")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = conditioning.split_scenes("hero stands in a field")
    assert len(out["scenes"]) == 1
    assert out["scenes"][0]["keys"] == {"mykey"}
    assert conditioning.refinement_key_pool_for("hero stands in a field") == {"mykey"}


def test_split_scenes_ignores_triggers_inside_expansions(monkeypatch):
    """Regression: a transition word that only appears INSIDE a shortcut's expansion must NOT
    create a phantom cut (the 'Scene cut.' / 'scene 2' trap that produced 4 chunks instead of 2)."""
    db = _db({"qcut": (["qcut"], ["The video rapidly cuts to the next view."], ""),
              "cut": (["cut"], ["Scene cut."], ""),          # expansion contains the word 'cut'
              "man": (["man"], ["a man, scene 2 framing, stands"], ""),  # expansion contains 'scene 2'
              "a1": (["act1"], ["a backflip"], "keyA"),
              "a3": (["act3"], ["sitting"], "cs")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"qcut": {"placement": "start", "visual_effect": "none"},
                                 "cut": {"placement": "silent", "visual_effect": "none"}})
    out = conditioning.split_scenes("char a woman man qcut act1 cut act3")
    assert len(out["scenes"]) == 2                # exactly two — no phantom splits
    assert out["scenes"][0]["keys"] == {"keyA"}
    assert out["scenes"][1]["keys"] == {"cs"}
    assert "Scene cut." not in out["scenes"][1]["expanded"]  # silent trigger dropped


def test_split_scenes_literal_scene_label_still_splits(monkeypatch):
    """The generic 'scene N' label MUST still split when it's LITERAL text — the Movie Editor
    injects it into the generation prompt to force a split for a unit with no leading trigger
    (build_combined_prompt for_generation). But the SAME label inside a shortcut expansion must NOT
    split (that was the user's phantom 'scene 2'). One rule: literal cuts, expansion-content doesn't."""
    db = _db({"qcut": (["qcut"], ["rapid cut"], ""),
              "tth": (["tth"], ["a backflip"], ""),
              "cof": (["cof"], ["sitting, scene 9 of many"], "cs")})  # 'scene 9' INSIDE expansion
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"qcut": {"placement": "start", "visual_effect": "none"}})
    # explicit qcut leads scene 0; the editor-injected literal 'scene 2' leads scene 1
    out = conditioning.split_scenes("CHAR a woman man qcut tth scene 2 cof")
    assert len(out["scenes"]) == 2
    assert out["scenes"][0]["keys"] == set()      # tth has no key
    assert out["scenes"][1]["keys"] == {"cs"}     # cof, despite 'scene 9' in its expansion
    assert out["anchor"] == "CHAR a woman man"


def test_single_scene_with_custom_key_is_detected(monkeypatch):
    """Regression: a custom key fired in a single-logical-scene prompt MUST be detected (was
    silently dropped to empty when the old code compared scene counts and fell back)."""
    db = _db({"hero": (["hero"], ["a tall knight"], "mykey")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    assert conditioning.resolve_scene_refinement_keys("hero stands in a field") == [{"mykey"}]
    # two custom keys in one scene -> both detected
    db2 = _db({"hero": (["hero"], ["knight"], "mykey"), "vil": (["villain"], ["masked"], "badguy")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db2)
    assert conditioning.resolve_scene_refinement_keys("hero meets villain") == [{"mykey", "badguy"}]


def test_resolve_reads_keys_per_scene(monkeypatch):
    """Multi-scene: each scene gets exactly the keys whose shortcuts fired in it (read per scene,
    no scene-count comparison)."""
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    out = conditioning.resolve_scene_refinement_keys("wide cut alpha here cut beta here")
    assert out == [{"keyA"}, {"keyB"}]


def test_me_scene_ratings_custom_key_skips_default(monkeypatch):
    """Training law: a scene owned by a custom key trains ONLY that key — the project default key
    is left untouched. Default learns solely from scenes with no custom key."""
    studio = conditioning.FunPackVideoRefinerV2()
    calls = {"default_state": 0, "default_vf": 0, "custom": []}
    monkeypatch.setattr(studio, "_v2_build_scene_learning_run",
                        lambda run, idx, *a, **k: {"conditioning": {"x": 1}, "scene_index": idx})
    monkeypatch.setattr(studio, "_v2_axis_feedback",
                        lambda *a, **k: {"missing_axes": [], "satisfied_axes": []})
    monkeypatch.setattr(studio, "_v2_learn_scene_into_state",
                        lambda *a, **k: (calls.__setitem__("default_state", calls["default_state"] + 1), "seed: ok")[1])
    monkeypatch.setattr(studio, "_v2_train_value_function",
                        lambda *a, **k: (calls.__setitem__("default_vf", calls["default_vf"] + 1), 1)[1])
    monkeypatch.setattr(studio, "_v2_learn_scene_into_keys",
                        lambda keys, *a, **k: (calls["custom"].append(list(keys)), list(keys))[1])
    monkeypatch.setattr(studio, "_v2_learn_absolute", lambda *a, **k: None)
    prev = {"scene_count": 2, "scene_texts": ["s1", "s2"],
            "scene_refinement_keys": [["mykey"], []]}  # scene1 custom, scene2 default
    ratings = [{"index": 0, "rating": "Perfect"}, {"index": 1, "rating": "Perfect"}]
    studio._v2_apply_movie_editor_scene_ratings({}, prev, ratings, 1, refinement_key="default")
    assert calls["custom"] == [["mykey"]]   # scene 1 trained its custom key
    assert calls["default_state"] == 1      # default learned ONLY scene 2
    assert calls["default_vf"] == 1         # default VF trained ONLY scene 2


def test_wrong_ratings_skip_value_function():
    """Wrong-* are prompt-repair signals, not quality rewards: they must NOT train the value
    function (a 0.0/low reward on a visually-good gen poisons the learned quality landscape).
    Missing-*/Perfect/Awful are real quality signals and MUST still train it."""
    profiles = conditioning.V2_RATING_PROFILES
    wrong = [k for k in profiles if k.startswith("Wrong")]
    assert wrong, "expected Wrong-* ratings to exist"
    for label in wrong:
        assert profiles[label].get("skip_value_function") is True, f"{label} should skip VF"
    for label in ("Perfect", "Nailed it", "Missing details", "Missing quality", "Awful"):
        assert not profiles[label].get("skip_value_function"), f"{label} must still train VF"


def test_learn_absolute_skips_repair_ratings(monkeypatch):
    """The keyless Absolute taste store reads reward as pure quality, so a Wrong-* repair rating
    must be a no-op there (otherwise a good gen gets stored as global bad-taste)."""
    studio = conditioning.FunPackVideoRefinerV2()
    touched = []
    monkeypatch.setattr(studio, "_v2_load_absolute_global",
                        lambda: touched.append(True) or ({"global": {}}, {}))
    run = {"conditioning": {"shape": [1, 1, 4], "data": [0.0, 0.0, 0.0, 0.0]}}
    # repair rating -> early return, never loads/writes the absolute store
    studio._v2_learn_absolute(run, conditioning.normalize_refiner_v2_rating("Wrong appearance"))
    assert touched == []
    # a real quality rating proceeds (loads the store)
    studio._v2_learn_absolute(run, conditioning.normalize_refiner_v2_rating("Perfect"))
    assert touched == [True]


def test_bridge_scene_refinement_keys_preview(monkeypatch):
    """Preview payload: explicit keys vs project-default fallback per scene."""
    from movie_editor.backend import bridge
    db = _db({"a": (["alpha"], ["AAA"], "keyA")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers",
                        lambda: {"cut": {"placement": "start", "visual_effect": "none"}})
    out = bridge.scene_refinement_keys("wide shot cut alpha here cut plain", "myproj")
    assert out[0] == {"keys": ["keyA"], "uses_default": False, "default_key": "myproj"}
    assert out[1] == {"keys": ["myproj"], "uses_default": True, "default_key": "myproj"}
