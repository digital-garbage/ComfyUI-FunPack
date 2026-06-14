"""Multi-refinement-key support: shortcut->key binding, per-scene attribution, fallback."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRoutes:
    def get(self, _p):
        return lambda f: f

    def post(self, _p):
        return lambda f: f

    def delete(self, _p):
        return lambda f: f


sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir="", get_user_directory=lambda: ""))
sys.modules.setdefault(
    "server",
    types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=_FakeRoutes()))),
)

import templates
import conditioning
from templates import apply_prompt_shortcuts, normalize_shortcut_item


def _db(shortcuts):
    """Build a normalized shortcut DB from {name: (triggers, replacements, key)}."""
    out = {"version": 1, "source": "test", "shortcuts": {}}
    for name, (triggers, reps, key) in shortcuts.items():
        item = normalize_shortcut_item({
            "name": name, "triggers": triggers, "replacements": reps, "refinement_key": key,
        })
        out["shortcuts"][templates.shortcut_key(name)] = item
    return out


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


def test_prompt_scene_shortcut_keys_per_scene(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    # generic "scene N" labels split: chunk0=anchor (no key), chunk1=alpha->A, chunk2=beta->B
    n, scene_sets, all_keys = conditioning.prompt_scene_shortcut_keys(
        "wide shot scene 1 alpha here scene 2 beta here"
    )
    assert n == 2
    assert scene_sets[0] == {"keyA"}
    assert scene_sets[1] == {"keyB"}
    assert all_keys == {"keyA", "keyB"}


def test_prompt_scene_shortcut_keys_anchor_in_every_scene(monkeypatch):
    # anchor-bound key participates in every scene; per-scene keys add on top.
    db = _db({"anc": (["zeta"], ["ZZZ"], "anchorKey"),
              "a": (["alpha"], ["AAA"], "keyA"),
              "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    n, scene_sets, _ = conditioning.prompt_scene_shortcut_keys(
        "zeta intro scene 1 alpha here scene 2 beta here"
    )
    assert n == 2
    assert scene_sets[0] == {"anchorKey", "keyA"}
    assert scene_sets[1] == {"anchorKey", "keyB"}


def test_prompt_scene_shortcut_keys_no_keys(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "")})  # default key only
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    n, scene_sets, all_keys = conditioning.prompt_scene_shortcut_keys("alpha scene 2 stuff")
    assert n == 0 and scene_sets == [] and all_keys == set()


def test_studio_scene_refinement_keys_fallback(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    raw = "wide shot scene 1 alpha here scene 2 beta here"  # attribution finds 2 scenes
    # scene_count matches -> precise per-scene
    assert studio._v2_scene_refinement_keys(raw, 2) == [{"keyA"}, {"keyB"}]
    # scene_count diverges (advisor rewrote / stacking) -> safe union for every scene
    out = studio._v2_scene_refinement_keys(raw, 3)
    assert out == [{"keyA", "keyB"}, {"keyA", "keyB"}, {"keyA", "keyB"}]


def test_studio_scene_refinement_keys_no_bindings(monkeypatch):
    db = _db({"a": (["alpha"], ["AAA"], "")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    assert studio._v2_scene_refinement_keys("alpha scene 2 here", 2) == [set(), set()]


def test_resolver_matches_studio(monkeypatch):
    """The Movie Editor preview resolver and the Studio generation path are one function."""
    db = _db({"a": (["alpha"], ["AAA"], "keyA"), "b": (["beta"], ["BBB"], "keyB")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    studio = conditioning.FunPackVideoRefinerV2()
    raw = "wide shot scene 1 alpha here scene 2 beta here"
    for n in (2, 3):
        assert conditioning.resolve_scene_refinement_keys(raw, n) == studio._v2_scene_refinement_keys(raw, n)


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


def test_bridge_scene_refinement_keys_preview(monkeypatch):
    """Preview payload: explicit keys vs project-default fallback per scene."""
    from movie_editor.backend import bridge
    db = _db({"a": (["alpha"], ["AAA"], "keyA")})
    monkeypatch.setattr(templates, "load_shortcut_db", lambda: db)
    monkeypatch.setattr(templates, "load_custom_transition_triggers", lambda: {})
    out = bridge.scene_refinement_keys("wide shot scene 1 alpha here scene 2 plain", 2, "myproj")
    assert out[0] == {"keys": ["keyA"], "uses_default": False, "default_key": "myproj"}
    assert out[1] == {"keys": ["myproj"], "uses_default": True, "default_key": "myproj"}
