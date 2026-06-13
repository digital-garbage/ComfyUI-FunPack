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
