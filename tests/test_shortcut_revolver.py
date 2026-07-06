"""Shortcut revolver: no-repeat cycling through a shortcut's replacements.

Covers: sequential cycle order + wrap-around, random mode as a full no-repeat permutation
per cycle, peek (previews) vs commit (generation) semantics on apply_prompt_shortcuts,
fingerprint reset when the replacement set is edited, settings changes resetting state,
and off-by-default behavior staying byte-identical to the seeded random pick.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRoutes:
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
from templates import (
    apply_prompt_shortcuts,
    load_revolver,
    normalize_shortcut_item,
    revolver_next_replacement,
    save_revolver,
    set_revolver_settings,
)

REPS = ["alpha one", "alpha two", "alpha three"]


def _db():
    item = normalize_shortcut_item({
        "name": "Alpha", "triggers": ["aa"], "replacements": list(REPS), "refinement_key": "",
    })
    single = normalize_shortcut_item({
        "name": "Solo", "triggers": ["ss"], "replacements": ["only one"], "refinement_key": "",
    })
    return {"version": 1, "source": "test",
            "shortcuts": {templates.shortcut_key("Alpha"): item,
                          templates.shortcut_key("Solo"): single}}


def _use_tmp_store(monkeypatch, tmp_path):
    monkeypatch.setattr(templates, "template_store_dir", lambda: str(tmp_path))


def test_sequential_cycle_then_wraps():
    state = {}
    draws = [revolver_next_replacement(state, "k", REPS, False) for _ in range(7)]
    assert draws == REPS + REPS + [REPS[0]]


def test_random_cycle_is_full_permutation_without_repeats():
    state = {}
    for _ in range(20):  # many cycles: each must be a permutation of the full set
        cycle = [revolver_next_replacement(state, "k", REPS, True) for _ in range(len(REPS))]
        assert sorted(cycle) == sorted(REPS)


def test_replacement_edit_resets_cycle():
    state = {}
    revolver_next_replacement(state, "k", REPS, False)
    edited = REPS + ["alpha four"]
    assert revolver_next_replacement(state, "k", edited, False) == edited[0]


def test_apply_shortcuts_peek_does_not_advance(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    save_revolver({"enabled": True, "random": False, "state": {}})
    first, _ = apply_prompt_shortcuts("an aa here", seed=1, shortcut_db=_db())
    second, _ = apply_prompt_shortcuts("an aa here", seed=2, shortcut_db=_db())
    assert first == second == f"an {REPS[0]} here"


def test_apply_shortcuts_commit_advances_and_wraps(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    save_revolver({"enabled": True, "random": False, "state": {}})
    seen = [apply_prompt_shortcuts("an aa here", seed=1, shortcut_db=_db(),
                                   revolver_commit=True)[0] for _ in range(4)]
    assert seen == [f"an {r} here" for r in REPS + [REPS[0]]]


def test_multiple_firings_in_one_prompt_consume_consecutive_chambers(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    save_revolver({"enabled": True, "random": False, "state": {}})
    expanded, applied = apply_prompt_shortcuts("aa then aa", seed=1, shortcut_db=_db(),
                                               revolver_commit=True)
    assert expanded == f"{REPS[0]} then {REPS[1]}"
    assert [a["replacement"] for a in applied] == [REPS[0], REPS[1]]


def test_single_replacement_shortcut_untouched_by_revolver(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    save_revolver({"enabled": True, "random": False, "state": {}})
    expanded, _ = apply_prompt_shortcuts("a ss here", seed=1, shortcut_db=_db(),
                                         revolver_commit=True)
    assert expanded == "a only one here"
    assert load_revolver()["state"] == {}  # nothing multi-replacement fired -> no state written


def test_disabled_matches_seeded_random(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    a, _ = apply_prompt_shortcuts("an aa here", seed=7, shortcut_db=_db())
    b, _ = apply_prompt_shortcuts("an aa here", seed=7, shortcut_db=_db(), revolver_commit=True)
    assert a == b  # same seed -> same pick; no revolver file involved
    assert not (tmp_path / templates.REVOLVER_STORE_FILENAME).exists()


def test_settings_change_resets_state(monkeypatch, tmp_path):
    _use_tmp_store(monkeypatch, tmp_path)
    save_revolver({"enabled": True, "random": False,
                   "state": {"x": {"fp": "abc", "queue": [2]}}})
    settings = set_revolver_settings(random_order=True)
    assert settings == {"enabled": True, "random": True}
    assert load_revolver()["state"] == {}
    # No-op update keeps state.
    save_revolver({"enabled": True, "random": True,
                   "state": {"x": {"fp": "abc", "queue": [2]}}})
    set_revolver_settings(enabled=True, random_order=True)
    assert load_revolver()["state"] != {}
