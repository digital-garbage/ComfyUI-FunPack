"""Reframe within the shot: a kept shortcut boundary turns the camera in one continuous take, never a cut."""

from .. import engine, memory, nodes

DANCE = "She sways her hips from side to side, smiling at the room."
WAVE = "She raises her hand and waves at the camera with her fingers."


def test_reframe_turns_the_camera_between_shortcuts_without_a_cut(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} {WAVE}"
    out, said, chose = nodes.rewrite(text, {"reframe_same_shot": True, "reframe_chance": 1}, seconds=10, pieces=[DANCE, WAVE])
    assert out.count("[Shot ") == 1 and "cuts to" not in out, "one take: no new shot, no cut words"
    head, tail = out.split("Without a cut, the camera ")
    assert DANCE in head and WAVE in tail, "the reframe sits between the two actions, in order"
    assert "hand" in tail.split(".")[0] or "fingers" in tail.split(".")[0], "moves to what the next action is about"
    assert "reframes: shot 1 to " in said
    assert chose["arms"][0] == "reframe:yes" and chose["arms"][1] in ("reframe:keep", "reframe:turn")


def test_every_reframe_moves_to_a_new_focus_and_the_view_may_stay():
    text = f"[Shot 1] {DANCE} {DANCE} {DANCE} {DANCE}"
    out, info = engine.add_reframes(text, seed="s", chance=1, pieces=[DANCE])
    targets = [a["target"] for a in info["added"]]
    assert len(targets) == 3 and all(t for t in targets)
    assert all(a != b for a, b in zip(targets, targets[1:])), "the focus changes at every reframe"
    assert all(v != "View from behind" for v in (a["view"] for a in info["added"]) if v), "a smiling face is not turned away from"
    kept = sum(1 for seed in range(40) for a in engine.add_reframes(text, seed=seed, chance=1, pieces=[DANCE])[1]["added"] if a["view"] is None)
    assert kept > 0, "keeping the view is allowed"


def test_at_one_every_boundary_reframes_even_with_nothing_new_to_aim_at():
    plain = "She smiles warmly at the room again."
    out, info = engine.add_reframes(f"[Shot 1] {plain} {plain}", seed=1, chance=1, pieces=[plain])
    assert out.count("Without a cut") == 1 and info["added"][0]["target"] is None, "1.0 means every boundary: the camera still moves"


def test_zero_chance_adds_nothing_and_no_boundary_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} {WAVE}"
    assert engine.add_reframes(text, seed=1, chance=0, pieces=[DANCE, WAVE])[0] == text
    out, said, _ = nodes.rewrite("[Shot 1] A woman walks.", {"reframe_same_shot": True, "reframe_chance": 1}, seconds=5, pieces=[])
    assert out == "[Shot 1] A woman walks." and "no shortcut boundary inside a shot" in said


def test_a_cut_boundary_is_not_also_reframed(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} {WAVE}"
    out, _, _ = nodes.rewrite(text, {"cut_same_shot": True, "shot_cuts_chance": 1, "reframe_same_shot": True, "reframe_chance": 1},
                              seconds=10, pieces=[DANCE, WAVE])
    assert out.count("[Shot ") == 2 and "Without a cut" not in out


def test_the_reframe_is_not_content_and_does_not_block_the_opening_view():
    text = f"[Shot 1] A woman walks in. [Shot 2] {DANCE} {WAVE}"
    out, _ = engine.add_reframes(text, seed=2, chance=1, pieces=[DANCE, WAVE])
    assert engine.content_fingerprint(out) == engine.content_fingerprint(text), "same prompt, same seed next run"
    shot2 = engine.shot_texts(out)[2]
    assert "Without a cut" not in shot2
    _, added = engine.add_shot_views(out, seed=0, chance=1)
    assert [a["shot"] for a in added] == [2], "a mid-shot reframe is not the view the shot opens on"


def test_ratings_tilt_the_reframe_chance(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    assert memory.reframe_chance(0.5) == 0.5
    memory.record_run("r", {"views": [], "arms": ["reframe:yes"]})
    memory.on_rating("r", "liked")
    assert memory.reframe_chance(0.5) > 0.5
    assert memory.reframe_chance(1.0) == 1.0, "always stays always"


def test_cut_within_the_shot_at_zero_leaves_every_boundary_to_reframe(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    text = f"[Shot 1] {DANCE} {WAVE} {DANCE}"
    out, _, _ = nodes.rewrite(text, {"cut_same_shot": True, "shot_cuts_chance": 0, "reframe_same_shot": True, "reframe_chance": 1},
                              seconds=10, pieces=[DANCE, WAVE])
    assert out.count("[Shot ") == 1 and out.count("Without a cut") == 2


def test_only_one_is_always_and_only_zero_is_never(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))
    for _ in range(30):                                   # a long run of likes: the strongest possible lean
        memory.record_run("r", {"views": [], "arms": ["reframe:yes", "split:yes", "move:yes"]})
        memory.on_rating("r", "liked")
        memory.record_run("q", {"views": [], "arms": ["reframe:no", "split:no", "move:no"]})
        memory.on_rating("q", "disliked")
    for fn in (memory.reframe_chance, memory.split_chance, memory.effective_chance):
        assert fn(1.0) == 1.0 and fn(0.0) == 0.0
        assert 0.0 < fn(0.6) < 1.0 and 0.0 < fn(0.05) < 1.0, f"{fn.__name__}: in between stays occasionally"
        assert fn(0.6) > 0.6, "the likes still lean it up"
    assert 0.0 < memory.detail_chance(0.9, 100, 0) < 1.0 and memory.detail_chance(1.0, 0, 100) == 1.0
