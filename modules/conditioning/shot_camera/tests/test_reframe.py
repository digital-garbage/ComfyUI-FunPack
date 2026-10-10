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
    assert "framing" in tail.split(".")[0], "aimed at what the next action is about"
    assert "reframes: shot 1" in said
    assert chose["arms"] == ["reframe:yes"] and chose["views"], "learned like the other camera choices"


def test_the_view_never_repeats_and_respects_what_the_part_shows():
    text = f"[Shot 1] {DANCE} {DANCE} {DANCE} {DANCE}"
    out, info = engine.add_reframes(text, seed="s", chance=1, pieces=[DANCE])
    views = [a["view"] for a in info["added"]]
    assert len(views) == 3 and all(a != b for a, b in zip(views, views[1:]))
    assert all(v != "View from behind" for v in views), "a smiling face is not shown from behind"


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
