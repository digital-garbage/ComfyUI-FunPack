"""Uneven shot lengths: a shot's share of the video is now and then changed, busier text tending to run longer."""

import re

import pytest

from .. import engine, memory, nodes

SHORT = "She waves."
MID = "She turns and smiles at the camera."
BUSY = ("She picks up the heavy red umbrella from the wet cobblestone street while the rain keeps falling "
        "on her shoulders and a car passes slowly behind her.")
TEXT = f"[Shot 1] {SHORT} {BUSY} {MID}"
PIECES = [SHORT, BUSY, MID]


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOT_CAMERA_MEMORY", str(tmp_path / "m.json"))


def _times(out):
    return [int(m.group(1)) * 60 + int(m.group(2)) for m in re.finditer(r"At (\d\d):(\d\d)\.000", out)]


def _cut(seconds=30, seed=0, uneven=0.0, text=TEXT, pieces=PIECES):
    out, info = engine.add_shot_cuts(text, seconds, seed=seed, chance=1, pieces=pieces, uneven=uneven)
    return _times(out), info


def test_at_zero_the_shots_are_equal_and_nothing_is_recorded():
    times, info = _cut(seconds=12)
    assert times == [4, 8]
    assert not [a for a in info["arms"] if a.startswith("length:")]


def test_at_one_every_shot_is_changed_and_the_times_stay_valid():
    seen = set()
    for seed in range(60):
        times, info = _cut(seconds=30, seed=seed, uneven=1.0)
        seen.add(tuple(times))
        assert len(times) == 2 and 0 < times[0] < times[1] < 30, times
        durations = [times[0], times[1] - times[0], 30 - times[1]]
        assert min(durations) >= engine.MIN_SHOT_SECONDS, (times, durations)
        assert [a for a in info["arms"] if a.startswith("length:")] == ["length:yes"] * 3
    assert len(seen) > 5, "the lengths differ from take to take"
    assert (10, 20) not in seen or len(seen) > 5


def test_busier_text_tends_to_run_longer():
    short, busy = [], []
    for seed in range(150):
        times, _ = _cut(seconds=30, seed=seed, uneven=1.0)
        short.append(times[0])
        busy.append(times[1] - times[0])
    assert sum(busy) / len(busy) > sum(short) / len(short) + 2, "the busy shot gets more screen time on average"


def test_between_zero_and_one_the_share_of_changed_shots_follows_the_slider():
    blocks = [[SHORT, ""], [BUSY, ""], [MID, ""]]
    for uneven in (0.2, 0.5, 0.8):
        yes = total = 0
        for seed in range(300):
            arms = []
            engine._screen_time(blocks, seed, uneven, arms)
            yes += arms.count("length:yes")
            total += len(arms)
        assert abs(yes / total - uneven) < 0.07, uneven


def test_the_same_seed_gives_the_same_lengths_and_a_single_shot_is_left_alone():
    assert _cut(seed=3, uneven=1.0)[0] == _cut(seed=3, uneven=1.0)[0]
    assert engine._screen_time([[SHORT, ""]], 1, 1.0, []) is None


def test_the_slider_set_in_the_node_changes_the_lengths_and_zero_does_not():
    base = {"cut_same_shot": True, "shot_cuts_chance": 1}
    equal = _times(nodes.rewrite(TEXT, base, seconds=30, pieces=PIECES)[0])
    assert equal == [10, 20]
    assert _times(nodes.rewrite(TEXT, {**base, "uneven_shots_chance": 0}, seconds=30, pieces=PIECES)[0]) == equal
    changed = {tuple(_times(nodes.rewrite(TEXT, {**base, "uneven_shots_chance": 1, "variation": v}, seconds=30, pieces=PIECES)[0])) for v in range(20)}
    assert changed != {tuple(equal)} and len(changed) > 3


def test_the_chance_follows_the_slider_at_the_ends_and_ratings_tilt_it_between():
    assert memory.length_chance(0.0) == 0.0 and memory.length_chance(1.0) == 1.0 and memory.length_chance(0.5) == 0.5
    memory.record_run("a", {"views": [], "arms": ["length:yes"]})
    memory.on_rating("a", "liked")
    assert memory.length_chance(0.5) > 0.5 and memory.length_chance(1.0) == 1.0
    memory.record_run("b", {"views": [], "arms": ["length:yes", "length:no"]})
    for k in range(10):
        memory.record_run(f"d{k}", {"views": [], "arms": ["length:yes"]})
        memory.on_rating(f"d{k}", "disliked")
    assert memory.length_chance(0.5) < 0.5 and memory.length_chance(0.0) == 0.0


def test_a_clock_time_in_the_story_does_not_count_as_a_written_cut_time():
    assert not engine.ANY_STAMP.search("She arrives at 8:30, and sits.")
    assert engine.ANY_STAMP.search("At 00:04.000, the shot cuts to her hand.")
