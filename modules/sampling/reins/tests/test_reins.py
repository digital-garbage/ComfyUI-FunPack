"""Taste steering on a real H3 forward, learning through a real taste key."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")
    monkeypatch.setattr(store, "current_prompt_id", lambda: "run-1")


def _load(tiny, key="fox", **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"reins": {"enabled": True, **values}}
    if key:
        settings["taste"] = {"key": key}
    return FunPackLoadModifiers.execute(tiny.patcher, settings).result


def _teach(block, liked, disliked):
    from modules.system.taste import store
    for i, (vec, rating) in enumerate([(liked, "liked"), (disliked, "disliked")] * 2):
        store.capture("fox", "reins", {block: vec}, prompt_id=f"t{i}")
        store.rate(f"t{i}", rating)


def test_no_taste_key_means_off(tiny_h3):
    _p, status = _load(tiny_h3, key=None)
    assert "reins" not in status.splitlines()[0]


def test_the_last_step_is_captured_at_every_block_and_a_rating_records_it(tiny_h3):
    from modules.system.taste import store
    patched, status = _load(tiny_h3, blocks="3")
    assert "learning, needs 2 liked + 2 disliked at 3 (0/0)" in status
    sched = torch.tensor([1.0, 0.5, 0.0])
    tiny_h3.sample(patched, sigma=1.0, sigmas=sched)        # not the last step
    assert not (store.ROOT / "fox" / "reins.pending.pt").exists()
    tiny_h3.sample(patched, sigma=0.5, sigmas=sched)
    assert store.rate("run-0", "liked")["why"]               # some other clip
    assert store.rate("run-1", "liked")["recorded"] == ["reins"]
    row = store.load("fox", "reins")["rows"][0]["rows"]
    hidden = tiny_h3.patcher.model.diffusion_model.blocks[0].attn.out_proj.weight.shape[0]
    assert sorted(row) == ["0", "1", "2", "3"] and row["3"].shape == (hidden,)


def test_a_learned_direction_moves_the_picture_and_never_the_sound(tiny_h3):
    hidden = tiny_h3.patcher.model.diffusion_model.blocks[0].attn.out_proj.weight.shape[0]
    liked = torch.zeros(hidden)
    liked[0] = 1.0
    _teach(3, liked, -liked)
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, blocks="3", strength=0.5)
    assert "steering 3 (2/2) at 0.5" in status
    video, audio = tiny_h3.sample(patched)
    assert not torch.allclose(video, base[0])
    # Block 3 is the last: only its picture rows were pushed, so sound is exact.
    assert torch.equal(audio, base[1])

    quiet, status = _load(tiny_h3, blocks="3", strength=0.0)
    assert "learning only" in status
    assert torch.equal(tiny_h3.sample(quiet)[0], base[0])
