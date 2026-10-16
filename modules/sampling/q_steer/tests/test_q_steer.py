"""Taste attention on a real H3 forward, and the rotation it relies on."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    return FunPackLoadModifiers.execute(
        tiny.patcher, {"taste": {"key": "fox"}, "q_steer": {"enabled": True, **values}}).result


def test_the_rotation_matches_the_models_own_and_undoes_exactly():
    from comfy_kitchen.backends.eager.rope import apply_rope_split_half1
    from comfy.ldm.minimax.model import rope_rotation_table
    from modules.models.minimax_h3.rope import query_rotation
    seq, heads, dim, rot = 6, 2, 8, 6
    table = rope_rotation_table(torch.randn(seq, rot), torch.float32)
    x = torch.randn(heads, seq, dim)
    rotate = query_rotation(table, seq)
    ours = rotate(x, slice(0, seq))
    theirs = torch.cat([apply_rope_split_half1(x[..., :rot].unsqueeze(0).transpose(1, 2),
                                               table).transpose(1, 2)[0], x[..., rot:]], -1)
    assert torch.allclose(ours, theirs, atol=1e-5)
    assert torch.allclose(rotate(ours, slice(0, seq), inverse=True), x, atol=1e-5)
    assert query_rotation(table, seq + 1) is None


def test_captures_every_block_on_the_last_step(tiny_h3):
    from modules.system.taste import store
    patched, status = _load(tiny_h3, blocks="3")
    assert "q_steer: learning at every block" in status
    tiny_h3.sample(patched)
    pending = torch.load(store.ROOT / "fox" / "q_steer.pending.pt")["rows"]
    attn = tiny_h3.patcher.model.diffusion_model.blocks[0].attn
    assert sorted(pending) == ["0", "1", "2", "3"]
    assert pending["2"].shape == (attn.heads * attn.head_dim,)


def test_a_learned_direction_moves_the_picture_and_never_the_sound(tiny_h3):
    from modules.system.taste import store
    attn = tiny_h3.patcher.model.diffusion_model.blocks[0].attn
    liked = torch.zeros(attn.heads * attn.head_dim)
    liked[::attn.head_dim] = 1.0
    for i, (vec, rating) in enumerate([(liked, "liked"), (-liked, "disliked")] * 2):
        store.capture("fox", "q_steer", {3: vec}, prompt_id=f"t{i}")
        store.rate(f"t{i}", rating)
    base = tiny_h3.run()
    patched, _status = _load(tiny_h3, blocks="3", strength=1.0)
    video, audio = tiny_h3.sample(patched)
    assert not torch.allclose(video, base[0])
    assert torch.equal(audio, base[1])                       # only picture queries moved
