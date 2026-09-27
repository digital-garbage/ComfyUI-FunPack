"""Taste slider: its direction, and three passes combined on the picture only."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _to(index, steps=4):
    sched = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": sched, "sigmas": sched[index:index + 1]}


def _rows(vectors):
    return [{"reward": w, "rows": {"pooled": v}} for v, w in vectors]


def test_the_direction_is_liked_minus_the_average_and_needs_three_liked():
    from modules.sampling.score_slider import direction
    e0, e1 = torch.eye(4)[0], torch.eye(4)[1]
    few = _rows([(e0, 1.0), (e0, 1.0), (e1, -1.0)])
    assert direction(few)[0] is None
    d, _ = direction(_rows([(e0, 1.0)] * 3 + [(e1, -1.0)] * 3))
    assert d[0] > 0 and d[1] < 0


def test_similar_prompts_pick_their_own_direction():
    from modules.sampling.score_slider import direction
    a, b, n = torch.tensor([1., 0, 0, 0]), torch.tensor([0., 1, 0, 0]), torch.tensor([0., 0, 1, 0])
    rows = _rows([(a + 0.1 * n, 1.0), (a + 0.1 * n, 1.0), (b, 1.0), (b, 1.0), (b, 1.0), (n, -1.0)])
    near_a, how = direction(rows, current=a, similar=True)
    everything, _ = direction(rows)
    assert "similar" in how
    assert near_a[0] > everything[0]


def test_late_steps_combine_three_passes_on_the_picture_words_only(tiny_h3):
    from comfy.nested_tensor import NestedTensor
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    from modules.system.taste import store

    dim = 8
    liked = torch.zeros(dim)
    liked[0] = 1.0
    for i in range(6):
        store.capture("fox", "prompt_taste", {"pooled": liked if i % 2 else -liked}, prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked" if i % 2 else "disliked")

    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {
        "taste": {"key": "fox"}, "score_slider": {"enabled": True, "strength": 1.0}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values():
        for o in ws:
            o(lambda: None)

    audio = torch.randn(1, 3, 5)
    seen = []

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        seen.append(c_crossattn.clone())
        return NestedTensor([torch.full((1, 2, 1, 2, 2), float(c_crossattn[0, :, 0].sum())), audio])

    c = torch.ones(1, 4, dim)
    tags = torch.tensor([1, 1, 0, 1])                  # row 2: a reference image's token
    x0 = executor(None, None, c_crossattn=c)
    seen.clear()
    early = wrap(executor, x0, None, None, c, None, _to(0), minimax_payload={"text_token_tags": tags})
    assert len(seen) == 1 and torch.equal(early.tensors[0], x0.tensors[0])

    seen.clear()
    late = wrap(executor, x0, None, None, c, None, _to(3), minimax_payload={"text_token_tags": tags})
    assert len(seen) == 3
    plus = seen[1]
    assert torch.equal(plus[0, 2], c[0, 2]) and not torch.equal(plus[0, 0], c[0, 0])
    assert not torch.equal(late.tensors[0], x0.tensors[0])
    assert late.tensors[1] is audio
