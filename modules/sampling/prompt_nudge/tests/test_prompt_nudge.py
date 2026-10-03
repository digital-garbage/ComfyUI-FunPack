"""Taste prompt nudge: words shift along the learned direction, late, with no extra pass."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.system.taste import store
    monkeypatch.setattr(store, "ROOT", tmp_path / "taste")


def _to(i, steps=4):
    s = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": s, "sigmas": s[i:i + 1]}


def test_words_move_along_the_direction_by_a_share_of_their_own_size():
    from modules.sampling.prompt_nudge import nudged
    c = torch.ones(1, 3, 4)
    d = torch.tensor([1.0, 0, 0, 0])
    words = torch.tensor([True, True, False])
    out = nudged(c, words, d, 0.1)
    assert torch.allclose(out[0, 0, 0], c[0, 0, 0] + 0.1)      # the strength is the row shift itself, as in v4
    assert torch.equal(out[0, 2], c[0, 2])                  # a reference image's row is left alone
    assert torch.equal(out[0, :, 1:], c[0, :, 1:])


def _load(tiny, **v):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, "prompt_nudge": {"enabled": True, **v}}).result
    wrap = [w for ws in patched.wrappers[WrappersMP.APPLY_MODEL].values() for w in ws][-1]
    for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values():
        for o in ws:
            o(lambda: None)
    return wrap, status


def _teach(dim):
    from modules.system.taste import store
    liked = torch.zeros(dim)
    liked[0] = 1.0
    for i in range(6):
        store.capture("fox", "prompt_taste", {"pooled": liked if i % 2 else -liked}, prompt_id=f"t{i}")
        store.rate(f"t{i}", "liked" if i % 2 else "disliked")


def test_late_steps_nudge_the_words_in_one_pass_and_early_ones_do_not(tiny_h3):
    dim = 8
    _teach(dim)
    wrap, _ = _load(tiny_h3, strength=0.1)
    c = torch.ones(1, 4, dim)
    seen = []

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        seen.append(c_crossattn.clone())
        return x

    x = torch.zeros(1, 1, 4)
    ts = torch.linspace(1.0, 0.0, 5)
    for i in (0, 3):
        wrap(executor, x, ts[i:i + 1], None, c, None, _to(i))
    assert len(seen) == 2                                    # no extra passes
    assert torch.equal(seen[0], c)                           # first half: untouched
    assert (seen[1][0, :, 0] > c[0, :, 0]).all() and torch.equal(seen[1][..., 1:], c[..., 1:])


def test_a_direction_from_another_text_width_is_said_not_silently_dropped(tiny_h3):
    from core import log
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.1)
    log.new_run()
    c = torch.ones(1, 4, 6)
    seen = []
    wrap(lambda x, t, *a, **k: seen.append(a[1]) or x, torch.zeros(1, 1, 4), torch.linspace(1.0, 0.0, 5)[3:4],
         None, c, None, _to(3))
    assert torch.equal(seen[0], c)
    assert any("different model" in e["message"] for e in log.history())


def test_off_installs_nothing(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"prompt_nudge": {"enabled": False}}).result
    assert not patched.wrappers.get(WrappersMP.APPLY_MODEL)


def test_the_sounds_text_channels_are_left_alone():
    from modules.sampling.prompt_nudge import nudged, picture_width

    class Dm:
        cross_attention_dim, audio_cross_attention_dim = 6, 2

    class P:
        model = type("M", (), {"diffusion_model": Dm()})()

    c = torch.ones(1, 3, 8)
    width = picture_width(P(), c)
    assert width == 6 and picture_width(P(), torch.ones(1, 3, 5)) is None
    out = nudged(c, torch.ones(3, dtype=torch.bool), torch.ones(8) / 8 ** 0.5, 1.0, width)
    assert (out[..., :6] > 1).all() and torch.equal(out[..., 6:], c[..., 6:])


def test_a_split_negative_call_does_not_teach_the_negative_prompt(tiny_h3):
    from modules.sampling.score_slider import cond_row
    c = torch.ones(2, 3, 4)
    assert cond_row(c, {"cond_or_uncond": [1, 0]}) == 1
    assert cond_row(c[:1], {"cond_or_uncond": [1]}) is None
    assert cond_row(c, {}) == 0


def test_a_run_that_never_nudged_says_so(tiny_h3):
    from core import log
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.1)
    log.new_run()
    c = torch.ones(1, 4, 8)
    for i in range(2):                                       # a two-step schedule: the late gate never opens
        s = torch.linspace(1.0, 0.0, 3)
        wrap(lambda x, t, *a, **k: x, torch.zeros(1, 1, 4), s[i:i + 1], None, c, None,
             {"sample_sigmas": s, "sigmas": s[i:i + 1]})
    assert any("Inactive" in e["message"] and "too short" in e["message"] for e in log.history())


def test_a_nudge_that_bf16_rounds_away_is_said(tiny_h3):
    from core import log
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.005)
    log.new_run()
    c = (torch.randn(1, 4, 8) * 50).to(torch.bfloat16)
    s = torch.linspace(1.0, 0.0, 5)
    wrap(lambda x, t, *a, **k: x, torch.zeros(1, 1, 4), s[3:4], None, c, None, _to(3))
    assert any("rounding" in e["message"] for e in log.history())


def test_a_run_that_starts_with_the_negative_prompt_does_not_learn_from_it(tiny_h3):
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.1, similar=True)
    c = torch.ones(1, 4, 8)
    s = torch.linspace(1.0, 0.0, 5)
    seen = []
    to = {**_to(3), "cond_or_uncond": [1]}
    wrap(lambda x, t, *a, **k: seen.append(a[1]) or x, torch.zeros(1, 1, 4), s[3:4], None, c, None, to)
    assert torch.equal(seen[0], c)            # no positive prompt yet: nothing learned, nothing nudged


def test_strength_zero_only_learns_and_does_not_cry_wolf(tiny_h3):
    from core import log
    _teach(8)
    wrap, status = _load(tiny_h3, strength=0.0)
    log.new_run()
    before = len(log.history())
    c = torch.ones(1, 4, 8)
    ts = torch.linspace(1.0, 0.0, 5)
    seen = []
    for i in range(4):
        wrap(lambda x, t, *a, **k: seen.append(a[1]) or x, torch.zeros(1, 1, 4), ts[i:i + 1], None, c, None, _to(i))
    assert all(torch.equal(s, c) for s in seen) and "learning only" in str(status)
    assert not [e["message"] for e in log.history()[before:] if "nothing was nudged" in e["message"]]


def test_a_probe_does_not_teach_the_taste_key(tiny_h3):
    from core import dit_hooks
    from modules.system.taste import store
    _teach(8)
    wrap, _ = _load(tiny_h3, strength=0.1)
    c = torch.ones(1, 4, 8)
    ts = torch.linspace(1.0, 0.0, 5)
    wrap(lambda x, t, *a, **k: x, torch.zeros(1, 1, 4), ts[3:4], None, c, None, {**_to(3), dit_hooks.PROBE: True})
    pending = store._dir("fox") / "prompt_taste.pending.pt"
    assert not pending.exists()                              # nothing was captured from the probe
