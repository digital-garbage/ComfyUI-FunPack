"""Shot memory: the noise maths, the Thompson choice, and the wrapper on packed H3 noise."""

import math
import random
import types

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


SHAPES = [torch.Size([1, 4, 2, 8, 8]), torch.Size([1, 3, 5])]
N_VIDEO = 4 * 2 * 8 * 8


def test_the_coarse_part_is_unit_variance_and_swapping_it_keeps_the_fine_part():
    from modules.sampling.shot_memory import coarse_of, with_coarse
    torch.manual_seed(0)
    v = torch.randn(1, 4, 6, 32, 32)
    c = coarse_of(v)
    assert c.shape == (4, 8, 8) and c.std() == pytest.approx(1.0, abs=0.15)
    other = torch.randn_like(c)
    w = with_coarse(v, other)
    assert torch.allclose(coarse_of(w), other, atol=1e-4)
    fine = lambda x: x[0, :, :, :32, :32].reshape(4, 6, 8, 4, 8, 4) - x[0].reshape(4, 6, 8, 4, 8, 4).mean(dim=(1, 3, 5), keepdim=True)
    assert torch.allclose(fine(w), fine(v), atol=1e-5)


def test_one_cell_grids_are_left_alone():
    from modules.sampling.shot_memory import coarse_of
    assert coarse_of(torch.randn(1, 4, 2, 4, 4)) is None and coarse_of(torch.randn(2, 4, 2, 16, 16)) is None


def test_a_stored_cpu_shot_blends_into_noise_on_another_device():
    from modules.sampling.shot_memory import blend, fit
    device = "mps" if torch.backends.mps.is_available() else "meta"
    own = torch.randn(4, 2, 2, device=device)
    assert fit(torch.randn(4, 2, 2).half(), 2, 2, device=own.device).device == own.device
    assert blend(own, torch.randn(4, 2, 2).half(), 0.7).device == own.device


def test_the_blend_never_raises_the_layout_strength_even_for_the_same_seed():
    from modules.sampling.shot_memory import blend
    c = torch.randn(4, 8, 8)
    out = blend(c, c.clone(), 0.7)                      # rerunning the liked shot's own seed
    assert out.std(dim=(1, 2)).max() <= 1.0 + 1e-5


def _row(i, parent, reward, channels=4, amount=0.7):
    return {"id": i, "parent": parent, "amount": amount, "reward": reward,
            "coarse": torch.randn(channels, 8, 8), "cond": None}


def test_no_liked_shot_means_fresh_and_a_liked_one_can_be_reused():
    from modules.sampling.shot_memory import choose_parent
    assert choose_parent([_row(1, -1, -1.0)], None, 4)[0] is None
    rows = [_row(1, -1, 1.0)] + [_row(i, 1, 1.0) for i in range(2, 12)]   # reuse keeps winning
    hits = sum(choose_parent(rows, None, 4, random.Random(s))[0] is not None for s in range(50))
    assert hits > 40
    assert choose_parent([_row(1, -1, 1.0, channels=8)], None, 4)[0] is None   # other model's channels


def test_the_learned_amount_moves_toward_liked_reuses_and_away_from_disliked():
    from modules.sampling.shot_memory import START_AMOUNT, learned_amount
    assert learned_amount([_row(1, -1, 1.0)]) == START_AMOUNT        # a fresh shot says nothing about amount
    assert learned_amount([_row(2, 1, 1.0, amount=0.9)]) > START_AMOUNT
    assert learned_amount([_row(2, 1, -1.0, amount=0.9)]) < START_AMOUNT


class Guider:
    def __init__(self):
        self.inner_model = types.SimpleNamespace(latent_shapes=SHAPES)
        self.conds = {"positive": [{"cross_attn": torch.randn(1, 5, 8)}]}


class Executor:
    def __init__(self):
        self.noise = None

    def __call__(self, guider, sigmas, extra_args, callback, noise, *rest):
        self.noise = noise
        return noise


def _load(tiny, **values):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, _ = FunPackLoadModifiers.execute(tiny.patcher, {
        "taste": {"key": "fox"}, "shot_memory": {"enabled": True, **values}}).result
    sample = [w for ws in patched.wrappers[WrappersMP.SAMPLER_SAMPLE].values() for w in ws][-1]
    outer = [w for ws in patched.wrappers[WrappersMP.OUTER_SAMPLE].values() for w in ws]
    return sample, outer


def _go(sample, outer, noise, latent=None, guider=None):
    ex = Executor()
    latent = torch.zeros_like(noise) if latent is None else latent

    def call():
        return sample(ex, guider or Guider(), torch.tensor([1.0, 0.0]), {"model_options": {}}, None,
                      noise, latent, None, True)

    run = call
    for o in reversed(outer):
        run = (lambda w, inner: (lambda: w(lambda: inner())))(o, run)
    run()
    return ex.noise


def _noise():
    from conftest import packed_av
    torch.manual_seed(4)
    return packed_av(torch.randn(1, 4, 2, 8, 8), torch.randn(1, 3, 5))[0]


def test_a_first_run_is_fresh_noise_is_untouched_and_the_shot_waits_for_a_rating(tiny_h3):
    from modules.system.taste import store
    sample, outer = _load(tiny_h3)
    noise = _noise()
    out = _go(sample, outer, noise)
    assert torch.equal(out, noise)
    assert store.rate("run-1", "liked")["recorded"] == ["shot_memory"]
    row = store.load("fox", "shot_memory")["rows"][0]["rows"]
    assert row["coarse"].shape == (4, 2, 2) and int(row["parent"]) == -1


def test_once_a_shot_is_liked_the_next_run_starts_from_its_layout_and_keeps_sound_and_fine_noise(tiny_h3):
    from modules.sampling.shot_memory import coarse_of
    from modules.system.taste import store
    from conftest import unpacked
    sample, outer = _load(tiny_h3, mode="manual", amount=0.95)
    first = _noise()
    _go(sample, outer, first)
    store.rate("run-1", "liked")
    liked = store.load("fox", "shot_memory")["rows"][0]["rows"]["coarse"].float()
    for _ in range(40):                                  # Thompson may draw "fresh"; keep asking
        from conftest import packed_av
        second = packed_av(torch.randn(1, 4, 2, 8, 8), torch.randn(1, 3, 5))[0]
        out = _go(sample, outer, second)
        if not torch.equal(out, second):
            break
    else:
        pytest.fail("a liked shot was never reused")
    v_out, a_out = unpacked(out, SHAPES)
    v_in, a_in = unpacked(second, SHAPES)
    assert torch.equal(a_out, a_in)                      # sound noise untouched
    cos = lambda a, b: torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    assert cos(coarse_of(v_out), liked) > cos(coarse_of(v_in), liked)


def test_a_non_empty_latent_is_left_alone_and_says_so_and_records_nothing(tiny_h3):
    from core import log
    from modules.system.taste import store
    sample, outer = _load(tiny_h3)
    noise = _noise()
    log.new_run()
    out = _go(sample, outer, noise, latent=torch.ones_like(noise))
    assert torch.equal(out, noise)
    assert any("not empty" in e["message"] for e in log.history())
    assert store.rate("run-1", "liked")["why"]            # nothing was captured to rate


def test_no_taste_key_is_off_and_says_so(tiny_h3):
    from comfy.patcher_extension import WrappersMP
    from core import log
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    log.new_run()
    patched, _ = FunPackLoadModifiers.execute(tiny_h3.patcher, {"shot_memory": {"enabled": True}}).result
    assert any("Taste key" in e["message"] for e in log.history())
    assert not patched.wrappers.get(WrappersMP.SAMPLER_SAMPLE)


def test_a_manual_amount_of_zero_is_a_fresh_run_not_a_recorded_reuse(tiny_h3):
    from modules.system.taste import store
    sample, outer = _load(tiny_h3, mode="manual", amount=0.95)
    _go(sample, outer, _noise())
    store.rate("run-1", "liked")
    sample, outer = _load(tiny_h3, mode="manual", amount=0.0)
    for _ in range(30):                                  # Thompson would draw a reuse often
        import random
        random.seed(_)
        _go(sample, outer, _noise())
        pending = store.ROOT / "fox" / "shot_memory.pending.pt"
        assert int(torch.load(pending)["rows"]["parent"]) == -1
