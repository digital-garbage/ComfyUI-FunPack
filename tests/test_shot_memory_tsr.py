"""Shot memory (shot_memory.py) and decisiveness (tsr.py): the noise surgery keeps
the noise Gaussian, a liked shot is actually carried, ratings move what is learned,
and TSR is exactly off at k=1 and near pure noise."""
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shot_memory as sm  # noqa: E402
import tsr  # noqa: E402


def _store(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "_path", lambda key, mode: str(tmp_path / f"{key}.{mode}.pt"))
    monkeypatch.setattr(tsr, "_path", lambda key: str(tmp_path / f"{key}.tsr.json"))


def test_coarse_swap_is_exact_and_keeps_fine_noise():
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1, 24, 7, 27, 36, generator=g)
    assert torch.allclose(sm.with_coarse(v, sm.coarse_of(v)), v, atol=1e-5)
    target = torch.randn(24, 6, 9, generator=g)
    out = sm.with_coarse(v, target)
    assert torch.allclose(sm.coarse_of(out), target, atol=1e-4)
    # The fine part (block residual) is untouched, and so are the leftover edge cells.
    fine = lambda x: x[0, :, :, :24, :36] - x[0, :, :, :24, :36].reshape(24, 7, 6, 4, 9, 4) \
        .mean(dim=(1, 3, 5), keepdim=True).repeat_interleave(1, 0).reshape(24, 1, 6, 1, 9, 1) \
        .expand(24, 7, 6, 4, 9, 4).reshape(24, 7, 24, 36)
    assert torch.allclose(fine(out), fine(v), atol=1e-5)
    assert torch.equal(out[..., 24:, :], v[..., 24:, :])
    assert abs(float(out.std()) - 1.0) < 0.02


def test_shape_carries_the_liked_shot_by_amount(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    liked = torch.randn(24, 6, 9)
    rows = [{"id": 1, "parent": -1, "amount": 0.0, "reward": 1.0, "coarse": liked.half(), "cond": None}]
    torch.save({"rows": rows}, tmp_path / "k.shot_memory.pt")

    class AlwaysReuse(random.Random):
        def betavariate(self, a, b):
            # the untried reuse arm (1,1) beats fresh, which holds the liked fresh shot (2,1)
            return 1.0 if a == b == 1.0 else a / (a + b)

    mem = sm.ShotMemory("k", "manual", 0.9, rng=AlwaysReuse(0))
    noise = torch.randn(1, 24, 7, 24, 36)
    out = mem.shape(noise, torch.zeros_like(noise), torch.randn(1, 5, 8), record=True)
    corr = torch.nn.functional.cosine_similarity(sm.coarse_of(out).flatten(), liked.float().flatten(), dim=0)
    assert float(corr) > 0.8
    assert mem.used[0]["parent"] == 1 and mem.used[0]["amount"] == 0.9
    # A latent that already holds a picture (second pass) is left alone.
    assert torch.equal(mem.shape(noise, torch.ones_like(noise), None), noise)


def test_pending_commit_roundtrip(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    mem = sm.ShotMemory("k", "learned")
    noise = torch.randn(1, 24, 3, 8, 8)
    mem.shape(noise, torch.zeros_like(noise), None, record=True)
    mem.save_pending()
    assert sm.commit("k", 0.0) is None                     # neutral: discarded
    assert sm.commit("k", 1.0) is None                     # nothing pending any more
    mem.save_pending()
    assert sm.commit("k", -0.4) == 1
    assert sm.load_rows("k")[0]["reward"] == -1.0


def test_learned_amount_follows_ratings():
    liked = [{"parent": 1, "amount": 0.9, "reward": 1.0}] * 4
    assert sm.learned_amount(liked) > 0.85
    disliked = [{"parent": 1, "amount": 0.9, "reward": -1.0}] * 4
    assert sm.learned_amount(disliked) < sm.START_AMOUNT
    fresh_only = [{"parent": -1, "amount": 0.0, "reward": 1.0}] * 4
    assert sm.learned_amount(fresh_only) == sm.START_AMOUNT


def test_reuse_vs_fresh_learns_from_ratings():
    c = torch.zeros(24, 2, 2)
    base = [{"id": 1, "parent": -1, "reward": 1.0, "coarse": c}]
    hated_reuse = base + [{"id": 10 + i, "parent": 1, "reward": -1.0, "coarse": c} for i in range(8)]
    loved_reuse = base + [{"id": 10 + i, "parent": 1, "reward": 1.0, "coarse": c} for i in range(8)]
    rng = random.Random(0)
    reused = lambda rows: sum(sm.choose_parent(rows, None, 24, rng)[0] is not None for _ in range(300))
    assert reused(hated_reuse) < 60
    assert reused(loved_reuse) > 240
    assert sm.choose_parent(base, None, 128, rng)[0] is None   # another model's channels


def test_tsr_is_off_at_k1_and_near_pure_noise():
    x, x0 = torch.randn(10), torch.randn(10)
    assert torch.equal(tsr.rescale_x0(x, x0, 0.5, 1.0), x0)
    assert abs(tsr.factor(0.999, 1.3) - 1.0) < 1e-3
    assert abs(tsr.factor(0.2, 1.3) - 1.3) < 0.02
    sharp = tsr.rescale_x0(x, x0, 0.5, 1.3)
    # Same point on the path: x = a*x0' + sigma*r*eps holds.
    eps = (x - 0.5 * x0) / 0.5
    assert torch.allclose(0.5 * sharp + 0.5 * tsr.factor(0.5, 1.3) * eps, x, atol=1e-5)


def test_tsr_learns_toward_liked_k(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    rng = random.Random(1)
    for _ in range(30):
        k, _note = tsr.choose_k("k", "learned", rng=rng)
        tsr.save_pending("k", k)
        tsr.commit("k", 1.0 if k > 1.0 else -1.0)
    k, _note = tsr.choose_k("k", "learned", rng=random.Random(2))
    assert k > 1.05
    assert tsr.commit("k", 1.0) is None                    # nothing pending
    assert tsr.choose_k("k", "off")[0] == 1.0


def test_tsr_wrapper_touches_picture_rows_only(monkeypatch):
    import samplers
    monkeypatch.setattr(samplers, "_video_span", lambda model, x: (0, 6, (1, 1, 1, 2, 3)))

    class Model:
        model_options = {}

    model = Model()
    samplers.FunPackLTXAVSceneChainSampler()._build_tsr_wrapper(model, 1.3)
    x, x0 = torch.randn(1, 1, 10), torch.randn(1, 1, 10)
    out = model.model_options["model_function_wrapper"](
        lambda inp, t, **c: x0.clone(), {"input": x, "timestep": torch.tensor([0.5]), "c": {}})
    assert torch.equal(out[..., 6:], x0[..., 6:])          # sound untouched
    assert torch.allclose(out[..., :6], tsr.rescale_x0(x[..., :6], x0[..., :6], 0.5, 1.3))


def test_a_run_with_the_feature_off_clears_the_older_pending(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    mem = sm.ShotMemory("k", "learned")
    noise = torch.randn(1, 24, 3, 8, 8)
    mem.shape(noise, torch.zeros_like(noise), None, record=True)
    mem.save_pending()                       # run A, unrated
    sm.clear_pending("k")                    # run B, shot memory off
    assert sm.commit("k", -1.0) is None      # B's rating teaches nothing about A
    tsr.save_pending("k", 1.1)
    tsr.clear_pending("k")
    assert tsr.commit("k", -1.0) is None


def test_neutral_rating_teaches_and_reports_nothing(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    tsr.save_pending("k", 1.1)
    assert tsr.commit("k", 0.0) is None
    assert tsr.commit("k", 1.0) is None                    # and the pending is gone


def test_tsr_wrapper_leaves_an_unreadable_packed_latent_alone(monkeypatch):
    import samplers
    monkeypatch.setattr(samplers, "_video_span", lambda model, x: None)

    class Model:
        model_options = {}

    model = Model()
    samplers.FunPackLTXAVSceneChainSampler()._build_tsr_wrapper(model, 1.3)
    x = torch.randn(1, 1, 10)
    x0 = torch.randn(1, 1, 10)
    wrapper = model.model_options["model_function_wrapper"]
    call = lambda inp, t, **c: x0.clone()
    assert torch.equal(wrapper(call, {"input": x, "timestep": torch.tensor([0.5]), "c": {}}), x0)
    x5, y5 = torch.randn(1, 2, 3, 4, 4), torch.randn(1, 2, 3, 4, 4)
    out = wrapper(lambda inp, t, **c: y5.clone(), {"input": x5, "timestep": torch.tensor([0.5]), "c": {}})
    assert torch.allclose(out, tsr.rescale_x0(x5, y5, 0.5, 1.3))


class _Nested:
    """comfy.nested_tensor.NestedTensor's surface as prepare_noise hands it over."""
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors


def test_picture_and_sound_noise_only_the_picture_changes(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    monkeypatch.setattr(sys.modules["comfy.nested_tensor"], "NestedTensor", _Nested, raising=False)
    torch.save({"rows": [{"id": 1, "parent": -1, "amount": 0.0, "reward": 1.0,
                          "coarse": torch.randn(24, 2, 2).half(), "cond": None}]},
               tmp_path / "k.shot_memory.pt")
    mem = sm.ShotMemory("k", "manual", 0.9, rng=random.Random(3))
    mem.plans[(None, 24)] = (sm.load_rows("k")[0], 0.9)       # force the reuse decision
    video, audio = torch.randn(1, 24, 3, 8, 8), torch.randn(1, 32, 2, 40)
    noise = _Nested([video, audio])
    empty = _Nested([torch.zeros_like(video), torch.zeros_like(audio)])
    out = mem.shape(noise, empty, None, record=True)
    assert torch.equal(out.tensors[1], audio)
    assert not torch.equal(out.tensors[0], video)


def test_one_rating_is_one_row_and_skips_are_counted(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    mem = sm.ShotMemory("k", "learned")
    for _scene in range(3):
        n = torch.randn(1, 24, 3, 8, 8)
        mem.shape(n, torch.zeros_like(n), None, record=True)
    mem.shape(n, torch.ones_like(n), None, record=True)       # e.g. a second pass
    assert len(mem.used) == 1 and mem.skipped == 1
    mem.save_pending()
    assert sm.commit("k", 1.0) == 1


def test_reusing_a_shot_with_its_own_seed_keeps_normal_strength(monkeypatch, tmp_path):
    """Same seed as the liked shot: the two layouts are one pattern, and a plain blend
    would stack it (1.41x at amount 0.7)."""
    _store(monkeypatch, tmp_path)
    noise = torch.randn(1, 24, 7, 24, 36, generator=torch.Generator().manual_seed(5))
    mem = sm.ShotMemory("k", "manual", 0.7, rng=random.Random(0))
    mem.plans[(None, 24)] = ({"id": 1, "coarse": sm.coarse_of(noise).half()}, 0.7)
    out = mem.shape(noise.clone(), torch.zeros_like(noise), None)
    assert abs(float(sm.coarse_of(out).std()) - 1.0) < 0.02


def test_a_frame_too_small_for_a_layout_is_left_alone_and_said(monkeypatch, tmp_path):
    """64x64 px on H3 is one 64px cell: nothing to rescale (the std of one value is NaN)."""
    _store(monkeypatch, tmp_path)
    mem = sm.ShotMemory("k", "manual", 0.7, rng=random.Random(0))
    mem.plans[(None, 24)] = ({"id": 1, "coarse": torch.randn(24, 1, 1).half()}, 0.7)
    noise = torch.randn(1, 24, 7, 4, 4)
    out = mem.shape(noise, torch.zeros_like(noise), None, record=True)
    assert torch.equal(out, noise) and not torch.isnan(out).any()
    assert any("left alone" in n for n in mem.notes)


def test_a_two_cell_grid_never_makes_the_noise_stronger(monkeypatch, tmp_path):
    """2x2 cells: 4 values per channel, so their measured spread is often low by chance."""
    _store(monkeypatch, tmp_path)
    worst = plain = 0.0
    for i in range(300):
        torch.manual_seed(i)
        mem = sm.ShotMemory("k", "manual", 0.95, rng=random.Random(i))
        mem.plans[(None, 24)] = ({"id": 1, "coarse": torch.randn(24, 2, 2).half()}, 0.95)
        noise = torch.randn(1, 24, 3, 8, 8)
        out = mem.shape(noise, torch.zeros_like(noise), None, record=True)
        plain = max(plain, float(sm.coarse_of(noise).abs().max()))
        worst = max(worst, float(sm.coarse_of(out).abs().max()))
    assert worst < 1.2 * plain
