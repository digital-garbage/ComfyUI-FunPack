"""Trajectory guidance: one value function per schedule bucket, each steering in its own
window.

The point of the feature is reaching the FIRST half of a generation, which every other
rating-driven mechanism is gated out of. So the tests that matter are: a head learns from
its own bucket and not from the others, an untrained bucket does nothing rather than
something arbitrary, and the wrapper actually changes the prediction where it claims to and
leaves it byte-identical where it does not — because "present and inert" is the failure this
project keeps shipping, and a same-seed A/B is exactly how it gets caught.
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in (
    "comfy", "comfy.k_diffusion", "comfy.k_diffusion.sampling",
    "comfy.model_sampling", "comfy.nested_tensor", "comfy.sample",
    "comfy.samplers", "comfy.utils",
):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.nested_tensor"].NestedTensor = object

import samplers  # noqa: E402
import trajectory_guidance as tg  # noqa: E402
import trajectory_probe as probe  # noqa: E402

DIM = 512


def _row(reward, per_bucket, buckets=4, stamp="s"):
    """One rated run: a descriptor per bucket, built by `per_bucket(bucket)`."""
    return {"reward": reward, "stamp": stamp, "prompt_hash": "p",
            "rows": [{"scene": 0, "pass": 0, "bucket": b, "desc": per_bucket(b)}
                     for b in range(buckets)]}


def _separable(n=14):
    """Runs where bucket 1 tells good from bad and bucket 0 carries nothing."""
    gen = torch.Generator().manual_seed(5)
    rows = []
    for i in range(n):
        good = i % 2 == 0
        def per_bucket(b, good=good, gen=gen):
            noise = torch.randn(DIM, generator=gen) * 0.1
            if b == 1:
                return noise + (1.0 if good else -1.0)
            return noise
        rows.append(_row(1.0 if good else -1.0, per_bucket, stamp=f"s{i:02d}"))
    return rows


def _store(monkeypatch, tmp_path):
    monkeypatch.setattr(tg, "state_path", lambda key: str(tmp_path / f"{key}.buckets.pt"))


# --- learning -------------------------------------------------------------


def test_a_head_learns_from_its_own_bucket_only(monkeypatch, tmp_path):
    """The whole claim of the feature: what steers at step 2 learned from step 2. A head
    fed every bucket's descriptors would just be output_guidance with extra files.

    Measured as ranking accuracy over the descriptors each head actually saw, not as a score
    at some constant vector: a head trained on noise will happily extrapolate to a large
    arbitrary value far outside its data, which made an earlier version of this test pass or
    fail on `train_on`'s unseeded bootstrap sampling rather than on anything real.
    """
    import random as _random
    _random.seed(7)                      # train_on bootstraps with the global RNG
    _store(monkeypatch, tmp_path)
    rows = _separable(n=28)
    train, held_out = rows[:20], rows[20:]
    tg.train_from_rows("k", train)
    model = tg.load("k")
    assert model is not None

    def accuracy(bucket, on):
        pos, neg = [], []
        with torch.inference_mode(False), torch.no_grad():
            for row in on:
                desc = next(c["desc"] for c in row["rows"] if c["bucket"] == bucket)
                score = float(model._head(bucket)(desc.unsqueeze(0)).mean())
                (pos if row["reward"] > 0 else neg).append(score)
        return sum(p > q for p in pos for q in neg) / (len(pos) * len(neg))

    # HELD OUT, not the training rows. A few-hundred-parameter MLP fits 20 noise vectors
    # perfectly, so training accuracy is 1.0 for every bucket and says nothing about whether
    # a head learned anything -- including the head that will be steering real generations.
    assert accuracy(1, held_out) > 0.9, accuracy(1, held_out)
    assert accuracy(0, held_out) < 0.9, accuracy(0, held_out)


def test_an_untrained_bucket_is_not_ready(monkeypatch, tmp_path):
    """Under MIN_SAMPLES a head has nothing to say, and saying it anyway would steer on
    three ratings."""
    _store(monkeypatch, tmp_path)
    tg.train_from_rows("k", _separable(n=4))
    assert tg.load("k") is None


def test_heads_survive_a_restart(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    counts = tg.train_from_rows("k", _separable())
    reloaded = tg.load("k")
    assert reloaded is not None
    assert reloaded.trained() == counts


def test_the_second_pass_does_not_train_the_first_pass_windows(monkeypatch, tmp_path):
    """A second pass runs its own schedule from its own starting sigma, so its bucket 2 is
    not the window bucket 2 steers in."""
    _store(monkeypatch, tmp_path)
    rows = _separable()
    for r in rows:
        r["rows"].append({"scene": 0, "pass": 1, "bucket": 1,
                          "desc": torch.full((DIM,), 99.0)})
    tg.train_from_rows("k", rows)
    model = tg.load("k")
    assert all(abs(v) < 50 for v in model._head(1).buffer_c[0].tolist()[:5])


def test_per_scene_rewards_beat_the_run_reward(monkeypatch, tmp_path):
    """A multi-scene run rated per scene carries one reward per scene; using the aggregate
    would train a Perfect scene as a bad sample."""
    _store(monkeypatch, tmp_path)
    rows = []
    for i in range(14):
        good = i % 2 == 0
        rows.append({
            "reward": -0.9, "stamp": f"m{i}", "scene_rewards": {0: 1.0 if good else -1.0},
            "rows": [{"scene": 0, "pass": 0, "bucket": 1,
                      "desc": torch.full((DIM,), 1.0 if good else -1.0)}],
        })
    tg.train_from_rows("k", rows)
    model = tg.load("k")
    with torch.inference_mode(False), torch.no_grad():
        assert float(model._head(1)(torch.full((1, DIM), 1.0)).mean()) > \
               float(model._head(1)(torch.full((1, DIM), -1.0)).mean())


def test_nothing_rated_trains_nothing(monkeypatch, tmp_path):
    _store(monkeypatch, tmp_path)
    assert tg.train_from_rows("k", []) is None
    assert tg.train_from_rows(None, _separable()) is None
    assert tg.load("k") is None


# --- steering -------------------------------------------------------------

C, T, H, W = 4, 3, 2, 2
VIDEO_N = C * T * H * W
AUDIO_N = 12


def _model():
    ls = types.SimpleNamespace(cond=[(1, C, T, H, W), (1, AUDIO_N)])
    guider = types.SimpleNamespace(conds={"positive": [{"model_conds": {"latent_shapes": ls}}]})
    return types.SimpleNamespace(model_options={}, inner_model=guider)


def _install(model, buckets, bucket_for, strength=0.05):
    samplers.FunPackLTXAVSceneChainSampler()._build_trajectory_guidance_wrapper(
        model, buckets, strength, bucket_for)
    return model.model_options["model_function_wrapper"]


def _predict(wrapper, latent, sigma=0.9):
    return wrapper(lambda x, t, **kw: latent,
                   {"input": latent, "timestep": torch.tensor([sigma]), "c": {}})


class _Ready:
    """A stand-in whose gradient is known, so the wrapper's own arithmetic is what is
    under test rather than a value function's."""
    def __init__(self, ready_buckets, width=VIDEO_N):
        self.n_buckets = 4
        self._ready = set(ready_buckets)
        self._width = width
        self.asked = []

    def ready(self, b):
        return b in self._ready

    def gradient(self, b, x):
        self.asked.append(b)
        return torch.ones_like(x)


def test_a_ready_window_changes_the_prediction():
    model = _model()
    buckets = _Ready([0])
    wrapper = _install(model, buckets, lambda s: 0)
    latent = torch.ones(1, 1, VIDEO_N + AUDIO_N)
    out = _predict(wrapper, latent)
    assert not torch.equal(out, latent), "a window that reports itself ready must act"
    assert buckets.asked == [0]


def test_an_unready_window_leaves_the_prediction_byte_identical():
    """The same-seed A/B this feature is meant to be judged by: off and on must be
    indistinguishable wherever it cannot act, or 'nothing changed' stops being readable."""
    model = _model()
    wrapper = _install(model, _Ready([1]), lambda s: 0)     # asked for 0, only 1 is ready
    latent = torch.ones(1, 1, VIDEO_N + AUDIO_N)
    assert torch.equal(_predict(wrapper, latent), latent)


def test_audio_is_never_touched():
    model = _model()
    wrapper = _install(model, _Ready([0]), lambda s: 0)
    latent = torch.cat([torch.ones(VIDEO_N), torch.arange(AUDIO_N, dtype=torch.float32)])
    out = _predict(wrapper, latent.view(1, 1, -1))
    assert torch.equal(out[..., VIDEO_N:], latent[VIDEO_N:].view(1, 1, -1))


def test_a_batched_prediction_is_left_alone():
    """At CFG>1 comfy stacks the unconditional row into the same call, and a correction on
    that blend steers toward the average of a prediction and its negation."""
    model = _model()
    buckets = _Ready([0])
    wrapper = _install(model, buckets, lambda s: 0)
    batched = torch.ones(2, 1, VIDEO_N + AUDIO_N)
    assert torch.equal(_predict(wrapper, batched), batched)
    assert buckets.asked == []


def test_a_sigma_off_the_schedule_does_not_steer():
    model = _model()
    buckets = _Ready([0, 1, 2, 3])
    wrapper = _install(model, buckets, lambda s: None)
    latent = torch.ones(1, 1, VIDEO_N + AUDIO_N)
    assert torch.equal(_predict(wrapper, latent), latent)
    assert buckets.asked == []


def test_the_step_is_scaled_to_the_streams_own_norm():
    """`strength` means a fraction of the video stream's norm, the same convention
    output_guidance and the score slider use, so the number transfers between them."""
    model = _model()
    wrapper = _install(model, _Ready([0]), lambda s: 0, strength=0.05)
    latent = torch.cat([torch.ones(VIDEO_N), torch.zeros(AUDIO_N)]).view(1, 1, -1)
    out = _predict(wrapper, latent)
    moved = (out[..., :VIDEO_N] - latent[..., :VIDEO_N]).norm()
    assert abs(float(moved / latent[..., :VIDEO_N].norm()) - 0.05) < 1e-4


def test_a_failing_head_costs_the_step_not_the_generation():
    class _Boom(_Ready):
        def gradient(self, b, x):
            raise RuntimeError("no")
    model = _model()
    wrapper = _install(model, _Boom([0]), lambda s: 0)
    latent = torch.ones(1, 1, VIDEO_N + AUDIO_N)
    assert torch.equal(_predict(wrapper, latent), latent)


def test_it_chains_onto_whatever_is_already_installed():
    model = _model()
    calls = []
    model.model_options["model_function_wrapper"] = lambda fn, a: (
        calls.append(1), fn(a["input"], a["timestep"], **a.get("c", {})))[1]
    wrapper = _install(model, _Ready([]), lambda s: 0)
    _predict(wrapper, torch.ones(1, 1, VIDEO_N + AUDIO_N))
    assert calls == [1]


def test_training_works_inside_inference_mode(monkeypatch, tmp_path):
    """ComfyUI executes a node under torch.inference_mode(), so a descriptor read out of the
    log becomes an inference tensor and autograd refuses it:

        Trajectory probe intake failed: Inference tensors do not track version counter.

    Seen on a real rental. Every head stayed empty and the feature reported "nothing learned
    yet" on every run, so it was OFF while appearing merely untrained — the one failure that
    is invisible from the outside, because "not steering" is also what correct behaviour
    looks like before enough ratings.
    """
    _store(monkeypatch, tmp_path)
    with torch.inference_mode():
        counts = tg.train_from_rows("k", _separable())
    assert counts, "training produced nothing under inference mode"
    assert tg.load("k") is not None


def test_steering_works_inside_inference_mode(monkeypatch, tmp_path):
    """The other half: sampling also runs under inference mode, so the gradient the wrapper
    asks for is computed there too."""
    _store(monkeypatch, tmp_path)
    with torch.inference_mode():
        tg.train_from_rows("k", _separable())
    model = tg.load("k")
    ready = [b for b in range(model.n_buckets) if model.ready(b)]
    assert ready, "nothing trained, so this would pass for the wrong reason"

    host = _model()
    wrapper = _install(host, model, lambda s: ready[0])
    latent = torch.randn(1, 1, VIDEO_N + AUDIO_N)
    with torch.inference_mode():
        out = _predict(wrapper, latent)
    assert not torch.equal(out, latent)
    assert torch.isfinite(out).all()
