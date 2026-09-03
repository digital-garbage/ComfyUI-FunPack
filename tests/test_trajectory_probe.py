"""Trajectory probe (trajectory_probe.py) + its sampler observer.

The probe exists to answer one question — do ratings separate in x0_hat space at EARLY
schedule positions, where nothing currently steers — so these tests cover the two things
that would make its answer wrong rather than merely absent:

  * recording: H3 bucketing over shift-generated schedules, the clamp to few-step runs,
    the CFG>1 dedupe, per-scene separation, and the run/rating pairing;
  * the statistic: separable data has to register, noise must NOT, and a separation that
    is really about which prompt was running must not pass as a rating effect.

Plus the observer's one hard contract: it records without changing the prediction.
"""
import os
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
import trajectory_probe as probe  # noqa: E402


def _patch_store(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_probe_dir", lambda: str(tmp_path))


def _h3_sigmas(steps, shift=12.0):
    """H3's schedule: sigma = shift*t / (1 + (shift-1)*t) over uniform t, descending.

    The point of bucketing by POSITION is that this stays near 1.0 for most of the run at
    shift 12 — the exact reason the steering gate had to stop reading absolute sigma.
    """
    out = []
    for i in range(steps + 1):
        t = 1.0 - i / steps
        out.append(shift * t / (1.0 + (shift - 1.0) * t))
    return torch.tensor(out)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def test_h3_buckets_follow_position_not_sigma_value():
    """On a 12-step shift-12 schedule every sigma but the last sits above 0.5, so an
    absolute-sigma reading would put the whole run in one bucket. Position must not."""
    sigmas = _h3_sigmas(12, shift=12.0)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    buckets = [rec.bucket_for(float(sigmas[k])) for k in range(12)]
    assert buckets == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    # ...and the naive reading really would have collapsed: most steps are still high sigma.
    assert sum(1 for k in range(12) if float(sigmas[k]) > 0.5) >= 9


def test_buckets_clamp_to_step_count_on_turbo():
    """4-step turbo asked for 4 buckets gets one step each; a 3-step run cannot report 4
    buckets it never measured."""
    assert probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4).n_buckets == 4
    assert probe.TrajectoryRecorder(_h3_sigmas(3), n_buckets=4).n_buckets == 3
    assert probe.TrajectoryRecorder(_h3_sigmas(7), n_buckets=4).n_buckets == 4


def test_recorder_without_a_schedule_records_nothing():
    rec = probe.TrajectoryRecorder(None)
    assert rec.bucket_for(0.5) is None
    assert rec.observe(0.5, torch.ones(probe.DESCRIPTOR_DIM)) is False
    assert rec.is_empty()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_bucket_mean_averages_its_steps():
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=2)
    rec.observe(float(sigmas[0]), torch.full((probe.DESCRIPTOR_DIM,), 1.0))
    rec.observe(float(sigmas[1]), torch.full((probe.DESCRIPTOR_DIM,), 3.0))
    rows = rec.cell_rows()
    assert [r["bucket"] for r in rows] == [0]
    assert rows[0]["steps"] == 2
    assert torch.allclose(rows[0]["desc"].float(), torch.full((probe.DESCRIPTOR_DIM,), 2.0))


def test_repeated_sigma_is_recorded_once():
    """H3 samples at CFG 1.0 (one call per step), so this never fires there. On a CFG>1
    model it stops the unconditional prediction being averaged into the measurement."""
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    assert rec.observe(float(sigmas[0]), torch.ones(probe.DESCRIPTOR_DIM)) is True
    assert rec.observe(float(sigmas[0]), torch.zeros(probe.DESCRIPTOR_DIM)) is False
    rows = rec.cell_rows()
    assert rows[0]["steps"] == 1
    assert torch.allclose(rows[0]["desc"].float(), torch.ones(probe.DESCRIPTOR_DIM))


def test_scenes_do_not_average_into_each_other():
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    rec.begin_scene(0)
    rec.observe(float(sigmas[0]), torch.ones(probe.DESCRIPTOR_DIM))
    rec.begin_scene(1)
    rec.observe(float(sigmas[0]), torch.full((probe.DESCRIPTOR_DIM,), 5.0))
    rows = rec.cell_rows()
    assert [(r["scene"], r["bucket"]) for r in rows] == [(0, 0), (1, 0)]
    assert float(rows[0]["desc"][0]) == 1.0 and float(rows[1]["desc"][0]) == 5.0


def test_a_second_schedules_steps_are_not_nearest_matched_into_the_first():
    """A scene can sample twice: a second pass runs its own schedule IN FULL from its own
    starting sigma, with the observer still installed. Those sigmas have no position on the
    first schedule, and nearest-matching them silently averages two different denoisings
    into one bucket."""
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    for k in range(4):
        rec.observe(float(_h3_sigmas(4)[k]), torch.ones(probe.DESCRIPTOR_DIM))
    first = {r["bucket"]: r["desc"].clone() for r in rec.cell_rows()}

    # Every one of these is far below the first schedule's lowest step sigma, so all four
    # would nearest-match into its last bucket.
    for sigma in (0.35, 0.2, 0.08, 0.01):
        assert rec.observe(sigma, torch.full((probe.DESCRIPTOR_DIM,), 99.0)) is False
    assert rec._foreign == 4
    for row in rec.cell_rows():
        assert torch.equal(row["desc"], first[row["bucket"]]), "pass 1 was contaminated"


def test_a_bound_second_pass_is_measured_separately():
    second = torch.tensor([0.35, 0.2, 0.08, 0.0])
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    for k in range(4):
        rec.observe(float(_h3_sigmas(4)[k]), torch.ones(probe.DESCRIPTOR_DIM))
    rec.begin_pass(second, index=1)
    for k in range(3):
        rec.observe(float(second[k]), torch.full((probe.DESCRIPTOR_DIM,), 99.0))

    rows = rec.cell_rows()
    assert {r["pass"] for r in rows} == {0, 1}
    assert all(float(r["desc"][0]) == 1.0 for r in rows if r["pass"] == 0)
    assert all(float(r["desc"][0]) == 99.0 for r in rows if r["pass"] == 1)
    assert rec._foreign == 0
    assert rec.pass_steps == {0: 4, 1: 3}


def test_analysis_reads_the_first_pass_only():
    """Pass 2's bucket 1 is a different sigma window than pass 1's, so pooling them would
    compare unlike things while reporting a single bucket."""
    rows = [{
        "reward": 1.0, "prompt_hash": "p",
        "rows": [{"bucket": 0, "scene": 0, "pass": 0, "desc": torch.ones(4)},
                 {"bucket": 0, "scene": 0, "pass": 1, "desc": torch.full((4,), 50.0)}],
    }, {
        "reward": -1.0, "prompt_hash": "p",
        "rows": [{"bucket": 0, "scene": 0, "pass": 0, "desc": torch.zeros(4)},
                 {"bucket": 0, "scene": 0, "pass": 1, "desc": torch.full((4,), 50.0)}],
    }]
    descriptors, labels, _ = probe.collect_bucket(rows, 0)
    assert labels == [1, 0]
    assert torch.equal(descriptors[0], torch.ones(4))
    assert torch.equal(descriptors[1], torch.zeros(4))
    second, _, _ = probe.collect_bucket(rows, 0, pass_index=1)
    assert all(torch.equal(d, torch.full((4,), 50.0)) for d in second)


def test_unbound_steps_are_recorded_on_the_run(monkeypatch, tmp_path):
    """A measurement taken from fewer steps than the run actually sampled has to say so."""
    _patch_store(monkeypatch, tmp_path)
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    for k in range(4):
        rec.observe(float(_h3_sigmas(4)[k]), torch.ones(probe.DESCRIPTOR_DIM))
    rec.observe(0.123456, torch.ones(probe.DESCRIPTOR_DIM))
    probe.save_pending("k", rec)
    probe.commit("k", 1.0)
    assert probe.load_log("k")[0]["unbound_steps"] == 1


def test_wrong_width_descriptor_is_refused():
    rec = probe.TrajectoryRecorder(_h3_sigmas(4))
    assert rec.observe(1.0, torch.ones(8)) is False
    assert rec.is_empty()


# ---------------------------------------------------------------------------
# Run/rating pairing
# ---------------------------------------------------------------------------


def _recorded(sigmas, value=1.0, n_buckets=4):
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=n_buckets)
    for k in range(len(sigmas) - 1):
        rec.observe(float(sigmas[k]), torch.full((probe.DESCRIPTOR_DIM,), value))
    return rec


def test_pending_pairs_with_the_rating_that_scores_it(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    sigmas = _h3_sigmas(4)
    assert probe.save_pending("k", _recorded(sigmas), prompt_hash="p1", seed=7,
                              meta={"h3": True, "shift_video": 12.0, "shift_audio": 3.0})
    assert probe.commit("k", 1.0, rating_key="perfect") == 1

    rows = probe.load_log("k")
    assert len(rows) == 1
    assert rows[0]["reward"] == 1.0 and rows[0]["rating"] == "perfect"
    assert rows[0]["prompt_hash"] == "p1" and rows[0]["seed"] == 7
    assert rows[0]["steps"] == 4
    assert probe.schedule_id(rows[0]) == (4, 12.0, 3.0)
    # The pending is consumed, so a later unrelated rating cannot log this run twice.
    assert probe.commit("k", -1.0) is None
    assert len(probe.load_log("k")) == 1


def test_commit_without_a_run_logs_nothing(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    assert probe.commit("k", 1.0) is None
    assert probe.load_log("k") == []


def test_a_run_that_measured_nothing_still_replaces_the_pending(monkeypatch, tmp_path):
    """Otherwise the previous run's pending survives and the NEXT rating consumes it, filing
    one run's descriptors under another run's rating — a silently mislabelled row in the one
    dataset the module exists to produce."""
    _patch_store(monkeypatch, tmp_path)
    sigmas = _h3_sigmas(4)
    probe.save_pending("k", _recorded(sigmas, value=1.0), prompt_hash="run-A", seed=111)

    # Run B measures nothing (every step refused — e.g. a CFG>1 batch on every call).
    empty = probe.TrajectoryRecorder(sigmas)
    empty.note_unmeasurable()
    assert empty.is_empty()
    probe.save_pending("k", empty, prompt_hash="run-B", seed=222)

    # Rating run B must not log run A's descriptors under run B's rating.
    assert probe.commit("k", 1.0, rating_key="perfect") is None
    assert probe.load_log("k") == []
    # ...and the stale candidate is gone, so no later rating can pick it up either.
    assert probe.commit("k", -1.0) is None


def test_save_pending_needs_a_key_and_a_recorder(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    assert probe.save_pending(None, _recorded(_h3_sigmas(4))) is False
    assert probe.save_pending("k", None) is False


def test_log_is_capped(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setattr(probe, "MAX_ROWS", 3)
    sigmas = _h3_sigmas(4)
    for i in range(5):
        probe.save_pending("k", _recorded(sigmas, value=float(i)))
        probe.commit("k", float(i))
    rows = probe.load_log("k")
    assert len(rows) == 3
    assert [r["reward"] for r in rows] == [2.0, 3.0, 4.0]  # oldest rolled off


def test_logs_pool_across_disposable_keys(monkeypatch, tmp_path):
    """Keys get cleared between rentals; the measurement is supposed to outlive them."""
    _patch_store(monkeypatch, tmp_path)
    for key in ("alpha", "beta"):
        probe.save_pending(key, _recorded(_h3_sigmas(4)))
        probe.commit(key, 1.0)
    assert sorted(k for k, _ in probe.load_all_logs()) == ["alpha", "beta"]
    assert sum(len(r) for _, r in probe.load_all_logs()) == 2


def test_schedule_mix_names_incompatible_runs():
    rows = [
        {"steps": 4, "meta": {"shift_video": 12.0, "shift_audio": 3.0}},
        {"steps": 4, "meta": {"shift_video": 12.0, "shift_audio": 3.0}},
        {"steps": 12, "meta": {"shift_video": 3.0, "shift_audio": 3.0}},
    ]
    mix = probe.schedule_mix(rows)
    assert mix[(4, 12.0, 3.0)] == 2 and mix[(12, 3.0, 3.0)] == 1


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


def _descriptors(offsets, dim=32, noise=0.05):
    gen = torch.Generator().manual_seed(0)
    return [torch.randn(dim, generator=gen) * noise + off for off in offsets]


def test_separable_groups_register():
    base = torch.zeros(32)
    shift = torch.zeros(32)
    shift[0] = 4.0
    descriptors = _descriptors([base] * 5 + [shift] * 5)
    labels = [1] * 5 + [0] * 5
    result = probe.permutation_test(descriptors, labels, trials=500, seed=1)
    assert result["separation"] > 0
    assert result["p_value"] <= 0.05


def test_noise_does_not_register():
    gen = torch.Generator().manual_seed(3)
    descriptors = [torch.randn(32, generator=gen) for _ in range(12)]
    labels = [1, 0] * 6
    result = probe.permutation_test(descriptors, labels, trials=500, seed=1)
    assert result["p_value"] > 0.05


def test_prompt_confound_does_not_pass_as_a_rating_effect():
    """Every good run on prompt A, every bad run on prompt B. Pooled, that separates
    perfectly and means nothing. Restricted to same-prompt pairs there is no evidence at
    all — which is why the analysis reports the stratified row."""
    a, b = torch.zeros(32), torch.zeros(32)
    b[0] = 6.0
    descriptors = _descriptors([a] * 4 + [b] * 4)
    labels = [1] * 4 + [0] * 4
    groups = ["A"] * 4 + ["B"] * 4

    assert probe.permutation_test(descriptors, labels, trials=300, seed=1)["p_value"] <= 0.05
    # No pair inside a prompt has differing labels, so there is nothing to compare.
    assert probe.separation_statistic(descriptors, labels, groups) is None
    assert probe.permutation_test(descriptors, labels, groups, trials=300, seed=1) is None


def test_statistic_needs_both_labels():
    descriptors = _descriptors([torch.zeros(32)] * 4)
    assert probe.separation_statistic(descriptors, [1, 1, 1, 1]) is None
    assert probe.separation_statistic(descriptors[:2], [1, 0]) is None


def test_neutral_ratings_are_dropped_not_assigned():
    """'Forget it' carries reward 0.0 and is explicitly not a judgement."""
    rows = [
        {"reward": 1.0, "prompt_hash": "p", "rows": [{"bucket": 0, "scene": 0, "desc": torch.ones(4)}]},
        {"reward": 0.0, "prompt_hash": "p", "rows": [{"bucket": 0, "scene": 0, "desc": torch.ones(4)}]},
        {"reward": -0.5, "prompt_hash": "p", "rows": [{"bucket": 0, "scene": 0, "desc": torch.zeros(4)}]},
    ]
    descriptors, labels, _ = probe.collect_bucket(rows, 0)
    assert labels == [1, 0] and len(descriptors) == 2


def test_analyse_reports_every_recorded_bucket():
    def _row(reward, scale):
        return {"reward": reward, "prompt_hash": "p",
                "rows": [{"bucket": b, "scene": 0, "desc": torch.full((8,), scale * (b + 1))}
                         for b in range(4)]}
    rows = [_row(1.0, 1.0) for _ in range(3)] + [_row(-1.0, -1.0) for _ in range(3)]
    results = probe.analyse(rows, trials=100)
    assert [e["bucket"] for e in results] == [0, 1, 2, 3]
    assert all(e["n_good"] == 3 and e["n_bad"] == 3 for e in results)


# ---------------------------------------------------------------------------
# The sampler observer
# ---------------------------------------------------------------------------

C, T, H, W = 4, 3, 2, 2
VIDEO_N = C * T * H * W
AUDIO_N = 12


def _packed_av_model():
    ls = types.SimpleNamespace(cond=[(1, C, T, H, W), (1, AUDIO_N)])
    guider = types.SimpleNamespace(conds={"positive": [{"model_conds": {"latent_shapes": ls}}]})
    return types.SimpleNamespace(model_options={}, inner_model=guider)


def test_observer_records_without_touching_the_prediction():
    model = _packed_av_model()
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    sampler = samplers.FunPackLTXAVSceneChainSampler()
    sampler._build_trajectory_probe_wrapper(model, rec)
    wrapper = model.model_options["model_function_wrapper"]

    denoised = torch.arange(VIDEO_N + AUDIO_N, dtype=torch.float32).view(1, 1, -1)
    for k in range(4):
        out = wrapper(lambda x, t, **kw: denoised,
                      {"input": denoised, "timestep": torch.tensor([float(sigmas[k])]), "c": {}})
        assert torch.equal(out, denoised), "the probe must observe, never steer"
    assert [r["bucket"] for r in rec.cell_rows()] == [0, 1, 2, 3]


def test_observer_pools_video_only():
    """Audio must not reach the descriptor: it is not the domain LatentValueFunction is
    trained on, and on H3 the audio stream is a different clock entirely."""
    model = _packed_av_model()
    sigmas = _h3_sigmas(4)
    sampler = samplers.FunPackLTXAVSceneChainSampler()

    def _descriptor_for(audio_fill):
        rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
        model.model_options.pop("model_function_wrapper", None)
        sampler._build_trajectory_probe_wrapper(model, rec)
        latent = torch.cat([torch.ones(VIDEO_N), torch.full((AUDIO_N,), audio_fill)]).view(1, 1, -1)
        model.model_options["model_function_wrapper"](
            lambda x, t, **kw: latent,
            {"input": latent, "timestep": torch.tensor([float(sigmas[0])]), "c": {}})
        return rec.cell_rows()[0]["desc"].float()

    assert torch.allclose(_descriptor_for(0.0), _descriptor_for(99.0))


def test_descriptors_survive_inference_mode(monkeypatch, tmp_path):
    """Sampling runs under inference_mode, and these descriptors outlive the call that made
    them: accumulated across steps, then written to disk. This drives the whole path that
    way — record, accumulate, save, reload.

    The observer takes the inference_mode(False) precaution _save_output_value_snapshot
    takes, but this test passes with or without it on the pinned torch, so read it as
    coverage of the path, not as proof that the precaution is load-bearing.
    """
    _patch_store(monkeypatch, tmp_path)
    model = _packed_av_model()
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    samplers.FunPackLTXAVSceneChainSampler()._build_trajectory_probe_wrapper(model, rec)
    wrapper = model.model_options["model_function_wrapper"]

    with torch.inference_mode():
        latent = torch.arange(VIDEO_N + AUDIO_N, dtype=torch.float32).view(1, 1, -1)
        for k in range(4):
            wrapper(lambda x, t, **kw: latent,
                    {"input": latent, "timestep": torch.tensor([float(sigmas[k])]), "c": {}})

    assert [r["bucket"] for r in rec.cell_rows()] == [0, 1, 2, 3]
    assert probe.save_pending("k", rec, prompt_hash="p", seed=1, meta={"h3": True})
    assert probe.commit("k", 1.0) == 1
    assert len(probe.load_log("k")[0]["rows"]) == 4


def test_batched_prediction_is_refused_not_averaged():
    """At CFG>1 comfy stacks the unconditional row into the SAME call's batch, and the
    descriptor pools over batch — recording it would average a prediction with its own
    negation and report the run as cleanly measured."""
    model = _packed_av_model()
    sigmas = _h3_sigmas(4)
    rec = probe.TrajectoryRecorder(sigmas, n_buckets=4)
    samplers.FunPackLTXAVSceneChainSampler()._build_trajectory_probe_wrapper(model, rec)
    wrapper = model.model_options["model_function_wrapper"]

    cond = torch.full((1, 1, VIDEO_N + AUDIO_N), 10.0)
    uncond = torch.zeros(1, 1, VIDEO_N + AUDIO_N)
    batched = torch.cat([cond, uncond], dim=0)          # comfy's cond/uncond stacking
    out = wrapper(lambda x, t, **kw: batched,
                  {"input": batched, "timestep": torch.tensor([float(sigmas[0])]), "c": {}})

    assert torch.equal(out, batched), "the probe must observe, never steer"
    assert rec.is_empty(), "a cond/uncond blend must not be recorded as a measurement"
    assert rec._foreign == 1, "and the run has to be able to say it was not measured"


def test_a_step_with_no_timestep_is_counted_not_dropped():
    """Every path that cannot measure a step counts it, so a run can say the measurement is
    incomplete. A step with no timestep is one of those paths, not an exception to it."""
    model = _packed_av_model()
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    samplers.FunPackLTXAVSceneChainSampler()._build_trajectory_probe_wrapper(model, rec)
    latent = torch.zeros(1, 1, VIDEO_N + AUDIO_N)
    out = model.model_options["model_function_wrapper"](
        lambda x, t, **kw: latent, {"input": latent, "timestep": None, "c": {}})
    assert torch.equal(out, latent)
    assert rec.is_empty()
    assert rec._foreign == 1


def test_observer_survives_a_broken_prediction():
    """A probe failure must cost the measurement, never the generation."""
    model = _packed_av_model()
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    sampler = samplers.FunPackLTXAVSceneChainSampler()
    sampler._build_trajectory_probe_wrapper(model, rec)
    out = model.model_options["model_function_wrapper"](
        lambda x, t, **kw: "not a tensor",
        {"input": torch.zeros(1, 1, 4), "timestep": torch.tensor([0.9]), "c": {}})
    assert out == "not a tensor"
    assert rec.is_empty()
    # A step lost to an exception is as unmeasured as one refused on purpose, and has to be
    # counted, or a run that silently lost steps reads downstream as fully measured.
    assert rec._foreign == 1


def test_observer_chains_onto_an_existing_wrapper():
    """It installs outermost, over the steering stack — whatever was already there has to
    keep running, or the probe would silently disable steering it is meant to observe."""
    model = _packed_av_model()
    calls = []

    def _existing(apply_fn, args):
        calls.append("inner")
        return apply_fn(args["input"], args["timestep"], **args.get("c", {}))

    model.model_options["model_function_wrapper"] = _existing
    rec = probe.TrajectoryRecorder(_h3_sigmas(4), n_buckets=4)
    samplers.FunPackLTXAVSceneChainSampler()._build_trajectory_probe_wrapper(model, rec)
    latent = torch.zeros(1, 1, VIDEO_N + AUDIO_N)
    model.model_options["model_function_wrapper"](
        lambda x, t, **kw: latent,
        {"input": latent, "timestep": torch.tensor([0.9]), "c": {}})
    assert calls == ["inner"]


# ---------------------------------------------------------------------------
# The rating intake (conditioning.refine_v2)
# ---------------------------------------------------------------------------


def _refiner(tmp_path):
    from conditioning import FunPackVideoRefinerV2
    r = FunPackVideoRefinerV2()
    r._v2_state_path = lambda refinement_key: str(tmp_path / "state.json")
    return r


class _Clip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [(torch.ones(1, 4, 3), {"pooled_output": torch.ones(1, 3)})]


# The Movie Editor branch needs the run being rated to have been MULTI-SCENE, and the editor
# hands its own scene list in rather than relying on any delimiter in the prompt.
_SEGMENTS = {"anchor": "a woman", "scenes": ["walking through neon rain", "turning to camera"]}


def _rate(refiner, key, rating, me_ratings=None, segments=_SEGMENTS):
    return refiner.refine_v2("a woman walking through neon rain", _Clip(), rating, key,
                             split_by_transitions=True, scene_segments=segments,
                             movie_editor_scene_ratings=me_ratings)


def test_a_rating_logs_the_run_whichever_ui_rated_it(monkeypatch, tmp_path):
    """Multi-scene chains rated per scene from the Movie Editor go down a different branch
    of refine_v2 than a single overall rating. A run rated there is a labelled data point
    like any other — the probe that misses it collects nothing in the workflow that
    generates most of the runs."""
    _patch_store(monkeypatch, tmp_path / "probe")
    monkeypatch.setenv("FUNPACK_TRAJECTORY_PROBE", "1")
    sigmas = _h3_sigmas(4)

    refiner = _refiner(tmp_path)
    _rate(refiner, "key", "Perfect")                       # establishes a previous run

    probe.save_pending("key", _recorded(sigmas), prompt_hash="p", seed=1)
    _rate(refiner, "key", "Perfect")
    assert len(probe.load_log("key")) == 1, "a plain rating must log its run"

    # Asserted, not assumed: the editor branch has its own preconditions (a multi-scene
    # PREVIOUS run among them), and without them this test passes by quietly taking the
    # single-rating path it is supposed to be contrasting with.
    import conditioning
    entered = []
    original = conditioning.FunPackVideoRefinerV2._v2_apply_movie_editor_scene_ratings
    monkeypatch.setattr(conditioning.FunPackVideoRefinerV2,
                        "_v2_apply_movie_editor_scene_ratings",
                        lambda self, *a, **kw: (entered.append(1), original(self, *a, **kw))[1])

    probe.save_pending("key", _recorded(sigmas), prompt_hash="p", seed=2)
    _rate(refiner, "key", "Perfect",
          me_ratings=[{"index": 0, "rating": "Perfect"}, {"index": 1, "rating": "Awful"}])
    # "index" is the key the editor actually sends (movie_editor/server.py). With any other
    # key the aggregate lookup finds nothing and this silently becomes a second copy of the
    # single-rating case above.
    assert entered == [1], "the Movie Editor branch was never taken — this proves nothing"
    assert len(probe.load_log("key")) == 2, "a Movie Editor per-scene rating must log too"


def test_the_intake_is_silent_while_the_probe_is_off(monkeypatch, tmp_path):
    """With the probe off the sampler writes no pending, so consuming one would pair an
    older run's descriptors with this rating."""
    _patch_store(monkeypatch, tmp_path / "probe")
    monkeypatch.delenv("FUNPACK_TRAJECTORY_PROBE", raising=False)

    refiner = _refiner(tmp_path)
    _rate(refiner, "key", "Perfect")
    probe.save_pending("key", _recorded(_h3_sigmas(4)))
    _rate(refiner, "key", "Perfect")

    assert probe.load_log("key") == []
    assert os.path.exists(probe.pending_path("key")), "the pending must be left untouched"


def test_each_rated_scene_is_its_own_labelled_sample():
    """A multi-scene run rated per scene carries one reward PER SCENE. Collapsing to the
    run's aggregate would file a scene the user called Perfect as a bad sample whenever some
    other scene in the same run was rated badly — mixing the two labels the whole
    measurement is trying to tell apart."""
    row = {
        "reward": -0.9, "prompt_hash": "p",          # the aggregate: the worst scene wins
        "scene_rewards": {0: 1.0, 1: -0.85},
        "rows": [{"bucket": 0, "scene": 0, "pass": 0, "desc": torch.ones(4)},
                 {"bucket": 0, "scene": 1, "pass": 0, "desc": torch.zeros(4)}],
    }
    descriptors, labels, groups = probe.collect_bucket([row], 0)
    assert labels == [1, 0], "the Perfect scene must not be labelled bad"
    assert torch.equal(descriptors[0], torch.ones(4))
    assert torch.equal(descriptors[1], torch.zeros(4))
    assert groups == ["p", "p"]


def test_a_run_rated_once_is_still_one_sample():
    row = {
        "reward": 1.0, "prompt_hash": "p",
        "rows": [{"bucket": 0, "scene": 0, "pass": 0, "desc": torch.ones(4)},
                 {"bucket": 0, "scene": 1, "pass": 0, "desc": torch.zeros(4)}],
    }
    descriptors, labels, _ = probe.collect_bucket([row], 0)
    assert labels == [1]
    assert torch.allclose(descriptors[0], torch.full((4,), 0.5))  # scenes averaged


def test_per_scene_rewards_reach_the_log(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    probe.save_pending("k", _recorded(_h3_sigmas(4)))
    probe.commit("k", -0.9, rating_key="awful", scene_rewards={0: 1.0, 1: -0.85})
    assert probe.load_log("k")[0]["scene_rewards"] == {0: 1.0, 1: -0.85}


def test_the_switch_survives_a_restart(monkeypatch, tmp_path):
    """A rented box restarts. If the switch lived only in the environment, recording would
    stop and the panel would be the only place that ever said so."""
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.delenv("FUNPACK_TRAJECTORY_PROBE", raising=False)
    assert probe.probe_enabled() is False

    assert probe.set_probe_enabled(True) is True
    assert probe.probe_enabled() is True
    monkeypatch.delenv("FUNPACK_TRAJECTORY_PROBE")      # the restart
    assert probe.probe_enabled() is True, "the switch did not survive"

    probe.set_probe_enabled(False)
    monkeypatch.delenv("FUNPACK_TRAJECTORY_PROBE")
    assert probe.probe_enabled() is False, "turning it off did not survive either"


def test_the_environment_wins_while_it_is_set(monkeypatch, tmp_path):
    """The sampler runs in this process, so the live toggle has to beat the saved one."""
    _patch_store(monkeypatch, tmp_path)
    probe.set_probe_enabled(False)
    monkeypatch.setenv("FUNPACK_TRAJECTORY_PROBE", "1")
    assert probe.probe_enabled() is True


def test_the_switch_file_is_not_mistaken_for_a_log(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    probe.set_probe_enabled(True)
    probe.save_pending("k", _recorded(_h3_sigmas(4)))
    probe.commit("k", 1.0)
    assert sorted(k for k, _ in probe.load_all_logs()) == ["k"]


# ---------------------------------------------------------------------------
# Carrying the measurement between machines
# ---------------------------------------------------------------------------


def test_importing_the_same_export_twice_adds_nothing(monkeypatch, tmp_path):
    """A rental is replaced whenever something breaks, so an export gets carried onto a
    new box and may well be loaded twice. Appending blindly would double the sample, and
    the permutation test reads n directly — duplicated runs report a separation as far
    more significant than the evidence supports."""
    _patch_store(monkeypatch, tmp_path)
    sigmas = _h3_sigmas(4)
    for i in range(3):
        probe.save_pending("k", _recorded(sigmas, value=float(i)), prompt_hash="p", seed=i)
        probe.commit("k", 1.0 if i else -0.9)
    exported = [r for _key, rows in probe.load_all_logs() for r in rows]
    assert len(exported) == 3

    fresh = tmp_path / "other-box"
    monkeypatch.setattr(probe, "_probe_dir", lambda: str(fresh))
    assert probe.merge_rows("imported", exported) == (3, 3)
    assert probe.merge_rows("imported", exported) == (3, 0)


def test_two_boxes_exports_merge(monkeypatch, tmp_path):
    """The point of carrying it: runs from separate rentals add up to one sample."""
    _patch_store(monkeypatch, tmp_path)
    sigmas = _h3_sigmas(4)

    def runs(stamp, n):
        out = []
        for i in range(n):
            rec = _recorded(sigmas, value=float(i))
            out.append({"rows": rec.cell_rows(), "reward": 1.0, "stamp": stamp,
                        "seed": i, "prompt_hash": "p"})
        return out

    assert probe.merge_rows("imported", runs("box-a", 3)) == (3, 3)
    assert probe.merge_rows("imported", runs("box-b", 4)) == (7, 4)


def test_a_row_with_nothing_in_it_is_not_a_run(monkeypatch, tmp_path):
    _patch_store(monkeypatch, tmp_path)
    junk = [{}, {"rows": []}, {"rows": [{"bucket": 0}], "reward": "not a number"},
            "a string", None]
    assert probe.merge_rows("imported", junk) == (0, 0)
