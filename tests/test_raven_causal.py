"""Chunk-causal rollout: RAVEN's sequence shaping, on FunPack's schedule and step rule.

RAVEN's contribution is not a sampler. It is a way of shaping the sequence — cut the clip into
time chunks, carry each to completion, commit it into a KV cache as clean context, generate the
next one attending to that cache. The model gets real memory of what it already drew.

That shaping is separable from the 4-step consistency sampler it ships with, and these tests
pin the separation: the chunking is theirs, the schedule is ours, and the step rule is a knob
so a chunked run can be compared against an unchunked one without two variables moving.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import raven_causal as rc


# ── the chunk cut ───────────────────────────────────────────────────────────

def test_video_cuts_every_five_latents():
    bounds = rc.chunk_bounds(latent_t=12, audio_t=100)
    assert [(a, b) for a, b, _c, _d in bounds] == [(0, 5), (5, 10), (10, 12)]


def test_the_tail_chunk_is_shorter_not_dropped():
    """H3's grid is 5k+2 latents, so there is ALWAYS a 2-latent tail. Dropping it would cut
    the last five frames off every clip."""
    bounds = rc.chunk_bounds(latent_t=17, audio_t=140)
    assert bounds[-1][0], bounds[-1][1] == (15, 17)
    assert bounds[-1][1] - bounds[-1][0] == 2


def test_every_video_latent_lands_in_exactly_one_chunk():
    bounds = rc.chunk_bounds(latent_t=22, audio_t=180)
    covered = []
    for start, stop, _a, _b in bounds:
        covered.extend(range(start, stop))
    assert covered == list(range(22))


def test_every_audio_latent_lands_in_exactly_one_chunk():
    bounds = rc.chunk_bounds(latent_t=22, audio_t=180)
    covered = []
    for _s, _e, start, stop in bounds:
        covered.extend(range(start, stop))
    assert covered == list(range(180))


def test_audio_boundaries_come_from_the_clock_not_a_cadence():
    """40 audio latents/s against a fixed 24 fps, 17 frames per chunk. The 28/29 alternation
    is a CONSEQUENCE of that ratio — hard-coding it would silently drift on a changed grid."""
    bounds = rc.chunk_bounds(latent_t=17, audio_t=int(round((17 * 3 + 5) / 24 * 40)))
    assert bounds[0][3] == round(17 * 40 / 24)
    assert bounds[1][3] == round(34 * 40 / 24)


def test_audio_boundaries_never_go_backwards():
    for latent_t in (2, 7, 12, 22, 57):
        audio_t = int(round((17 * ((latent_t - 2) // 5) + 5) / 24 * 40))
        bounds = rc.chunk_bounds(latent_t, audio_t)
        cursors = [b[2] for b in bounds] + [bounds[-1][3]]
        assert cursors == sorted(cursors)


def test_a_single_chunk_clip_is_one_chunk():
    assert len(rc.chunk_bounds(latent_t=2, audio_t=8)) == 1


def test_an_empty_clip_plans_nothing():
    assert rc.chunk_bounds(latent_t=0, audio_t=0) == []


# ── the step rules ──────────────────────────────────────────────────────────

def test_consistency_jumps_to_x0_and_renoises_with_fresh_noise():
    """RAVEN's own transition, and the one its LoRA is distilled for."""
    x0 = torch.full((4,), 2.0)
    noise = torch.full((4,), 1.0)
    out = rc.step("consistency", torch.zeros(4), x0, sigma=0.8, sigma_next=0.5, noise=noise)
    assert torch.allclose(out, 0.5 * x0 + 0.5 * noise)


def test_consistency_at_the_last_step_is_exactly_x0():
    x0 = torch.full((4,), 3.0)
    out = rc.step("consistency", torch.ones(4), x0, sigma=0.3, sigma_next=0.0,
                  noise=torch.full((4,), 9.0))
    assert torch.allclose(out, x0)


def test_consistency_ignores_the_noise_already_in_x_t():
    """Fresh noise every step is the point — reusing x_t's noise would correlate the steps."""
    x0 = torch.full((4,), 2.0)
    a = rc.step("consistency", torch.zeros(4), x0, 0.8, 0.5, torch.full((4,), 1.0))
    b = rc.step("consistency", torch.full((4,), 99.0), x0, 0.8, 0.5, torch.full((4,), 1.0))
    assert torch.allclose(a, b)


def test_euler_is_the_ordinary_flow_step():
    x_t = torch.full((4,), 1.0)
    x0 = torch.full((4,), 0.0)
    out = rc.step("euler", x_t, x0, sigma=1.0, sigma_next=0.5, noise=torch.zeros(4))
    # derivative = (x_t - x0)/sigma = 1; x + (0.5 - 1.0) * 1 = 0.5
    assert torch.allclose(out, torch.full((4,), 0.5))


def test_euler_reaches_x0_exactly_at_sigma_zero():
    x_t = torch.full((4,), 1.0)
    x0 = torch.full((4,), 0.25)
    out = rc.step("euler", x_t, x0, sigma=0.7, sigma_next=0.0, noise=torch.zeros(4))
    assert torch.allclose(out, x0, atol=1e-6)


def test_euler_ignores_the_noise_it_is_handed():
    x_t = torch.full((4,), 1.0)
    x0 = torch.zeros(4)
    a = rc.step("euler", x_t, x0, 1.0, 0.5, torch.zeros(4))
    b = rc.step("euler", x_t, x0, 1.0, 0.5, torch.full((4,), 50.0))
    assert torch.allclose(a, b)


def test_ancestral_with_zero_eta_is_euler():
    x_t, x0 = torch.full((4,), 1.0), torch.zeros(4)
    a = rc.step("euler_ancestral", x_t, x0, 1.0, 0.5, torch.full((4,), 3.0), eta=0.0)
    b = rc.step("euler", x_t, x0, 1.0, 0.5, torch.zeros(4))
    assert torch.allclose(a, b)


def test_ancestral_adds_noise_when_eta_is_positive():
    x_t, x0 = torch.full((4,), 1.0), torch.zeros(4)
    a = rc.step("euler_ancestral", x_t, x0, 1.0, 0.5, torch.full((4,), 3.0), eta=1.0)
    b = rc.step("euler", x_t, x0, 1.0, 0.5, torch.zeros(4))
    assert not torch.allclose(a, b)


def test_ancestral_uses_the_rectified_flow_form_not_the_vp_one():
    """The VP formula is wrong on a flow model — it was a real bug here before. At eta=1 the
    step goes all the way down to sigma_down = sigma_next**2/sigma, not to sigma_next."""
    x_t, x0 = torch.full((4,), 1.0), torch.zeros(4)
    sigma, sigma_next, eta = 1.0, 0.5, 1.0
    out = rc.step("euler_ancestral", x_t, x0, sigma, sigma_next, torch.zeros(4), eta=eta)
    down_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    expected = x_t + (sigma_next * down_ratio - sigma) * ((x_t - x0) / sigma)
    assert torch.allclose(out, expected)


def test_ancestral_at_the_last_step_does_not_renoise():
    """Noise added at sigma_next == 0 would land in the output."""
    x_t, x0 = torch.full((4,), 1.0), torch.full((4,), 0.25)
    out = rc.step("euler_ancestral", x_t, x0, 0.7, 0.0, torch.full((4,), 99.0), eta=1.0)
    assert torch.allclose(out, x0, atol=1e-6)


def test_an_unknown_rule_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unknown step rule"):
        rc.step("dpmpp_2m", torch.zeros(4), torch.zeros(4), 1.0, 0.5, torch.zeros(4))


def test_every_advertised_rule_actually_runs():
    for rule in rc.STEP_RULES:
        out = rc.step(rule, torch.ones(4), torch.zeros(4), 1.0, 0.5, torch.zeros(4))
        assert out.shape == (4,)


# ── the package probe ───────────────────────────────────────────────────────

def test_a_missing_package_names_what_to_install(monkeypatch):
    monkeypatch.setitem(rc._PROBE, "state", None)
    monkeypatch.setattr(rc.importlib, "import_module",
                        lambda name: (_ for _ in ()).throw(ImportError(name)))
    monkeypatch.setattr(rc.os.path, "isdir", lambda p: False)
    module, reason = rc.locate_raven()
    assert module is None
    assert "RAVEN-Streaming" in reason and "custom_nodes" in reason


def test_the_probe_is_only_paid_once(monkeypatch):
    monkeypatch.setitem(rc._PROBE, "state", ("sentinel", ""))
    assert rc.locate_raven() == ("sentinel", "")


# ── the rollout ─────────────────────────────────────────────────────────────
#
# Everything the model does arrives as a callable, so the loop runs here with no ComfyUI, no
# RAVEN package and no weights. What is being tested is the ORDER of things: which chunks are
# sampled, which are taken as given, what gets committed to the cache, and that a seeded run
# draws its noise in a fixed sequence.

class Recorder:
    def __init__(self, latent_t=12, audio_t=100, channels=2):
        self.chunks = rc.chunk_bounds(latent_t, audio_t)
        self.video_noise = torch.zeros(1, channels, latent_t, 2, 2)
        self.audio_noise = torch.zeros(1, channels, 2, audio_t)
        self.forwards = []
        self.commits = []
        self.draws = 0

    def forward(self, video_xt, audio_xt, index, sigma):
        self.forwards.append((index, round(float(sigma), 6)))
        return torch.zeros_like(video_xt), torch.zeros_like(audio_xt)

    def commit(self, video_x0, audio_x0, index):
        self.commits.append((index, tuple(video_x0.shape), tuple(audio_x0.shape)))

    def draw(self, shape):
        self.draws += 1
        return torch.zeros(shape)

    def run(self, sigmas=(1.0, 0.5, 0.0), **kw):
        return rc.causal_rollout(
            chunks=self.chunks, sigmas=sigmas, forward=self.forward, commit=self.commit,
            draw_noise=self.draw, video_noise=self.video_noise,
            audio_noise=self.audio_noise, **kw)


def test_every_chunk_is_committed_to_the_cache():
    """A chunk that is not committed is a chunk the rest of the clip cannot see — which is
    the whole reason to chunk in the first place."""
    r = Recorder()
    r.run()
    assert [c[0] for c in r.commits] == list(range(len(r.chunks)))


def test_a_chunk_is_committed_once_not_once_per_step():
    r = Recorder()
    r.run(sigmas=(1.0, 0.8, 0.5, 0.2, 0.0))
    assert len(r.commits) == len(r.chunks)


def test_chunks_are_generated_in_order():
    """Chunk i+1 attends to chunk i's cache, so out-of-order would read an empty cache."""
    r = Recorder()
    r.run()
    assert [i for i, _s in r.forwards] == sorted(i for i, _s in r.forwards)


def test_the_schedule_is_ours_not_a_fixed_four():
    """Any scheduler, any step count — the distilled 4 is a default, not a constraint."""
    r = Recorder()
    r.run(sigmas=(1.0, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10, 0.0))
    per_chunk = [s for i, s in r.forwards if i == 0]
    assert per_chunk == [1.0, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10]


def test_every_step_sees_its_own_sigma():
    r = Recorder()
    r.run(sigmas=(1.0, 0.5, 0.0))
    assert [s for i, s in r.forwards if i == 1] == [1.0, 0.5]


def test_noise_is_drawn_for_both_streams_every_step():
    """Even at sigma_next == 0, where it is multiplied away: skipping the draw would shift
    the stream for every later chunk and a seeded run would stop reproducing."""
    r = Recorder()
    r.run(sigmas=(1.0, 0.5, 0.0))
    assert r.draws == len(r.chunks) * 2 * 2


def test_the_output_covers_the_whole_clip():
    r = Recorder()
    video, audio = r.run()
    assert video.shape == r.video_noise.shape
    assert audio.shape == r.audio_noise.shape


def test_the_initial_noise_is_not_mutated():
    r = Recorder()
    before = r.video_noise.clone()
    r.run()
    assert torch.allclose(r.video_noise, before)


# ── the pre-known first chunk: how an i2v anchor survives ───────────────────

def test_a_known_chunk_is_not_resampled():
    """Chunk 0 comes from FunPack's ordinary dense path, where pins and references work. The
    causal lane is never asked to model conditioning rows it has no layout for."""
    r = Recorder()
    r.run(known_chunks=1,
          known_video=torch.full_like(r.video_noise, 7.0),
          known_audio=torch.full_like(r.audio_noise, 7.0))
    assert 0 not in [i for i, _s in r.forwards]


def test_a_known_chunk_is_still_committed():
    """The anchor chunk must reach the cache or the rest of the clip continues from a clip
    that appears to start after it."""
    r = Recorder()
    r.run(known_chunks=1,
          known_video=torch.full_like(r.video_noise, 7.0),
          known_audio=torch.full_like(r.audio_noise, 7.0))
    assert r.commits[0][0] == 0


def test_the_known_chunk_reaches_the_output_verbatim():
    r = Recorder()
    known_v = torch.full_like(r.video_noise, 7.0)
    video, _audio = r.run(known_chunks=1, known_video=known_v,
                          known_audio=torch.full_like(r.audio_noise, 7.0))
    v_start, v_stop = r.chunks[0][0], r.chunks[0][1]
    assert torch.allclose(video[:, :, v_start:v_stop], known_v[:, :, v_start:v_stop])


def test_later_chunks_are_still_sampled_after_a_known_one():
    r = Recorder()
    r.run(known_chunks=1,
          known_video=torch.full_like(r.video_noise, 7.0),
          known_audio=torch.full_like(r.audio_noise, 7.0))
    assert sorted({i for i, _s in r.forwards}) == list(range(1, len(r.chunks)))


def test_known_chunks_without_latents_is_refused():
    """Silently sampling over the anchor would look like the anchor was ignored."""
    r = Recorder()
    with pytest.raises(ValueError, match="known_chunks"):
        r.run(known_chunks=1)


# ── refusals and cancellation ───────────────────────────────────────────────

def test_an_unknown_step_rule_is_refused_before_any_forward():
    r = Recorder()
    with pytest.raises(ValueError, match="unknown step rule"):
        r.run(step_rule="dpmpp_3m")
    assert r.forwards == []


def test_a_schedule_with_no_steps_is_refused():
    r = Recorder()
    with pytest.raises(ValueError, match="at least one step"):
        r.run(sigmas=(1.0,))


def test_cancellation_is_checked_before_each_chunk():
    r = Recorder()
    seen = []

    def cancel(index):
        seen.append(index)
        if index == 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        r.run(cancel=cancel)
    assert seen == [0, 1]


def test_the_chunk_callback_reports_whether_it_was_generated():
    """A preview lane needs to know which chunks are new; a pre-known one has nothing to show."""
    r = Recorder()
    events = []
    r.run(known_chunks=1,
          known_video=torch.full_like(r.video_noise, 7.0),
          known_audio=torch.full_like(r.audio_noise, 7.0),
          on_chunk=lambda i, v, a, was_known: events.append((i, was_known)))
    assert events[0] == (0, True)
    assert all(not known for _i, known in events[1:])


# ── the bridge to the RAVEN package ─────────────────────────────────────────
#
# build_session assembles their contracts/layout/cache/causal model; run_session drives it
# with OUR schedule and step rule. Tested against a fake package, because the real one needs
# weights and a GPU — what is being pinned is the wiring, not the numerics.

class FakeLayout:
    def __init__(self, latent_t=12, audio_t=100, latent_h=4, latent_w=4):
        self.latent_t, self.audio_t = latent_t, audio_t
        self.latent_h, self.latent_w = latent_h, latent_w
        bounds = rc.chunk_bounds(latent_t, audio_t)
        self.chunks = [types.SimpleNamespace(index=i, video_start=a, video_stop=b,
                                             audio_start=c, audio_stop=d)
                       for i, (a, b, c, d) in enumerate(bounds)]

    def video_latent_shape(self, dim=24):
        return (1, dim, self.latent_t, self.latent_h, self.latent_w)

    def audio_latent_shape(self, dim=32):
        return (1, dim, 2, self.audio_t)


class FakeCausalModel:
    dtype = torch.float32
    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0

    def __init__(self):
        self.blocks = [object()] * 4
        self.prefills = 0
        self.forwards = []
        self.commits = []

    def prefill_text(self, context, **kw):
        self.prefills += 1
        return int(context.shape[1])

    def forward_chunk(self, *, video_latent, audio_latent, chunk_index, role,
                      video_sigma=None, audio_sigma=None, update_cache=None, **kw):
        if role == "clean":
            self.commits.append(chunk_index)
        else:
            self.forwards.append((chunk_index, video_sigma, audio_sigma))
        return [torch.zeros_like(video_latent), torch.zeros_like(audio_latent)]


def _session(model=None, layout=None):
    model = model or FakeCausalModel()
    return {
        "module": None, "model": model, "patcher": None,
        "conditioning": types.SimpleNamespace(
            cross_attn=torch.zeros(1, 7, 16), token_tags=None, text_len=7),
        "request": None, "layout": layout or FakeLayout(),
        "cache": object(), "device": torch.device("cpu"),
        "compute_dtype": torch.float32,
        "video_channels": 24, "audio_channels": 32,
        "shift_video": 12.0, "shift_audio": 3.0,
    }


def test_the_text_is_prefilled_exactly_once():
    """Text is cache chunk 0 and is written alone: folding it into the first media chunk
    would let text rows attend media rows, and every later chunk assumes otherwise."""
    model = FakeCausalModel()
    rc.run_session(_session(model), sigmas=(1.0, 0.5, 0.0))
    assert model.prefills == 1


def test_the_audio_stream_gets_its_own_sigma():
    """H3 denoises audio on a shift of 3 against the video's 12. Handing the audio rows the
    video's sigma is the error h3_audio_clock exists to correct."""
    model = FakeCausalModel()
    rc.run_session(_session(model), sigmas=(1.0, 0.83, 0.0))
    video_sigmas = [v for _i, v, _a in model.forwards]
    audio_sigmas = [a for _i, _v, a in model.forwards]
    assert 0.83 in video_sigmas
    assert not any(abs(a - 0.83) < 1e-9 for a in audio_sigmas)
    assert any(abs(a - rc.audio_sigma_for(0.83, 12.0, 3.0)) < 1e-9 for a in audio_sigmas)


def test_the_audio_remap_matches_comfys_own_time_shift():
    """Pinned against upstream's formula so a drift in ours is caught here rather than as
    quiet audio distortion.

    Transcribed rather than imported: the suite stubs `comfy`, so importing the real module
    is not possible here. This is `time_shift_sigma` from comfy/ldm/minimax/model.py verbatim
    — invert to the base grid, re-apply the other shift. Verified equal to the live function
    across this range when run against the real checkout.
    """
    def upstream(sigma, from_shift, to_shift):
        base = sigma / (from_shift + sigma * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)

    for sigma in (1.0, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10, 0.01):
        assert abs(rc.audio_sigma_for(sigma, 12.0, 3.0)
                   - upstream(sigma, 12.0, 3.0)) < 1e-9


def test_equal_shifts_leave_the_sigma_alone():
    assert rc.audio_sigma_for(0.7, 12.0, 12.0) == 0.7


def test_sigma_zero_stays_zero():
    """A remapped zero would leave the last step short of clean."""
    assert rc.audio_sigma_for(0.0, 12.0, 3.0) == 0.0


def test_every_chunk_is_committed_through_the_clean_role():
    model = FakeCausalModel()
    session = _session(model)
    rc.run_session(session, sigmas=(1.0, 0.5, 0.0))
    assert model.commits == [c.index for c in session["layout"].chunks]


def test_the_noise_shapes_come_from_the_layout():
    model = FakeCausalModel()
    layout = FakeLayout(latent_t=7, audio_t=37)
    video, audio = rc.run_session(_session(model, layout), sigmas=(1.0, 0.0))
    assert tuple(video.shape) == layout.video_latent_shape(24)
    assert tuple(audio.shape) == layout.audio_latent_shape(32)


def test_the_same_seed_gives_the_same_clip():
    a = rc.run_session(_session(), sigmas=(1.0, 0.5, 0.0), seed=7)[0]
    b = rc.run_session(_session(), sigmas=(1.0, 0.5, 0.0), seed=7)[0]
    assert torch.allclose(a, b)


def test_a_different_seed_gives_a_different_clip():
    a = rc.run_session(_session(), sigmas=(1.0, 0.5, 0.0), seed=7)[0]
    b = rc.run_session(_session(), sigmas=(1.0, 0.5, 0.0), seed=8)[0]
    assert not torch.allclose(a, b)


def test_our_schedule_reaches_the_model_unchanged():
    model = FakeCausalModel()
    rc.run_session(_session(model), sigmas=(1.0, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10, 0.0))
    first = [v for i, v, _a in model.forwards if i == 0]
    assert first == [1.0, 0.94, 0.83, 0.72, 0.55, 0.30, 0.10]


def test_a_stock_bidirectional_model_is_refused_by_name(monkeypatch):
    """The causal lane needs the chunk-causal DiT; a stock H3 has no forward_chunk and would
    otherwise fail somewhere deep in the rollout instead of at the door."""
    monkeypatch.setitem(rc._PROBE, "state", (types.SimpleNamespace(), ""))
    fake_contracts = types.SimpleNamespace(
        resolve_model=lambda m, **k: types.SimpleNamespace(
            diffusion_model=types.SimpleNamespace(), patcher=None, load_device="cpu"),
        parse_conditioning=lambda p: types.SimpleNamespace(text_len=7, token_tags=None),
        parse_latent=lambda l, **k: types.SimpleNamespace(
            layout=lambda *a, **k: FakeLayout(), device="cpu", dtype=torch.float32),
    )
    # The parent module has to be registered too, and the submodules attached to it as
    # ATTRIBUTES: `from raven_streaming import cache` resolves through the parent, so a
    # sys.modules entry alone leaves the import reading nothing. Same trap the suite's own
    # comfy stubs were built to avoid.
    fake_cache = types.SimpleNamespace(ChunkKVCache=lambda *a, **k: object())
    parent = types.ModuleType("raven_streaming")
    parent.contracts, parent.cache = fake_contracts, fake_cache
    monkeypatch.setitem(sys.modules, "raven_streaming", parent)
    monkeypatch.setitem(sys.modules, "raven_streaming.contracts", fake_contracts)
    monkeypatch.setitem(sys.modules, "raven_streaming.cache", fake_cache)
    session, reason = rc.build_session(object(), [], {})
    assert session is None
    assert "chunk-causal DiT" in reason and "RAVEN Model Loader" in reason


# ── the Chain Sampler dispatch ──────────────────────────────────────────────
#
# The causal lane REPLACES the sampler rather than wrapping it, so the thing that matters is
# that it falls through — loudly — whenever it cannot run. A mode that is silently ignored
# looks exactly like a mode that ran and did nothing.

def _sampler(**attrs):
    import samplers
    node = samplers.FunPackLTXAVSceneChainSampler.__new__(
        samplers.FunPackLTXAVSceneChainSampler)
    for k, v in attrs.items():
        setattr(node, k, v)
    return node


def test_the_mode_is_off_by_default():
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert fields["h3_causal_chunks"][1]["default"] is False


def test_the_widget_offers_exactly_the_rules_the_loop_implements():
    """A widget listing a rule the loop refuses would be a crash three minutes into a run."""
    import samplers
    spec = samplers.FunPackLTXAVSceneChainSampler.INPUT_TYPES()
    fields = {**spec.get("required", {}), **spec.get("optional", {})}
    assert tuple(fields["h3_causal_step_rule"][0]) == rc.STEP_RULES


def test_off_means_the_normal_sampler_runs():
    node = _sampler(_h3_causal_chunks=False)
    assert node._run_h3_causal(object(), [], object(), [1.0, 0.0], 0) is None


def test_a_non_h3_model_falls_through_and_says_so(monkeypatch, capsys):
    import samplers
    monkeypatch.setattr(samplers._log, "note_on_change",
                        lambda tag, who, msg: print(msg), raising=False)
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda m: False)
    node = _sampler(_h3_causal_chunks=True)
    assert node._run_h3_causal(object(), [], object(), [1.0, 0.0], 0) is None
    assert "not a MiniMax H3 model" in capsys.readouterr().out


def test_a_missing_raven_package_falls_through_and_says_why(monkeypatch, capsys):
    import samplers
    monkeypatch.setattr(samplers._log, "note_on_change",
                        lambda tag, who, msg: print(msg), raising=False)
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda m: True)
    monkeypatch.setattr(rc, "build_session",
                        lambda *a, **k: (None, "the pack is not installed."))
    node = _sampler(_h3_causal_chunks=True, _h3_causal_sink=2, _h3_causal_window=2)
    assert node._run_h3_causal(object(), [], object(), [1.0, 0.0], 0) is None
    out = capsys.readouterr().out
    assert "not installed" in out and "samples normally" in out


def test_a_rollout_failure_is_raised_not_swallowed(monkeypatch):
    """Falling back to ordinary sampling after a partial rollout would return a clip the user
    did not ask for, under a mode they think ran."""
    import samplers
    monkeypatch.setattr(samplers._log, "note_on_change", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(samplers._log, "failed", lambda *a, **k: None, raising=False)
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda m: True)
    monkeypatch.setattr(rc, "build_session",
                        lambda *a, **k: ({"layout": FakeLayout()}, ""))
    monkeypatch.setattr(rc, "run_session",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cache overflow")))
    node = _sampler(_h3_causal_chunks=True, _h3_causal_sink=2, _h3_causal_window=2,
                    _h3_causal_step_rule="consistency")
    with pytest.raises(RuntimeError, match="cache overflow"):
        node._run_h3_causal(object(), [], object(), [1.0, 0.0], 0)


def test_the_run_says_the_sampler_and_cfg_are_not_used(monkeypatch, capsys):
    """Both are wired on the node and ignored on this path. Dropping them quietly would be
    the kind of silent no-op this project logs against."""
    import samplers
    monkeypatch.setattr(samplers._log, "note_on_change",
                        lambda tag, who, msg: print(msg), raising=False)
    import minimax_h3
    monkeypatch.setattr(minimax_h3, "is_h3_model", lambda m: True)
    layout = FakeLayout()
    # the suite's comfy stub has NestedTensor as a bare object; the packer needs to construct one
    import comfy.nested_tensor
    monkeypatch.setattr(comfy.nested_tensor, "NestedTensor", lambda pair: tuple(pair),
                        raising=False)
    monkeypatch.setattr(rc, "build_session", lambda *a, **k: ({"layout": layout}, ""))
    monkeypatch.setattr(rc, "run_session",
                        lambda *a, **k: (torch.zeros(1, 24, 12, 4, 4), torch.zeros(1, 32, 2, 100)))
    node = _sampler(_h3_causal_chunks=True, _h3_causal_sink=2, _h3_causal_window=2,
                    _h3_causal_step_rule="consistency")
    node._run_h3_causal(object(), [], torch.zeros(1), [1.0, 0.5, 0.0], 0)
    out = capsys.readouterr().out
    assert "CFG" in out and "not used" in out.replace("NOT used", "not used")
    assert "no per-chunk redo" in out.lower()


# ── loading, and what loading also installs ─────────────────────────────────
#
# The ordinary path reaches the model through comfy.sample.sample_custom, whose
# prepare_sampling calls load_models_gpu. That call is ALSO what applies a patcher's object
# patches (partially_load -> patch_model), which is how FunPack's AdaLN modality gains and
# token-refiner edit get installed at all. This lane calls the DiT directly, so skipping the
# load would leave the weights offloaded AND both of those silently absent.

def test_the_model_is_loaded_before_the_rollout(monkeypatch):
    import comfy.model_management
    calls = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu",
                        lambda models, **kw: calls.append((models, kw)), raising=False)
    model = FakeCausalModel()
    session = _session(model)
    session["patcher"] = "the-patcher"
    rc.run_session(session, sigmas=(1.0, 0.0))
    assert calls and calls[0][0] == ["the-patcher"]


def test_the_load_happens_before_the_first_forward(monkeypatch):
    import comfy.model_management
    order = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu",
                        lambda models, **kw: order.append("load"), raising=False)
    model = FakeCausalModel()

    real_forward = model.forward_chunk
    def spy(**kw):
        order.append("forward")
        return real_forward(**kw)
    model.forward_chunk = spy

    session = _session(model)
    session["patcher"] = "the-patcher"
    rc.run_session(session, sigmas=(1.0, 0.0))
    assert order[0] == "load"


def test_the_model_is_loaded_once_not_per_chunk(monkeypatch):
    """Loading per chunk would fight Comfy's own offload decision all the way down the clip."""
    import comfy.model_management
    calls = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu",
                        lambda models, **kw: calls.append(models), raising=False)
    session = _session()
    session["patcher"] = "the-patcher"
    rc.run_session(session, sigmas=(1.0, 0.5, 0.0))
    assert len(calls) == 1


def test_a_failed_load_does_not_abort_the_run(monkeypatch, capsys):
    """It degrades to whatever Comfy already had resident, and says the patches may be
    missing — rather than turning a recoverable placement problem into a dead run."""
    import comfy.model_management
    monkeypatch.setattr(
        comfy.model_management, "load_models_gpu",
        lambda models, **kw: (_ for _ in ()).throw(RuntimeError("out of memory")),
        raising=False)
    session = _session()
    session["patcher"] = "the-patcher"
    video, _audio = rc.run_session(session, sigmas=(1.0, 0.0))
    assert video is not None
    assert "object patches may not be installed" in capsys.readouterr().out


def test_no_patcher_means_no_load_attempt(monkeypatch):
    import comfy.model_management
    calls = []
    monkeypatch.setattr(comfy.model_management, "load_models_gpu",
                        lambda models, **kw: calls.append(models), raising=False)
    rc.run_session(_session(), sigmas=(1.0, 0.0))     # _session has no patcher
    assert calls == []
