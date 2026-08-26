"""Assembling and driving a causal run: what it refuses, and what order it does things in.

The rollout arithmetic is tested in `test_causal_rollout.py`. What is tested here is the
wiring around it — which cache index each chunk writes to, that the prefix is cached exactly
once, and that a chunk's context is committed CLEAN rather than at the last noisy step.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_causal as hc  # noqa: E402

TEXT_LEN, LATENT_T, AUDIO_T, FRAME_ROWS = 40, 12, 64, 6


class _Layout:
    def __init__(self):
        self.segments = [(0, TEXT_LEN, "text"),
                         (TEXT_LEN, TEXT_LEN + AUDIO_T * 2, "audio"),
                         (TEXT_LEN + AUDIO_T * 2,
                          TEXT_LEN + AUDIO_T * 2 + LATENT_T * FRAME_ROWS, "video")]
        self.seq_len = self.segments[-1][1]
        self.position_ids = torch.zeros(self.seq_len, 3, dtype=torch.float64)


class _Diffusion:
    """Records what the rollout asked of it, and nothing else."""

    sigma_shift_video, sigma_shift_audio = 12.0, 3.0

    def __init__(self, layers=2):
        self.blocks = [object()] * layers
        self.calls = []

    def prefill_text(self, context, plan, cache, **kw):
        self.calls.append(("prefill", kw.get("video_sigma", 0.0)))
        cache.finish_chunk(0)

    def forward_chunk(self, video, audio, plan, index, cache, *, video_sigma, audio_sigma,
                      update_cache=False, **kw):
        self.calls.append(("commit" if update_cache else "forward", index,
                           round(float(video_sigma), 6), round(float(audio_sigma), 6)))
        if update_cache:
            for layer in range(len(self.blocks)):
                cache.write(layer, plan.cache_index(index), torch.zeros(1, 1, 1, 1),
                            torch.zeros(1, 1, 1, 1))
        return torch.zeros_like(video), torch.zeros_like(audio)


class _Samples:
    def __init__(self, video, audio):
        self._parts = (video, audio)

    def unbind(self):
        return self._parts


class _Patcher:
    load_device = torch.device("cpu")

    def __init__(self, diffusion, payload=None, fail=None):
        self.model = _Base(diffusion, payload, fail)
        self.model.diffusion_model = diffusion


class _Base:
    def __init__(self, diffusion, payload, fail):
        self.diffusion_model = diffusion
        self._payload = payload
        self._fail = fail

    def extra_conds(self, **kwargs):
        if self._fail:
            raise RuntimeError(self._fail)
        payload = dict(self._payload or {})
        payload.setdefault("layout", _Layout())
        return {"minimax_payload": _Cond(payload),
                "c_crossattn": _Cond(torch.zeros(1, TEXT_LEN, 8))}


class _Cond:
    def __init__(self, cond):
        self.cond = cond


def _latent(t=LATENT_T, batch=1):
    return {"samples": _Samples(torch.zeros(batch, 24, t, 8, 8),
                                torch.zeros(1, 32, 2, AUDIO_T))}


def _positive():
    return [[torch.zeros(1, TEXT_LEN, 8), {}]]


def _build(**kw):
    diffusion = kw.pop("diffusion", None) or _Diffusion()
    return hc.build_session(_Patcher(diffusion, kw.pop("payload", None), kw.pop("fail", None)),
                            _positive(), kw.pop("latent", None) or _latent(), **kw)


# ── what it refuses, and how it says so ─────────────────────────────────────

def test_a_stock_model_is_made_causal_rather_than_refused(monkeypatch):
    """Re-classing costs nothing and changes no weight, so a mode that is already switched ON
    must not refuse itself over a switch on another node. It does the work instead."""
    class Stock:
        blocks = [object()]

    made = []
    monkeypatch.setattr(hc, "make_causal",
                        lambda model: (made.append(model) or (True, "re-classed")))
    session, reason = _build(diffusion=Stock())
    assert made and session is not None, reason


def test_a_model_that_cannot_be_made_causal_says_what_it_actually_is():
    """The old message asserted the user had not switched something on, which was a guess —
    and a wrong one sends them looking in the wrong place."""
    class NotH3:
        blocks = [object()]

    session, reason = _build(diffusion=NotH3())
    assert session is None
    assert "NotH3" in reason


def test_a_model_with_no_blocks_is_refused():
    class Empty:
        blocks = []

        def forward_chunk(self):
            pass

    session, reason = _build(diffusion=Empty())
    assert session is None and "no DiT blocks" in reason


def test_a_batch_larger_than_one_is_refused():
    session, reason = _build(latent=_latent(batch=2))
    assert session is None and "one clip at a time" in reason


def test_a_latent_that_is_not_a_video_audio_pair_is_refused():
    session, reason = _build(latent={"samples": torch.zeros(1, 24, 4, 8, 8)})
    assert session is None and "causal lane can take" in reason


def test_a_clip_too_short_to_chunk_is_refused_rather_than_run():
    """One chunk means there is nothing for a cache to remember — running it would pay the
    lane's overhead for exactly none of its benefit."""
    session, reason = _build(latent=_latent(t=4))
    assert session is None and "nothing for a chunk cache to remember" in reason


def test_a_conditioning_the_model_cannot_prepare_is_reported_not_raised():
    session, reason = _build(fail="no keyframes here")
    assert session is None and "no keyframes here" in reason


def test_a_run_with_no_packed_layout_is_refused():
    session, reason = _build(payload={"layout": None})
    assert session is None and "no packed layout" in reason


def test_a_buildable_run_reports_no_reason():
    session, reason = _build()
    assert session is not None and reason == ""
    assert session["plan"].n_chunks == 3
    assert session["cache"].num_layers == 2


def test_the_cache_takes_the_sink_and_window_it_is_given():
    session, _ = _build(sink=1, window=None)
    assert session["cache"].sink == 1 and session["cache"].window is None


# ── the order things happen in ──────────────────────────────────────────────

def _run(**kw):
    diffusion = _Diffusion()
    session, reason = _build(diffusion=diffusion)
    assert session is not None, reason
    kw.setdefault("sigmas", [1.0, 0.5, 0.0])
    video, audio = hc.run_session(session, **kw)
    return diffusion, session, video, audio


def test_the_prefix_is_cached_once_for_the_whole_rollout():
    """Recomputing it per chunk would quietly make every later chunk's assumption false: that
    the text keys it attends are the ones the model saw."""
    diffusion, _, _, _ = _run()
    assert [c for c in diffusion.calls if c[0] == "prefill"] == [("prefill", 0.0)]


def test_every_chunk_is_forwarded_once_per_step_and_committed_once():
    diffusion, session, _, _ = _run()
    chunks = session["plan"].n_chunks
    forwards = [c for c in diffusion.calls if c[0] == "forward"]
    commits = [c for c in diffusion.calls if c[0] == "commit"]
    assert len(forwards) == chunks * 2          # two steps in the schedule
    assert [c[1] for c in commits] == list(range(chunks))


def test_a_chunk_is_committed_clean_not_at_its_last_noisy_step():
    """Caching the K/V of the last denoising step would hand every later chunk a context that
    still carries that step's noise."""
    diffusion, _, _, _ = _run()
    assert all(c[2] == 0.0 and c[3] == 0.0 for c in diffusion.calls if c[0] == "commit")


def test_the_audio_rows_are_given_their_own_sigma():
    """H3 denoises audio on its own shifted grid. Handing it the video's sigma is exactly the
    error `h3_audio_clock` exists to correct."""
    diffusion, _, _, _ = _run(sigmas=[0.8, 0.0])
    forward = next(c for c in diffusion.calls if c[0] == "forward")
    assert forward[3] != forward[2]


def test_the_commit_of_one_chunk_precedes_the_first_forward_of_the_next():
    """A chunk that is not committed before the next one starts is a chunk the rest of the
    clip cannot see."""
    diffusion, _, _, _ = _run()
    order = [c for c in diffusion.calls if c[0] in ("forward", "commit")]
    for position, call in enumerate(order):
        if call[0] == "forward" and call[1] > 0:
            assert ("commit", call[1] - 1, 0.0, 0.0) in order[:position]


def test_each_chunk_writes_to_its_own_cache_index_above_the_prefix():
    """The prompt holds 0 and the conditioning 1, so nothing a media chunk writes may land
    on either — overwriting the prompt's K/V would silently unprompt the rest of the clip."""
    _, session, _, _ = _run()
    written = {key[1] for key in session["cache"]._store}
    assert written <= set(range(2, session["plan"].n_chunks + 2))


def test_the_rollout_returns_a_clip_of_the_shape_it_was_given():
    _, session, video, audio = _run()
    assert tuple(video.shape) == session["video_shape"]
    assert tuple(audio.shape) == session["audio_shape"]


def test_a_seeded_run_reproduces_exactly():
    """The noise is drawn in a fixed order so a repeated seed gives the same clip; a skipped
    draw anywhere would shift the stream for every later chunk."""
    first = _run(seed=7)[2]
    second = _run(seed=7)[2]
    assert torch.equal(first, second)


def test_a_different_seed_gives_a_different_clip():
    assert not torch.equal(_run(seed=7)[2], _run(seed=8)[2])


def test_the_cache_starts_empty_even_if_the_session_is_reused():
    """A second scene through the same session must not inherit the first one's memory."""
    diffusion = _Diffusion()
    session, _ = _build(diffusion=diffusion)
    hc.run_session(session, sigmas=[1.0, 0.0])
    before = len(session["cache"]._store)
    hc.run_session(session, sigmas=[1.0, 0.0])
    assert len(session["cache"]._store) == before


def test_an_unknown_step_rule_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError):
        _run(step_rule="whatever")


# ── loading, and what loading also installs ─────────────────────────────────
#
# The ordinary path reaches the model through comfy.sample.sample_custom, whose
# prepare_sampling calls load_models_gpu. That call is ALSO what applies a patcher's object
# patches (partially_load -> patch_model), which is how FunPack's AdaLN modality gains and
# token-refiner edit get installed at all. This lane calls the DiT directly, so skipping the
# load would leave the weights offloaded AND both of those silently absent.

@pytest.fixture(autouse=True)
def helpers(monkeypatch):
    """A live comfy.sampler_helpers whose prepare_sampling is swappable PER TEST.

    Installed once and patched with monkeypatch afterwards: `install_module` merges onto a
    module object the whole session shares, so a test that installs a raising stub and does
    not restore it poisons every test that runs after it.
    """
    from tests._comfy_stubs import install_module

    module = install_module(
        "comfy.sampler_helpers",
        prepare_sampling=lambda *a, **kw: None,
        convert_cond=lambda cond: [dict(c[1], cross_attn=c[0]) for c in cond])

    def swap(fn):
        monkeypatch.setattr(module, "prepare_sampling", fn)

    monkeypatch.setattr(module, "prepare_sampling", lambda *a, **kw: None)
    return swap


@pytest.fixture
def loads(helpers):
    """Records every prepare_sampling call, the way ComfyUI's own would be made."""
    calls = []
    helpers(lambda *a, **kw: calls.append(a))
    return calls


def test_the_model_is_loaded_before_the_conditioning_is_prepared(helpers):
    """extra_conds runs H3's token refiner — a partial forward. Preparing it before the load
    is what produced `mat1 is on cuda:0, different from other tensors on cpu`."""
    order = []

    class Watched(_Base):
        def extra_conds(self, **kwargs):
            order.append("conds")
            return super().extra_conds(**kwargs)

    diffusion = _Diffusion()
    patcher = _Patcher(diffusion)
    patcher.model = Watched(diffusion, None, None)
    patcher.model.diffusion_model = diffusion
    helpers(lambda *a, **kw: order.append("load"))
    session, reason = hc.build_session(patcher, _positive(), _latent())
    assert session is not None, reason
    assert order[:2] == ["load", "conds"]


def test_the_load_is_given_the_clips_own_shape(loads):
    """prepare_sampling estimates memory from the noise shape; handing it the wrong one gets
    a placement decision made for a clip that is not this one."""
    _build()
    assert loads and loads[0][1] == (1, 24, LATENT_T, 8, 8)


def test_the_conditioning_is_converted_before_the_memory_estimate(loads):
    """estimate_memory calls extra_conds_shapes(**cond), so it wants ComfyUI's INTERNAL cond
    dicts — handing it the [tensor, meta] pairs a node receives raises, the load silently does
    not happen, and the failure resurfaces as a device mismatch three lines later."""
    _build()
    assert isinstance(loads[0][2]["positive"][0], dict)


def test_the_load_happens_before_the_first_forward(helpers):
    order = []
    diffusion = _Diffusion()
    session, reason = _build(diffusion=diffusion)
    assert session is not None, reason
    helpers(lambda *a, **kw: order.append("load"))
    real = diffusion.forward_chunk

    def spy(*a, **kw):
        order.append("forward")
        return real(*a, **kw)

    diffusion.forward_chunk = spy
    hc.run_session(session, sigmas=[1.0, 0.0])
    assert order[0] == "load"


def test_the_model_is_loaded_once_per_rollout_not_per_chunk(loads):
    """Loading per chunk would fight Comfy's own offload decision all the way down the clip."""
    session, reason = _build()
    assert session is not None, reason
    loads.clear()
    hc.run_session(session, sigmas=[1.0, 0.5, 0.0])
    assert len(loads) == 1


def _boom(*a, **kw):
    raise RuntimeError("out of memory")


def test_a_load_that_fails_at_build_time_refuses_the_lane(helpers):
    """Everything after the load assumes the model is resident, so a warning here surfaces
    three lines later as a device mismatch — one bug reported twice."""
    helpers(_boom)
    session, reason = _build()
    assert session is None
    assert "could not be placed" in reason and "out of memory" in reason


def test_a_load_that_fails_mid_rollout_does_not_abort_it(helpers, capsys):
    """By then the rollout is under way. It degrades to whatever Comfy already had resident
    and says so, rather than turning a recoverable placement problem into a dead run."""
    session, reason = _build()
    assert session is not None, reason
    helpers(_boom)
    video, _audio = hc.run_session(session, sigmas=[1.0, 0.0])
    assert video is not None
    assert "could not re-place the model" in capsys.readouterr().out


def test_no_patcher_means_no_load_attempt(loads):
    session, reason = _build()
    assert session is not None, reason
    session["patcher"] = None
    loads.clear()
    hc.run_session(session, sigmas=[1.0, 0.0])
    assert loads == []
