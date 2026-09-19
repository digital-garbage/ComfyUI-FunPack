"""Quality sharpness on a sampler whose loop we do not own.

The unsharp mask that recovers fine detail lived inside Hybrid Euler 2S and Distilled Flow
because that is where it was written. It reads only the current x0 prediction, the previous
one, and the step's sigma — and sigma is an argument of every model call — so it lifts out
through a denoiser proxy, the same way ALG does, and a stock KSampler can have it too.

What these tests pin: the window is a fraction of the schedule (not a raw sigma), audio on a
packed AV latent is never sharpened, and asking for it never costs you the sampler you chose.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("folder_paths", types.SimpleNamespace(models_dir=""))

import comfy.samplers as _cs  # noqa: E402
import samplers  # noqa: E402
from conditioning import FunPackStudio  # noqa: E402


def _sampler(fn):
    return types.SimpleNamespace(sampler_function=fn, extra_options={}, inpaint_options={})


class _Model:
    """A denoiser that returns a fixed prediction per call, and records what it was given."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def __call__(self, x, sigma, **kwargs):
        self.calls.append(float(sigma.flatten()[0]) if hasattr(sigma, "flatten") else float(sigma))
        return self._outputs[min(len(self.calls) - 1, len(self._outputs) - 1)]


def _capture(model, sigmas, sharpness=0.5, start_pct=0.5):
    """Run the wrapped sampler over `sigmas`, returning what the proxy handed back each call."""
    seen = []

    def fn(m, x, sig, extra_args=None, callback=None, disable=None, **options):
        for s in sig[:-1]:
            seen.append(m(x, s.reshape(1)))
        return x

    wrapped = samplers._sharpen_wrap_sampler(_sampler(fn), sharpness, start_pct)
    wrapped.sampler_function(model, torch.zeros(1, 4), sigmas)
    return seen


# --- the wrapper's contract -----------------------------------------------------------


def test_off_returns_the_very_same_sampler():
    """0 must not wrap. A no-op proxy still costs a Python call per model eval, and it would
    also hide a sampler that cannot be wrapped behind a wrapper that appears to work."""
    s = _sampler(lambda *a, **kw: None)
    assert samplers._sharpen_wrap_sampler(s, 0.0, 0.35) is s
    assert samplers._sharpen_wrap_sampler(s, None, 0.35) is s


def test_unwrappable_sampler_reports_rather_than_pretending():
    """Not every SAMPLER producer is KSAMPLER-shaped. None tells the caller to say so."""
    assert samplers._sharpen_wrap_sampler(object(), 0.5, 0.35) is None


def test_wrapping_preserves_the_samplers_options():
    s = types.SimpleNamespace(sampler_function=lambda *a, **kw: None,
                              extra_options={"eta": 0.5}, inpaint_options={"foo": 1})
    w = samplers._sharpen_wrap_sampler(s, 0.5, 0.35)
    assert w.extra_options == {"eta": 0.5}
    assert w.inpaint_options == {"foo": 1}


# --- the window -----------------------------------------------------------------------


def test_only_the_late_steps_are_sharpened():
    """start_pct is a fraction of the schedule, converted to the sigma that starts it — the
    same derivation Hybrid Euler 2S uses for high_quality_pct, so the number means the same
    thing on both samplers."""
    preds = [torch.full((1, 4), float(v)) for v in (1.0, 2.0, 4.0, 8.0)]
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    out = _capture(_Model(preds), sigmas, sharpness=1.0, start_pct=0.5)

    # 4 steps, last 50% = the final 2 -> threshold is sigmas[2] = 0.5.
    assert out[0] is preds[0]                       # sigma 1.0, early: untouched
    assert out[1] is preds[1]                       # sigma 0.75, early: untouched
    # sigma 0.5: first sharpened call, prev = 2.0 -> 4 + 1.0 * 0.5 * (4 - 2) = 5
    assert float(out[2][0, 0]) == pytest.approx(5.0)


def test_first_sharpened_call_has_no_previous_and_passes_through():
    """The window's opening evaluation has nothing to high-pass against. It must return the
    prediction unchanged rather than treating a missing reference as zero."""
    preds = [torch.full((1, 4), 3.0), torch.full((1, 4), 7.0)]
    out = _capture(_Model(preds), torch.tensor([1.0, 0.5, 0.0]), sharpness=1.0, start_pct=1.0)
    assert float(out[0][0, 0]) == pytest.approx(3.0)


def test_start_pct_zero_never_fires():
    """0% is off, and it must be off by never sharpening — not by sharpening at sigma 0."""
    preds = [torch.full((1, 4), float(v)) for v in (1.0, 5.0, 9.0)]
    out = _capture(_Model(preds), torch.tensor([1.0, 0.5, 0.25, 0.0]),
                   sharpness=1.0, start_pct=0.0)
    assert [float(o[0, 0]) for o in out] == [1.0, 5.0, 9.0]


def test_previous_is_the_sharpened_result_so_the_boost_is_self_limiting():
    """The in-loop samplers set prev_denoised AFTER sharpening, which keeps the high-pass
    measured against an already-sharpened reference instead of compounding every step. The
    proxy matches that, or the same setting drifts between sampler types."""
    preds = [torch.full((1, 4), float(v)) for v in (0.0, 2.0, 4.0, 6.0)]
    out = _capture(_Model(preds), torch.tensor([1.0, 0.9, 0.8, 0.7, 0.0]),
                   sharpness=1.0, start_pct=1.0)
    # prev=0 -> 2 + 0.5*(2-0) = 3 ; prev=3 -> 4 + 0.5*(4-3) = 4.5 ; prev=4.5 -> 6 + 0.75 = 6.75
    assert [float(o[0, 0]) for o in out] == pytest.approx([0.0, 3.0, 4.5, 6.75])


# --- audio protection -----------------------------------------------------------------


def test_audio_stream_of_a_packed_latent_is_not_sharpened(monkeypatch):
    """On a packed [video | audio] latent the unsharp is a video-tuned perturbation; letting
    it reach the waveform is how the soundtrack degrades while the picture looks fine."""
    monkeypatch.setattr(
        samplers, "_get_latent_shapes",
        lambda model: [torch.Size([1, 2, 2, 2]), torch.Size([1, 2])], raising=False)

    preds = [torch.full((1, 1, 10), 1.0), torch.full((1, 1, 10), 3.0)]
    seen = []

    def fn(m, x, sig, extra_args=None, callback=None, disable=None, **options):
        for s in sig[:-1]:
            seen.append(m(x, s.reshape(1)))
        return x

    wrapped = samplers._sharpen_wrap_sampler(_sampler(fn), 1.0, 1.0)
    wrapped.sampler_function(_Model(preds), torch.zeros(1, 1, 10),
                             torch.tensor([1.0, 0.5, 0.0]))

    assert seen[1].shape == preds[1].shape               # the mask must not reshape the result
    assert float(seen[1][0, 0, 0]) == pytest.approx(4.0)  # video: 3 + 0.5*(3-1)
    assert float(seen[1][0, 0, 9]) == pytest.approx(3.0)  # audio: untouched


def test_single_stream_latent_sharpens_everything():
    """No packed layout means no audio to protect — the mask is None and must not be read as
    'mask everything out'."""
    preds = [torch.full((1, 4), 1.0), torch.full((1, 4), 3.0)]
    out = _capture(_Model(preds), torch.tensor([1.0, 0.5, 0.0]), sharpness=1.0, start_pct=1.0)
    assert float(out[1][0, 0]) == pytest.approx(4.0)


# --- reaching it from the Studio ------------------------------------------------------


@pytest.fixture
def comfy_sampler_stub(monkeypatch):
    def calculate_sigmas(model_sampling, scheduler_name, steps):
        return torch.linspace(1.0, 0.0, steps + 1)

    monkeypatch.setattr(_cs, "calculate_sigmas", calculate_sigmas, raising=False)
    monkeypatch.setattr(_cs, "sampler_object",
                        lambda name: _sampler(lambda *a, **kw: None), raising=False)


class _FakeModel:
    def get_model_object(self, name):
        return f"model_sampling::{name}"


def test_studio_wraps_the_ksampler_when_sharpness_is_set(comfy_sampler_stub):
    sampler, _ = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_name": "euler", "ksampler_steps": 8,
         "ksampler_scheduler": "karras", "ksampler_sharpness": 0.3},
        model=_FakeModel(),
    )
    assert sampler.sampler_function.__name__.endswith("_sharpen")


def test_studio_leaves_the_ksampler_alone_at_the_default(comfy_sampler_stub):
    """The default is off, and off must be the untouched stock sampler object."""
    sampler, _ = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_name": "euler", "ksampler_steps": 8,
         "ksampler_scheduler": "karras"},
        model=_FakeModel(),
    )
    assert not sampler.sampler_function.__name__.endswith("_sharpen")


def test_studio_keeps_sampling_when_the_sampler_cannot_be_wrapped(monkeypatch, capsys):
    """Asking for sharpening must never cost you the sampler you picked."""
    monkeypatch.setattr(_cs, "calculate_sigmas",
                        lambda ms, name, steps: torch.linspace(1.0, 0.0, steps + 1),
                        raising=False)
    monkeypatch.setattr(_cs, "sampler_object", lambda name: "SAMPLER::" + name, raising=False)

    sampler, _ = FunPackStudio._build_one_sampler(
        {"type": "KSampler", "ksampler_name": "euler", "ksampler_steps": 8,
         "ksampler_scheduler": "karras", "ksampler_sharpness": 0.3},
        model=_FakeModel(),
    )
    assert sampler == "SAMPLER::euler"
    assert "no sampler_function to wrap" in capsys.readouterr().out
