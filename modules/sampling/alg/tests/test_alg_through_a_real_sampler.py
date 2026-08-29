"""ALG driven through ComfyUI's real KSAMPLER.

The stand-in is the sampler FUNCTION -- the innermost loop, which is the only
part that needs a GPU. Everything above it is ComfyUI's own: KSAMPLER.sample,
KSamplerX0Inpaint, the wrapper machinery. So "the denoiser sees a different
anchor early than it does late" is demonstrated rather than asserted.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


class _Guider:
    """The minimum ComfyUI's real KSAMPLER.sample needs of a guider.

    Only the innermost model call is stubbed; KSAMPLER.sample, KSamplerX0Inpaint
    and the noise scaling around them are the real ones.
    """

    def __init__(self):
        outer = self

        class ModelSampling:
            sigma_max = 1.0

            def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
                return noise

            def inverse_noise_scaling(self, sigma, x):
                return x

        class InnerModel:
            model_sampling = ModelSampling()

            def scale_latent_inpaint(self, x, sigma, noise, latent_image, denoise_mask):
                return latent_image

        self.inner_model = InnerModel()
        self.calls = 0

    def __call__(self, x, sigma, model_options={}, seed=None):
        self.calls += 1
        return x


class _Executor:
    """What a SAMPLER_SAMPLE wrapper is handed: the call it wraps, plus the
    sampler it belongs to."""

    def __init__(self, sampler, fallback):
        self.class_obj = sampler
        self._fallback = fallback
        self.fell_back = False

    def __call__(self, *args, **kwargs):
        self.fell_back = True
        return self._fallback


def _install_alg(**values):
    """ALG's wrapper, without a ModelPatcher in the way."""
    from comfy.patcher_extension import WrappersMP
    from modules.sampling import alg

    captured = {}

    class Patcher:
        def add_wrapper_with_key(self, kind, key, wrapper):
            captured["kind"] = kind
            captured["wrapper"] = wrapper

    settings = {"enabled": True, "strength": 4.0, "until_sigma": 0.6}
    settings.update(values)
    note = alg.install(Patcher(), settings, key="funpack.alg")
    return captured.get("wrapper"), note, captured.get("kind"), WrappersMP


def test_it_is_off_unless_asked_for():
    from modules.sampling import alg
    assert alg.install(object(), {"enabled": False}, key="k") is None


def test_it_installs_around_the_sampler_call():
    wrapper, note, kind, WrappersMP = _install_alg()
    assert wrapper is not None and "loosening" in note
    assert kind == WrappersMP.SAMPLER_SAMPLE


def test_the_anchor_the_denoiser_sees_changes_with_sigma():
    """The whole point, end to end through comfy's own KSAMPLER."""
    import torch
    import comfy.samplers
    from modules.sampling.alg.blur import blur_frames

    sharp = torch.randn(1, 3, 4, 16, 16)
    blurred = blur_frames(sharp, 4.0, (0,))
    seen = []

    def sampler_function(model_k, x, sigmas, extra_args=None, callback=None,
                         disable=None, **options):
        # What a real sampler loop does: call the denoiser once per sigma. The
        # anchor pinned for that call is what ALG decides.
        for sigma in (torch.tensor([0.9]), torch.tensor([0.2])):
            model_k(x, sigma, denoise_mask=None)
            seen.append(model_k.latent_image)
        return x

    wrapper, _note, _kind, _mp = _install_alg()
    real = comfy.samplers.KSAMPLER(sampler_function)
    executor = _Executor(real, fallback="unused")

    wrapper(executor, _Guider(), torch.tensor([0.9, 0.2]), {}, None,
            torch.zeros_like(sharp), sharp, torch.ones_like(sharp), True)

    assert len(seen) == 2, "the denoiser was not called per sigma"
    assert torch.equal(seen[0], blurred), "the early step did not get the loosened anchor"
    assert torch.equal(seen[1], sharp), "the late step did not get the sharp anchor back"


def test_the_real_denoiser_is_left_holding_the_sharp_anchor():
    """The denoiser outlives this call. Leaving a blurred anchor pinned on it is
    the shape of every leak this project has had."""
    import torch
    import comfy.samplers

    sharp = torch.randn(1, 3, 4, 16, 16)

    holder = {}

    def sampler_function(model_k, x, sigmas, extra_args=None, callback=None,
                         disable=None, **options):
        model_k(x, torch.tensor([0.9]), denoise_mask=None)    # leaves it on the loosened anchor
        holder["denoiser"] = model_k._inner      # the real one, behind the proxy
        return x

    wrapper, _n, _k, _mp = _install_alg()
    wrapper(_Executor(comfy.samplers.KSAMPLER(sampler_function), "unused"),
            _Guider(), torch.tensor([0.9]), {}, None,
            torch.zeros_like(sharp), sharp, torch.ones_like(sharp), True)

    assert torch.equal(holder["denoiser"].latent_image, sharp)


def test_a_sampler_error_still_restores_the_sharp_anchor():
    import torch
    import comfy.samplers

    sharp = torch.randn(1, 3, 4, 16, 16)

    holder = {}

    def sampler_function(model_k, x, sigmas, **kwargs):
        model_k(x, torch.tensor([0.9]), denoise_mask=None)
        holder["denoiser"] = model_k._inner
        raise RuntimeError("interrupted")

    wrapper, _n, _k, _mp = _install_alg()
    with pytest.raises(RuntimeError, match="interrupted"):
        wrapper(_Executor(comfy.samplers.KSAMPLER(sampler_function), "unused"),
                _Guider(), torch.tensor([0.9]), {}, None,
                torch.zeros_like(sharp), sharp, torch.ones_like(sharp), True)

    assert torch.equal(holder["denoiser"].latent_image, sharp), (
        "an interrupt left the anchor loosened")


@pytest.mark.parametrize("missing", ["latent_image", "denoise_mask"])
def test_with_no_pinned_anchor_it_stands_aside(missing):
    """No anchor means nothing to loosen, and the run must proceed untouched."""
    import torch
    import comfy.samplers

    sharp = torch.randn(1, 3, 4, 16, 16)
    wrapper, _n, _k, _mp = _install_alg()
    executor = _Executor(comfy.samplers.KSAMPLER(lambda *a, **k: None), "untouched")

    args = dict(latent_image=sharp, denoise_mask=torch.ones_like(sharp))
    args[missing] = None

    out = wrapper(executor, object(), torch.tensor([0.9]), {}, None,
                  torch.zeros_like(sharp), args["latent_image"], args["denoise_mask"], True)

    assert out == "untouched" and executor.fell_back


def test_an_image_latent_stands_aside_rather_than_reshaping_a_guess():
    """A 4-D latent has no anchor frame. v4 reached into a model-specific packed
    layout here; this declines instead of guessing."""
    import torch
    import comfy.samplers

    image_latent = torch.randn(1, 4, 64, 64)
    wrapper, _n, _k, _mp = _install_alg()
    executor = _Executor(comfy.samplers.KSAMPLER(lambda *a, **k: None), "untouched")

    out = wrapper(executor, object(), torch.tensor([0.9]), {}, None,
                  torch.zeros_like(image_latent), image_latent,
                  torch.ones_like(image_latent), True)

    assert out == "untouched" and executor.fell_back
