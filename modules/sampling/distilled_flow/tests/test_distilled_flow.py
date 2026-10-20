import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _line(model, x, sigmas, name):
    import comfy.k_diffusion.sampling as ks
    return getattr(ks, f"sample_{name}")(model, x, sigmas)


def _model(x, sigma, **_):
    return x * 0.5 + 0.1 * sigma.reshape(-1, *([1] * (x.dim() - 1)))      # any smooth x0 prediction


def test_the_presets_are_ordinary_comfy_sampler_names():
    import comfy.samplers
    import modules.sampling.distilled_flow  # noqa: F401
    for name in ("dflow_ord1", "dflow_ord2", "dflow_ord2_ramp", "dflow_ord1_heun2", "dflow_ord2_ramp_heun1"):
        assert name in comfy.samplers.SAMPLER_NAMES
        assert comfy.samplers.sampler_object(name).sampler_function is not None


def test_order_one_is_euler():
    import comfy.k_diffusion.sampling as ks
    import modules.sampling.distilled_flow  # noqa: F401
    x, sigmas = torch.randn(1, 4, 2, 4, 4), torch.linspace(1.0, 0.0, 5)
    assert torch.allclose(_line(_model, x, sigmas, "dflow_ord1"), ks.sample_euler(_model, x, sigmas), atol=1e-5)


def test_order_two_differs_and_ramp_differs_again():
    import modules.sampling.distilled_flow  # noqa: F401
    x, sigmas = torch.randn(1, 4, 2, 4, 4), torch.linspace(1.0, 0.0, 5)
    one, two, ramp = (_line(_model, x, sigmas, n) for n in ("dflow_ord1", "dflow_ord2", "dflow_ord2_ramp"))
    assert not torch.allclose(one, two) and not torch.allclose(two, ramp)


def test_heun_corrects_the_last_steps_with_one_more_call_each():
    import modules.sampling.distilled_flow  # noqa: F401
    x, sigmas = torch.randn(1, 4, 2, 4, 4), torch.linspace(1.0, 0.0, 5)
    for name, extra in (("dflow_ord1", 0), ("dflow_ord1_heun1", 1), ("dflow_ord1_heun2", 2)):
        calls = []
        _line(lambda a, s, **k: calls.append(1) or _model(a, s), x, sigmas, name)
        assert len(calls) == 4 + extra


def test_the_sound_stream_stays_first_order():
    from comfy.nested_tensor import NestedTensor
    import modules.sampling.distilled_flow  # noqa: F401
    video, audio = torch.randn(1, 4, 2, 4, 4), torch.randn(1, 8, 5)
    sigmas = torch.linspace(1.0, 0.0, 5)
    both = _line(lambda x, s, **k: x * 0.5, NestedTensor((video, audio)), sigmas, "dflow_ord2_heun2")
    alone = _line(lambda x, s, **k: x * 0.5, audio, sigmas, "dflow_ord1")
    assert torch.allclose(both.unbind()[1], alone, atol=1e-5)
