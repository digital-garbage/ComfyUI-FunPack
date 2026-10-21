"""Every sampler and scheduler the Sampler window offers, on a real (tiny) model class.

The picker lists whatever ComfyUI registers plus FunPack's own, so what is on offer is read at run time, and each
name must reach a finished, finite latent. The model is ComfyUI's own `model_base.MiniMaxH3` at a few thousand
parameters: it carries the real sampling object, packing and wrappers, which a bare diffusion module does not.
Random weights say nothing about quality; they prove the path runs end to end.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(scope="module")
def model(comfyui):
    import comfy.model_patcher as mp
    import comfy.supported_models as sm
    torch.manual_seed(0)
    config = sm.MiniMaxH3(dict(
        image_model="minimax_h3", hidden_size=64, num_layers=4, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=32, ffn_hidden_size=64, text_dim=48, timestep_input_dim=32, time_embed_hidden_size=64,
        time_embed_dim=32, rope_inv_freq_len=4))
    config.set_inference_dtype(torch.float32, None)
    base = config.get_model({}, device=torch.device("cpu"))
    for p in base.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.05)
        p.requires_grad_(False)
    return mp.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def _inputs():
    from comfy.nested_tensor import NestedTensor
    cond = [[torch.randn(1, 5, 48), {}]]
    latent = {"samples": NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    return cond, latent


def _names(kind):
    import comfy.samplers
    return list(comfy.samplers.SAMPLER_NAMES if kind == "sampler" else comfy.samplers.SCHEDULER_NAMES)


def _finite(out):
    return all(torch.isfinite(part).all() for part in out.result[0]["samples"].unbind())


def _run(model, **kw):
    from modules.sampling.sampler.nodes import FunPackSampler
    cond, latent = _inputs()
    args = dict(seed=1, steps=3, cfg=1.0, sampler_name="euler", scheduler="normal", denoise=1.0)
    args.update(kw)
    return FunPackSampler.execute(model, cond, cond, latent, **args)


def pytest_generate_tests(metafunc):
    if "sampler_name" in metafunc.fixturenames:
        metafunc.parametrize("sampler_name", _safe("sampler"))
    if "scheduler_name" in metafunc.fixturenames:
        metafunc.parametrize("scheduler_name", _safe("scheduler"))


def _safe(kind):
    try:
        import comfy.samplers  # noqa: F401
    except Exception:                                    # noqa: BLE001 -- no ComfyUI here: the fixture skips the test
        return ["euler"]
    from modules.sampling import sampler as _  # noqa: F401  (FunPack's own samplers register when their module loads)
    return _names(kind)


def test_every_sampler_finishes_with_a_finite_latent(model, sampler_name):
    assert _finite(_run(model, sampler_name=sampler_name))


def test_every_scheduler_finishes_with_a_finite_latent(model, scheduler_name):
    assert _finite(_run(model, scheduler=scheduler_name, steps=4))


def test_guidance_with_a_negative_and_a_partial_denoise_and_a_typed_schedule_run(model):
    import comfy.samplers
    assert _finite(_run(model, cfg=4.0))
    assert _finite(_run(model, denoise=0.5, steps=4))
    sigmas = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0])
    assert _finite(_run(model, sigmas=sigmas))
    assert "dflow_ord1" in comfy.samplers.SAMPLER_NAMES, "FunPack's own samplers were not offered"
