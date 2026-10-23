"""PTD on a tiny real MiniMax-H3: all three file layouts, the hop switching, the sampler swap and clean removal.

Structure follows ComfyUI-PTD-Compact's tests (Nynxz, Apache-2.0). They prove the wiring and the maths of
our side on random weights, nothing about picture quality on the real model.
"""

import pytest
import torch
import torch.nn.functional as F

from . import _tiny

HOP_TIMES = (0.999, 0.947, 0.857, 0.667)


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture
def ptd():
    import importlib
    return importlib.import_module("modules.sampling.ptd")


@pytest.fixture
def folder(tmp_path):
    import folder_paths
    folder_paths.add_model_folder_path("ptd", str(tmp_path), is_default=True)
    return tmp_path


def _guarded(patcher):
    from modules._core import patching
    return patching.GuardedPatcher(patcher, "funpack.ptd", patching.Dropped())


def _install(ptd, patcher, name, kernels="exact"):
    note = ptd.install(_guarded(patcher), {"enabled": True, "adapter": name, "kernels": kernels}, "funpack.ptd")
    return patcher.model_options["funpack_ptd_controller"], note


def _clone(base):
    return base.clone()


def _sample(model, sampler=None, sigmas=None):
    import comfy.nested_tensor
    import comfy.samplers
    guider = comfy.samplers.CFGGuider(model)
    context = torch.randn(1, 6, _tiny.DIMS["text_dim"], generator=torch.Generator().manual_seed(3))
    guider.set_conds([[context, {}]], [[context, {}]])
    guider.set_cfg(1.0)
    video, audio = torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)
    noise = comfy.nested_tensor.NestedTensor([torch.randn(video.shape, generator=torch.Generator().manual_seed(4)),
                                              torch.randn(audio.shape, generator=torch.Generator().manual_seed(5))])
    sampler = sampler or comfy.samplers.sampler_object("euler")
    sigmas = sigmas if sigmas is not None else torch.linspace(1.0, 0.0, 21)
    try:
        out = guider.sample(noise, comfy.nested_tensor.NestedTensor([video, audio]), sampler, sigmas,
                            disable_pbar=True, seed=7)
    finally:
        import comfy.model_management
        comfy.model_management.unload_all_models()
    return [t.clone() for t in out.unbind()]


@pytest.mark.parametrize("int8", [False, True], ids=["float", "int8"])
@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
def test_each_hop_adds_its_own_expert_to_the_layer(ptd, folder, int8, curve):
    base = _tiny.native_model(curve)
    native = base.get_model_object("diffusion_model")
    weights = {n: p.detach().clone() for n, p in native.named_parameters()}
    dense = _tiny.epfl_file(folder / "a.safetensors", native, int8=int8)
    controller, _ = _install(ptd, _clone(base), "a.safetensors")
    for hop, sigma in enumerate(HOP_TIMES):
        for name in ("blocks.0.attn.qkv_proj", "blocks.1.mlp.fc1", "blocks.1.mlp.fc2", "blocks.0.attn.out_proj",
                     "token_refiner.blocks.0.attn.qkv_proj"):
            layer = native.get_submodule(name)
            inp = torch.randn(5, layer.in_features)
            parts = [(p, d) for (n, p, h), d in dense.items() if n == name and h == hop]
            if name.endswith("qkv_proj"):
                order = {"attn.to_q": 0, "attn.to_k": 1, "attn.to_v": 2}
                delta = torch.cat([d for _, d in sorted(parts, key=lambda pd: order[pd[0]])], 0)
            else:
                (part, delta), = parts
                if part == "ff.net.0.proj":
                    up, gate = delta.chunk(2, 0)
                    delta = torch.cat([gate, up], 0)
            want = F.linear(inp, weights[name + ".weight"] + delta, weights.get(name + ".bias"))
            with controller.activate(hop, sigma), controller.forward_hooks(controller.current()):
                got = layer(inp)
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(layer(inp), F.linear(inp, weights[name + ".weight"], weights.get(name + ".bias")),
                                       rtol=1e-4, atol=1e-5)          # nothing stays hooked after the hop


def test_a_hop_must_run_at_its_trained_time(ptd, folder):
    base = _tiny.native_model()
    _tiny.epfl_file(folder / "a.safetensors", base.get_model_object("diffusion_model"))
    controller, _ = _install(ptd, _clone(base), "a.safetensors")
    with pytest.raises(ValueError, match="trained time"):
        with controller.activate(1, 0.5):
            pass
    with pytest.raises(RuntimeError, match="no PTD hop is running"):
        controller.current()


def test_the_chosen_sampler_is_replaced_by_the_adapters_own_hops(ptd, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    _tiny.epfl_file(folder / "a.safetensors", native)
    _tiny.epfl_file(folder / "zero.safetensors", native, zero_a=True)
    seen = []
    for name in ("a.safetensors", "zero.safetensors"):
        model = _clone(base)
        controller, note = _install(ptd, model, name)
        assert "4 hops" in note
        calls = []
        original = controller.activate

        def spy(hop, sigma, original=original, calls=calls):
            calls.append(hop)
            return original(hop, sigma)

        controller.activate = spy
        # twenty euler steps were asked for; PTD runs its four hops
        out = _sample(model)
        assert calls == [0, 1, 2, 3]
        assert all(torch.isfinite(t).all() for t in out)
        seen.append(out)
    assert not torch.allclose(seen[0][0], seen[1][0])
    # the base model is not left patched
    plain = _sample(_clone(base))
    assert not torch.allclose(plain[0], seen[0][0])


@pytest.mark.parametrize("curve", [False, True], ids=["full", "pruned"])
def test_exact_and_fast_blocks_agree(ptd, folder, curve):
    base = _tiny.native_model(curve)
    _tiny.pdd_file(folder / "r.safetensors", base.get_model_object("diffusion_model"))
    exact = _sample(_installed(ptd, base, "exact"))
    fast = _sample(_installed(ptd, base, "fast"))
    for e, f in zip(exact, fast):
        assert torch.isfinite(e).all()
        assert torch.allclose(e, f, rtol=1e-4, atol=1e-4)


def _installed(ptd, base, kernels):
    model = _clone(base)
    _install(ptd, model, "r.safetensors", kernels)
    return model


def test_int8_ref2va_stays_close_to_float(ptd, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    _tiny.pdd_file(folder / "r.safetensors", native)
    _tiny.pdd_file(folder / "q.safetensors", native, int8=True)
    ptd._kept.clear()
    full = _sample(_installed_file(ptd, base, "r.safetensors"))
    small = _sample(_installed_file(ptd, base, "q.safetensors"))
    for f, q in zip(full, small):
        assert torch.allclose(f, q, rtol=1e-2, atol=1e-2)


def _installed_file(ptd, base, name):
    model = _clone(base)
    _install(ptd, model, name)
    return model


def test_ref2va_heads_replace_the_native_heads_for_a_hop_only(ptd, folder):
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    _tiny.pdd_file(folder / "r.safetensors", native)
    controller, _ = _install(ptd, _clone(base), "r.safetensors")
    head = native.final_layer.video_out
    x = torch.randn(5, head.in_features)
    plain = head(x)
    hop = 1
    with controller.activate(hop, float(controller.grid[hop])), controller.forward_hooks(controller.current()):
        hopped = head(x)
    weight, bias = controller.adapter.heads[hop]["final_layer.video_out"]
    torch.testing.assert_close(hopped, F.linear(x, weight, bias), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(hopped, plain, atol=1e-3)
    assert torch.equal(head(x), plain)


def test_grids_are_the_published_ones(ptd, folder):
    from modules.sampling.ptd import schedule
    assert schedule.RECIPES["h3_4step"][:4] == HOP_TIMES
    for hops in (4, 8):
        grid = schedule.pdd_grid(hops)
        assert len(grid) == hops + 1 and abs(grid[0] - schedule.T_MAX) < 1e-12 and grid[-1] == 0.0
        assert all(a > b for a, b in zip(grid, grid[1:]))
        for video, audio in schedule.pdd_plans(hops):
            for plan in (video, audio):
                assert abs(float(plan.sum()) - 1) < 1e-6 and int((plan > 0).sum()) == 32 // hops


def test_a_file_for_another_model_is_refused_in_plain_words(ptd, folder):
    big = _tiny.native_model()
    _tiny.epfl_file(folder / "a.safetensors", big.get_model_object("diffusion_model"))
    smaller = _tiny.native_model(num_layers=1)
    with pytest.raises(ValueError, match="does not match the loaded model's layers"):
        _install(ptd, _clone(smaller), "a.safetensors")


def test_not_a_ptd_file_is_refused(ptd, folder):
    from safetensors.torch import save_file
    save_file({"x": torch.zeros(2)}, str(folder / "n.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Not a PTD adapter file"):
        _install(ptd, _clone(_tiny.native_model()), "n.safetensors")


def test_no_file_chosen_is_said_not_quarantined(ptd, folder):
    from modules._core import control
    with pytest.raises(control.Unavailable):
        ptd.install(_guarded(_clone(_tiny.native_model())), {"enabled": True, "adapter": "", "kernels": "exact"},
                    "funpack.ptd")


def test_off_installs_nothing(ptd, folder):
    patcher = _clone(_tiny.native_model())
    assert ptd.install(_guarded(patcher), {"enabled": False, "adapter": "", "kernels": "exact"}, "funpack.ptd") is None
    assert "funpack_ptd_controller" not in patcher.model_options


def test_load_modifiers_strips_ptd_so_a_second_pass_reinstalls_it_once(ptd, folder):
    from modules._core import patching
    base = _tiny.native_model()
    _tiny.epfl_file(folder / "a.safetensors", base.get_model_object("diffusion_model"))
    model = _clone(base)
    _install(ptd, model, "a.safetensors")
    again = model.clone()
    assert patching.strip(again, "funpack.") >= 3
    assert "funpack_ptd_controller" not in again.model_options
    assert again.model_options.get("model_function_wrapper") is None
    _install(ptd, again, "a.safetensors")                      # no "already carries" error
    assert "funpack_ptd_controller" in again.model_options
    assert "funpack_ptd_controller" not in base.model_options and not base.object_patches


def test_a_latent_that_does_not_start_empty_is_refused(ptd, folder):
    from modules.sampling.ptd.sampler import _refuse_unsupported

    class Model:
        latent_image = torch.ones(1, 2)

    with pytest.raises(ValueError, match="empty latent"):
        _refuse_unsupported(Model(), {})
    with pytest.raises(ValueError, match="masked"):
        _refuse_unsupported(object(), {"denoise_mask": torch.ones(1)})


def test_it_installs_through_load_modifiers_on_an_h3_model(ptd, folder, monkeypatch):
    from core import registry
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    base = _tiny.native_model()
    _tiny.pdd_file(folder / "r.safetensors", base.get_model_object("diffusion_model"))
    spec = registry.current().specs["ptd"]
    monkeypatch.setitem(spec.settings["adapter"], "options", [{"value": "r.safetensors", "label": "r"}])
    monkeypatch.setitem(spec.settings["adapter"], "default", "r.safetensors")
    patched, status = FunPackLoadModifiers.execute(_clone(base), {"ptd": {"enabled": True}}).result
    assert "ptd: r.safetensors, 4 hops" in status
    assert "failed" not in status
    out = _sample(patched)
    assert all(torch.isfinite(t).all() for t in out)
    # switched off, the same model samples as the plain one does
    plain, status = FunPackLoadModifiers.execute(_clone(base), {"ptd": {"enabled": False}}).result
    assert "ptd" not in status.splitlines()[0]


def test_an_ordinary_lora_stacks_with_ptd_in_either_order(ptd, folder):
    import comfy.sd
    base = _tiny.native_model()
    native = base.get_model_object("diffusion_model")
    dense = _tiny.epfl_file(folder / "a.safetensors", native)
    g = torch.Generator().manual_seed(2)
    layer = native.get_submodule("blocks.1.mlp.fc2")
    up, down = torch.randn(layer.out_features, 4, generator=g) * 0.1, torch.randn(4, layer.in_features, generator=g) * 0.1
    lora = {"blocks.1.mlp.fc2.lora_up.weight": up, "blocks.1.mlp.fc2.lora_down.weight": down}
    weight = layer.weight.detach().clone()
    for lora_first in (True, False):
        ptd._kept.clear()
        model = comfy.sd.load_lora_for_models(_clone(base), None, lora, 0.7, 0)[0] if lora_first else _clone(base)
        controller, _ = _install(ptd, model, "a.safetensors")
        if not lora_first:
            model = comfy.sd.load_lora_for_models(model, None, lora, 0.7, 0)[0]
        import comfy.model_management
        comfy.model_management.load_models_gpu([model])
        try:
            inp = torch.randn(5, layer.in_features)
            hop = 2
            want = F.linear(inp, weight + 0.7 * up @ down + dense[("blocks.1.mlp.fc2", "ff.net.2", hop)],
                            layer.bias)
            with controller.activate(hop, HOP_TIMES[hop]), controller.forward_hooks(controller.current()):
                torch.testing.assert_close(layer(inp), want, rtol=1e-4, atol=1e-5)
        finally:
            comfy.model_management.unload_all_models()


def test_step_aware_modules_are_given_the_hop_schedule(ptd, folder):
    base = _tiny.native_model()
    _tiny.epfl_file(folder / "a.safetensors", base.get_model_object("diffusion_model"))
    model = _clone(base)
    seen = []

    def probe(apply_fn, args):
        seen.append(args["c"]["transformer_options"]["sample_sigmas"].tolist())
        return apply_fn(args["input"], args["timestep"], **args["c"])

    model.set_model_unet_function_wrapper(probe)
    _install(ptd, model, "a.safetensors")
    _sample(model)                                  # twenty steps asked for
    assert seen and all(len(s) == 5 for s in seen)


def test_the_file_list_is_read_again_when_settings_open(ptd, folder, monkeypatch):
    pytest.importorskip("aiohttp")
    from core import registry, routes
    spec = registry.current().specs["ptd"]
    monkeypatch.setitem(spec.settings["adapter"], "options", [{"value": "", "label": "No file in models/ptd"}])
    monkeypatch.setitem(spec.settings["adapter"], "default", "")
    base = _tiny.native_model()
    _tiny.epfl_file(folder / "new.safetensors", base.get_model_object("diffusion_model"))
    listed = {m["id"]: m for m in routes.manifest()["modules"]}
    adapter = listed["ptd"]["settings"]["adapter"]
    assert "new.safetensors" in [o["value"] for o in adapter["options"]]
    assert adapter["default"] in [o["value"] for o in adapter["options"]]


def test_the_hop_schedule_sits_where_the_sampler_was_given_its_own(ptd, folder, monkeypatch):
    """On a GPU ComfyUI hands the sampler a CUDA schedule; the one PTD swaps in must be on the same device
    (a CPU copy met the call's CUDA sigma: "Expected all tensors to be on the same device"). The meta device
    stands in for CUDA here."""
    import comfy.samplers
    from comfy.patcher_extension import WrappersMP
    base = _tiny.native_model()
    _tiny.epfl_file(folder / "a.safetensors", base.get_model_object("diffusion_model"))
    model = _clone(base)
    _install(ptd, model, "a.safetensors")
    given = {}
    monkeypatch.setattr(comfy.samplers.KSAMPLER, "sample",
                        lambda self, wrap, sigmas, extra_args, *a, **k: given.update(sigmas=sigmas) or "done")

    class Guider:
        cfg = 1.0
        model_patcher = None

    extra = {"model_options": {"transformer_options": {}}}
    run = model.wrappers[WrappersMP.SAMPLER_SAMPLE]["funpack.ptd"][0]
    sigmas = torch.linspace(1.0, 0.0, 21, device="meta")
    assert run(None, Guider(), sigmas, extra, None, None, None, None, True) == "done"
    assert given["sigmas"].device == sigmas.device and given["sigmas"].dtype == sigmas.dtype
    seen = extra["model_options"]["transformer_options"]["sample_sigmas"]
    assert seen.device == sigmas.device and len(seen) == 5
