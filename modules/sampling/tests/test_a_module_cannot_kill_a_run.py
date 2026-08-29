"""The foundation's promise: a module is never worth the picture.

A modifier that fails to INSTALL was already survivable. The one that costs a
rental is the hook that works at step 0 and raises at step 3 -- the generation
dies after the GPU time is spent. This is the guarantee that it does not.

The case is real, not invented: the Detail modifier's maths comes from ComfyUI's
LatentOperationSharpen, which raises on a 5-D latent. It is deliberately NOT
being fixed -- it exists to be watchable on a machine with no GPU -- so it
stands as the permanent regression case for this.

Detail now declares `spatial_latent`, so it is not OFFERED where the shape it
cannot handle comes from. That closes the ordinary route to this failure and
does not close the failure: a hook meets the tensor it was handed, not the one
its module reasoned about. The model here says spatial and the hook is given
five dimensions, which is precisely the mismatch the guard exists for -- if the
only way to test it were a modifier that is wrong about itself, there would be
nothing to test.
"""

import pytest


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _pre_cfg_args(conds):
    # A FRESH list each call, because the hook mutates conds_out in place and
    # returns the same object. Comparing the result against the list you passed
    # in compares a tensor with itself, which is true whatever the hook did --
    # the first version of these tests did exactly that and proved nothing.
    conds = list(conds)
    return {"conds_out": conds, "conds": conds, "cond_scale": 7.0, "input": conds[0],
            "sigma": None, "model": None, "model_options": {}, "timestep": None}


def test_a_hook_that_raises_mid_sampling_does_not_end_the_run(patcher):
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, _status = FunPackLoadModifiers.execute(
        patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result

    hook = patched.model_options["sampler_pre_cfg_function"][0]
    video = [torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)]
    original = video[0].clone()

    # Thirty steps, as a real run would.
    for _ in range(30):
        out = hook(_pre_cfg_args(video))
        assert torch.equal(out[0], original), "the hook altered the latent after failing"

    assert "funpack.sharpen" in patched.funpack_dropped


def test_the_failure_is_recorded_with_its_reason(patcher):
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, _ = FunPackLoadModifiers.execute(
        patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result
    hook = patched.model_options["sampler_pre_cfg_function"][0]
    hook(_pre_cfg_args([torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)]))

    reasons = dict(patched.funpack_dropped.items())
    assert "funpack.sharpen" in reasons
    assert "NotImplementedError" in reasons["funpack.sharpen"]


def test_it_is_reported_once_not_once_per_step(patcher, capsys):
    """Thirty tracebacks say nothing the first one did not, and burying the
    console is its own way of hiding a failure."""
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, _ = FunPackLoadModifiers.execute(
        patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result
    hook = patched.model_options["sampler_pre_cfg_function"][0]

    capsys.readouterr()
    args = _pre_cfg_args([torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)])
    for _ in range(10):
        hook(args)

    said = capsys.readouterr().err
    assert said.count("is now OFF for the rest of this run") == 1


def test_a_working_hook_is_untouched_by_the_guard(patcher):
    """The guard must not cost the thing it protects."""
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, _ = FunPackLoadModifiers.execute(
        patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result
    hook = patched.model_options["sampler_pre_cfg_function"][0]

    image = [torch.randn(1, 4, 16, 16), torch.randn(1, 4, 16, 16)]
    original = image[0].clone()
    out = hook(_pre_cfg_args(image))
    assert not torch.equal(out[0], original), "the guard swallowed a working modifier"
    assert not patched.funpack_dropped


def test_one_failing_modifier_does_not_disable_another(patcher):
    from core import patching

    dropped = patching.Dropped()

    def boom(args):
        raise RuntimeError("boom")

    def fine(args):
        return "changed"

    bad = patching.guard(boom, "funpack.bad", lambda args: args["conds_out"], dropped)
    good = patching.guard(fine, "funpack.good", lambda args: args["conds_out"], dropped)

    assert bad({"conds_out": "original"}) == "original"
    assert good({"conds_out": "original"}) == "changed"
    assert "funpack.bad" in dropped and "funpack.good" not in dropped


def test_the_record_is_per_run_not_per_process(patcher):
    """v4's inert-reason logging deduped for the life of the interpreter, so a
    long rental session reported the first generation and never again."""
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    def run():
        patched, _ = FunPackLoadModifiers.execute(
            patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result
        hook = patched.model_options["sampler_pre_cfg_function"][0]
        hook(_pre_cfg_args([torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)]))
        return patched.funpack_dropped

    first, second = run(), run()
    assert first is not second
    assert "funpack.sharpen" in first and "funpack.sharpen" in second


def test_a_hook_shape_with_no_known_neutral_result_is_named(patcher):
    """Honesty about the limit: an unguardable hook is installed and said out
    loud, rather than counted as protected."""
    from core import patching

    guarded = patching.GuardedPatcher(patcher, "funpack.x", patching.Dropped())
    guarded.set_model_patch(lambda *a: None, "attn1_patch")
    assert "set_model_patch" in guarded.unguarded


def test_a_guarded_hook_can_still_be_stripped(patcher):
    """Guarding must not make a hook unremovable, or chaining stacks again."""
    from core import patching
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    values = {"sharpen": {"enabled": True, "amount": 0.4}}
    once = FunPackLoadModifiers.execute(patcher, settings=values).result[0]
    twice = FunPackLoadModifiers.execute(once, settings=values).result[0]
    assert len(twice.model_options.get("sampler_pre_cfg_function", [])) == 1


# --- full control: the guard steps aside, and says so ----------------------

def test_full_control_lets_a_failing_modifier_end_the_run(patcher):
    """Asking for the raw behaviour means getting it. Someone exploring the
    edges wants the real traceback at the real step, not a tidy recovery."""
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, status = FunPackLoadModifiers.execute(patcher, settings={
        "sharpen": {"enabled": True, "amount": 0.4},
        "full_control": {"enabled": True},
    }).result

    assert "full control is ON" in status
    hook = patched.model_options["sampler_pre_cfg_function"][0]

    with pytest.raises(NotImplementedError):
        hook(_pre_cfg_args([torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)]))


def test_full_control_is_never_silent(patcher):
    """A run made without the guards has to be identifiable as one afterwards."""
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    _model, status = FunPackLoadModifiers.execute(patcher, settings={
        "sharpen": {"enabled": True, "amount": 0.4},
        "full_control": {"enabled": True},
    }).result
    assert "consequences" in status or "will end the run" in status


def test_the_guards_are_on_unless_asked_otherwise(patcher):
    """The surprising direction is the one you have to ask for. A missing
    module, a missing key and an explicit false all mean guarded."""
    import torch
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    for settings in ({"sharpen": {"enabled": True, "amount": 0.4}},
                     {"sharpen": {"enabled": True, "amount": 0.4},
                      "full_control": {"enabled": False}}):
        patched, _ = FunPackLoadModifiers.execute(patcher, settings=settings).result
        hook = patched.model_options["sampler_pre_cfg_function"][0]
        video = [torch.randn(1, 4, 8, 16, 16), torch.randn(1, 4, 8, 16, 16)]
        original = video[0].clone()
        out = hook(_pre_cfg_args(video))          # must not raise
        assert torch.equal(out[0], original)


def test_is_on_reads_the_switch_the_same_way_everywhere():
    """One place decides what "on" means, so no caller gets the key or the
    default subtly wrong."""
    from modules.system import full_control as fc
    assert fc.is_on({"full_control": {"enabled": True}}) is True
    assert fc.is_on({"full_control": {"enabled": False}}) is False
    assert fc.is_on({}) is False
    assert fc.is_on(None) is False


def test_the_guard_never_swallows_a_cancel():
    """Cancel reaches a hook as an exception like any other, and a guard that
    caught it would leave the user pressing stop on a run that will not stop.

    It works today only because ComfyUI made InterruptProcessingException a
    BaseException rather than an Exception -- a detail this depends on and does
    not control, so it is pinned here. Widening the guard to BaseException would
    break Cancel with nothing else failing.
    """
    from comfy.model_management import InterruptProcessingException
    from core import patching

    def cancelled(args):
        raise InterruptProcessingException()

    guarded = patching.guard(cancelled, "funpack.x",
                             lambda args: args["conds_out"], patching.Dropped())

    with pytest.raises(InterruptProcessingException):
        guarded({"conds_out": "untouched"})


def test_a_keyboard_interrupt_is_not_swallowed_either():
    from core import patching

    def interrupted(args):
        raise KeyboardInterrupt()

    guarded = patching.guard(interrupted, "funpack.x",
                             lambda args: args["conds_out"], patching.Dropped())
    with pytest.raises(KeyboardInterrupt):
        guarded({"conds_out": "untouched"})


def test_detail_is_absent_on_a_model_whose_latent_it_cannot_handle(temporal_patcher):
    """Not dropped: absent.

    Being dropped mid-run is the safety net, and a safety net is not a design.
    A knob that can be switched on, reports nothing at the knob, and does
    nothing is the exact fault this project has already shipped once -- so a
    modifier whose maths cannot take the shape a model produces has to say so
    in what it REQUIRES, and stop being offered there.
    """
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, status = FunPackLoadModifiers.execute(
        temporal_patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result

    assert patched.model_options.get("sampler_pre_cfg_function", []) == []
    assert not getattr(patched, "funpack_dropped", None) or \
        "funpack.sharpen" not in patched.funpack_dropped, \
        "it was installed and then failed, rather than never being offered"
    assert "sharpen" in status and "spatial_latent" in status, status


def test_detail_is_offered_on_a_model_whose_latent_it_can_handle(patcher):
    """The other half, or the test above passes on a modifier that is simply
    broken everywhere."""
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers

    patched, _status = FunPackLoadModifiers.execute(
        patcher, settings={"sharpen": {"enabled": True, "amount": 0.4}}).result
    assert len(patched.model_options.get("sampler_pre_cfg_function", [])) == 1


def test_a_modifier_that_breaks_while_installing_is_not_called_absent(patcher, monkeypatch):
    """Loaded, chosen, and broke on the way in.

    The note beside it already said "failed to install"; the log line said "did
    not load", which is a different thing and a different place to go looking.
    """
    from core import log
    from modules.sampling.modifiers import nodes as mod

    def explode(_patcher, _values, key=None):
        raise RuntimeError("the wrapper is upside down")

    # Named, because picking the first provider found lands on one whose traits
    # exclude this model -- it is then never installed, nothing is logged, and a
    # test asserting "nothing said did not load" passes having run nothing.
    spec = mod.registry_mod.current().specs["sharpen"]
    monkeypatch.setitem(spec.provides, mod.CAPABILITY, explode)

    log._reset()
    mod.FunPackLoadModifiers.execute(
        patcher, settings={spec.id: {"enabled": True, "amount": 0.4}}).result

    said = [r["message"] for r in log.history()]
    # Asserted positively FIRST: "nothing says did not load" is also true of a
    # test where the modifier was never chosen and nothing was logged at all.
    assert any("failed while installing itself" in m for m in said), said
    assert not any("did not load" in m for m in said), said
