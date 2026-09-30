"""On H3 the step wrappers carry their edit into the NEXT step's input (_InputSteer) and
return the model's own answer, so the last answer, which is the output on a few-step
schedule, is never edited. Driven through a 4-step euler loop on the real wrappers."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import samplers  # noqa: E402
import tsr  # noqa: E402

SCHEDULE = torch.tensor([1.0, 0.973, 0.923, 0.8, 0.0])     # simple @ 4 steps, shift 12


class _Model:
    def __init__(self):
        self.model_options = {}


def _node(h3):
    node = samplers.FunPackLTXAVSceneChainSampler()
    node._is_h3 = h3
    return node


def _euler(wrapper, x, schedule=SCHEDULE):
    """comfy's euler over a model_function_wrapper -> (answers returned, (input seen,
    raw answer) per call, final latent). The toy model answers 0.5 * its input."""
    seen, answers = [], []

    def apply_fn(inp, t, **c):
        raw = 0.5 * inp
        seen.append((inp.clone(), raw.clone()))
        return raw

    for i in range(len(schedule) - 1):
        s, s_next = schedule[i], schedule[i + 1]
        to = {"sample_sigmas": schedule, "sigmas": s.reshape(1)}
        den = wrapper(apply_fn, {"input": x, "timestep": s.reshape(1),
                                 "c": {"transformer_options": to}, "cond_or_uncond": [0]})
        answers.append(den)
        x = x + (s_next - s) * (x - den) / s
    return answers, seen, x


def test_h3_decisiveness_never_edits_an_answer_and_pushes_the_next_input():
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.9)
    x = torch.randn(1, 2, 3, 4, 4, generator=torch.Generator().manual_seed(0))
    answers, seen, out = _euler(model.model_options["model_function_wrapper"], x)
    for den, (_inp, raw) in zip(answers, seen):
        assert torch.equal(den, raw)                   # the model's own answer, every step
    assert torch.allclose(out, answers[-1], atol=1e-6)               # so the output is too


def test_h3_push_lands_exactly_on_the_next_input():
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.9)
    wrapper = model.model_options["model_function_wrapper"]
    x = torch.randn(1, 2, 3, 4, 4, generator=torch.Generator().manual_seed(1))
    xs = []

    def apply_fn(inp, t, **c):
        xs.append(inp.clone())
        return 0.5 * inp

    def call(i, x):
        s = SCHEDULE[i]
        to = {"sample_sigmas": SCHEDULE, "sigmas": s.reshape(1)}
        return wrapper(apply_fn, {"input": x, "timestep": s.reshape(1),
                                  "c": {"transformer_options": to}})

    den = call(1, x)
    x2 = torch.randn_like(x)
    call(2, x2)
    s1, s2 = float(SCHEDULE[1]), float(SCHEDULE[2])
    push = tsr.rescale_x0(x, den, s1, 0.9) - den
    assert torch.allclose(xs[1], x2 + (1.0 - s2) * push, atol=1e-6)


def test_h3_a_new_pass_does_not_inherit_the_last_push():
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.9)
    wrapper = model.model_options["model_function_wrapper"]
    x = torch.randn(1, 2, 3, 4, 4)
    _euler(wrapper, x)
    _answers, seen, _out = _euler(wrapper, x)          # second pass, same wrapper
    assert torch.equal(seen[0][0], x)                  # first input untouched


def test_ltx_keeps_editing_the_answer():
    model = _Model()
    _node(False)._build_tsr_wrapper(model, 0.9)
    x = torch.randn(1, 2, 3, 4, 4)
    answers, seen, _out = _euler(model.model_options["model_function_wrapper"], x)
    inp, raw = seen[-1]
    assert torch.allclose(answers[-1], tsr.rescale_x0(inp, raw, 0.8, 0.9))


def test_h3_output_guidance_gates_on_the_step_it_lands_on():
    """Last-half gate: on 4 steps only step 3 lands (on the last step); the last step itself
    is never corrected."""
    class VF:
        calls = 0

        def gradient(self, target):
            VF.calls += 1
            return torch.ones_like(target)

    ramp = samplers._make_steer_ramp(SCHEDULE, True)
    model = _Model()
    _node(True)._build_output_guidance_wrapper(model, VF(), 0.1, ramp_fn=ramp)
    answers, seen, out = _euler(model.model_options["model_function_wrapper"],
                                torch.randn(1, 2, 3, 4, 4))
    assert VF.calls == 1                               # step index 2, landing on 3
    for den, (_inp, raw) in zip(answers, seen):
        assert torch.equal(den, raw)
    assert torch.allclose(out, answers[-1], atol=1e-6)


def test_h3_late_branch_runs_no_weak_copy_on_the_last_step():
    from test_late_guidance import _Blocks, _Model as _LModel, _apply
    net = _Blocks()
    node = _node(True)
    patched, stats = node._install_late_branch(_LModel(net), 3, 0.5)
    wrapper = patched.model_options["model_function_wrapper"]
    base_to = patched.model_options["transformer_options"]
    x = torch.zeros(1, 2, 1, 2, 2)
    for i in range(4):
        net.calls.clear()
        s = SCHEDULE[i]
        to = dict(base_to, sample_sigmas=SCHEDULE, sigmas=s.reshape(1))
        out = wrapper(_apply(net), {"input": x, "timestep": s.reshape(1),
                                    "c": {"transformer_options": to}, "cond_or_uncond": [0]})
        if i < 3:
            assert net.calls == [0, 1, 2, 3, 4, 5, 4, 5]
        else:
            assert net.calls == [0, 1, 2, 3, 4, 5]       # the output step: normal pass only
    # The last input carried step 2's push (w * (normal - weak) = 0.5 * 4 = 2 per value).
    assert torch.allclose(out, torch.full_like(x, 21.0 + (1.0 - 0.8) * 2.0))
    assert stats["guided"] == 3


def test_h3_a_second_call_in_the_same_step_gets_only_the_last_steps_push():
    """A split batch or hook group makes two calls in one step: neither may see a push made
    during that same step, and both get the previous step's."""
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.9)
    wrapper = model.model_options["model_function_wrapper"]
    xs = []

    def apply_fn(inp, t, **c):
        xs.append(inp.clone())
        return 0.5 * inp

    def call(i, x):
        s = SCHEDULE[i]
        to = {"sample_sigmas": SCHEDULE, "sigmas": s.reshape(1)}
        return wrapper(apply_fn, {"input": x, "timestep": s.reshape(1),
                                  "c": {"transformer_options": to}, "cond_or_uncond": [0]})

    x = torch.randn(1, 2, 3, 4, 4, generator=torch.Generator().manual_seed(2))
    den = call(1, x)
    push = tsr.rescale_x0(x, den, float(SCHEDULE[1]), 0.9) - den
    a, b = torch.randn_like(x), torch.randn_like(x)
    call(2, a)
    call(2, b)
    lift = (1.0 - float(SCHEDULE[2])) * push
    assert torch.allclose(xs[1], a + lift, atol=1e-6)
    assert torch.allclose(xs[2], b + lift, atol=1e-6)
    x3 = torch.randn_like(x)
    call(3, x3)
    assert not torch.allclose(xs[3], x3)                 # step 2's push reaches step 3


def test_h3_a_size_change_is_said_not_silent(capsys):
    import funpack_log
    funpack_log.begin_run()
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.9)
    wrapper = model.model_options["model_function_wrapper"]

    def call(i, x):
        s = SCHEDULE[i]
        to = {"sample_sigmas": SCHEDULE, "sigmas": s.reshape(1)}
        return wrapper(lambda inp, t, **c: 0.5 * inp, {"input": x, "timestep": s.reshape(1),
                                                         "c": {"transformer_options": to}})

    call(1, torch.randn(1, 2, 3, 4, 4))
    call(2, torch.randn(1, 2, 3, 4, 8))
    assert "latent size changed" in capsys.readouterr().out


def test_h3_a_one_step_schedule_says_it_does_nothing(capsys):
    import funpack_log
    funpack_log.begin_run()
    model = _Model()
    _node(True)._build_tsr_wrapper(model, 0.5)
    _euler(model.model_options["model_function_wrapper"], torch.randn(1, 2, 3, 4, 4),
           schedule=torch.tensor([1.0, 0.0]))
    assert "1-step schedule" in capsys.readouterr().out


def test_decisiveness_reach_tells_a_weak_schedule_from_a_strong_one():
    simple4 = [1.0, 0.973, 0.923, 0.8, 0.0]
    beta57_8 = [1.0, 0.99, 0.969, 0.933, 0.872, 0.764, 0.563, 0.235, 0.0]
    assert tsr.reach(simple4, 0.9) < tsr.INERT_SHARE          # ~2%: can't be felt
    assert tsr.reach(beta57_8, 0.95) > tsr.INERT_SHARE * 3    # a real push
    assert tsr.reach([1.0, 0.0], 0.9) == 0.0                  # one step: nothing to carry into
    assert tsr.reach(simple4, 1.0) == 0.0                     # k = 1 is exactly off
