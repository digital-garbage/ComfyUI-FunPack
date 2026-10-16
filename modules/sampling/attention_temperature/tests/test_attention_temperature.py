"""The randomizer, through the real modifier loader onto a real H3 forward."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _load(tiny, **values):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    settings = {"attention_temperature": {"enabled": True, **values}}
    return FunPackLoadModifiers.execute(tiny.patcher, settings).result


def test_it_changes_the_output(tiny_h3):
    base = tiny_h3.run()
    patched, status = _load(tiny_h3, amount=-0.5, blocks="1-2")
    assert "attention_temperature: 0.5x (sharper) at blocks 1-2" in status

    out = tiny_h3.run(patched)
    assert not torch.allclose(out[0], base[0]), "temperature ran and changed nothing"


def test_blocks_outside_the_list_see_the_original_query(tiny_h3):
    from core import dit_hooks
    patched, _ = _load(tiny_h3, amount=1.0, blocks="3")
    seen = []

    def spy(func, q, k, v, *a, **kw):
        seen.append(float(q.abs().mean()))
        return func(q, k, v, *a, **kw)

    # Installed on top, so it sees q BEFORE the temperature override divides it,
    # and a second spy underneath sees it after.
    under = []

    def spy_under(func, q, k, v, *a, **kw):
        under.append(float(q.abs().mean()))
        return func(q, k, v, *a, **kw)

    probe = patched.clone()
    dit_hooks.add_attention_override(probe, "test.spy", spy)
    tiny_h3.run(probe)
    # Rebuild with the spy UNDER the temperature: clone the base, spy first.
    low = tiny_h3.patcher.clone()
    dit_hooks.add_attention_override(low, "test.spy", spy_under)
    from modules.sampling import attention_temperature as m
    m.install(low, {"enabled": True, "amount": 1.0, "blocks": "3"}, key="funpack.attention_temperature")
    tiny_h3.run(low)
    # call order: refiner, blocks 0..3 -> only the last is halved
    ratios = [u / s for u, s in zip(under, seen)]
    assert [round(r, 4) for r in ratios] == [1.0, 1.0, 1.0, 1.0, 0.5]


def test_off_or_zero_installs_nothing(tiny_h3):
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    _p, status = FunPackLoadModifiers.execute(
        tiny_h3.patcher, {"attention_temperature": {"enabled": True, "amount": 0.0}}).result
    assert "attention_temperature" not in status.splitlines()[0]


def test_a_block_list_with_nothing_usable_is_refused_not_silently_inert(tiny_h3):
    _p, status = _load(tiny_h3, amount=0.5, blocks="40-49")
    assert "attention_temperature: failed to install" in status


def test_loading_twice_does_not_double_the_effect(tiny_h3):
    once, _ = _load(tiny_h3, amount=-0.5, blocks="1")
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    twice, status = FunPackLoadModifiers.execute(
        once, {"attention_temperature": {"enabled": True, "amount": -0.5, "blocks": "1"}}).result
    assert torch.allclose(tiny_h3.run(once)[0], tiny_h3.run(twice)[0])
