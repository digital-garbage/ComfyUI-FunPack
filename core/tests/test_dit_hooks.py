"""Chained block hooks / attention overrides, on a REAL H3 forward."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


def _clone(tiny):
    return tiny.patcher.clone()


def test_two_block_hooks_on_one_block_both_run_and_the_output_is_the_block(tiny_h3):
    from core import dit_hooks
    base = tiny_h3.run()
    p = _clone(tiny_h3)
    seen = []

    def hook(name):
        def h(args, extra):
            seen.append(name)
            return extra["original_block"](args)
        return h

    dit_hooks.add_block_hook(p, "funpack.a", 1, hook("a"))
    dit_hooks.add_block_hook(p, "funpack.b", 1, hook("b"))
    out = tiny_h3.run(p)
    assert seen == ["b", "a"], "the second hook replaced the first instead of chaining"
    assert torch.allclose(out[0], base[0]) and torch.allclose(out[1], base[1])


def test_two_attention_overrides_both_see_every_call(tiny_h3):
    from core import dit_hooks
    p = _clone(tiny_h3)
    calls = {"a": 0, "b": 0}

    def ov(name):
        def o(func, q, k, v, *a, **kw):
            calls[name] += 1
            return func(q, k, v, *a, **kw)
        return o

    dit_hooks.add_attention_override(p, "funpack.a", ov("a"))
    dit_hooks.add_attention_override(p, "funpack.b", ov("b"))
    tiny_h3.run(p)
    # 4 DiT blocks + 1 token-refiner block
    assert calls == {"a": 5, "b": 5}


def test_strip_removes_ours_and_puts_back_a_foreign_hook_underneath(tiny_h3):
    from core import dit_hooks, patching
    p = _clone(tiny_h3)
    foreign = []

    def theirs(args, extra):
        foreign.append(1)
        return extra["original_block"](args)

    dit = p.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault("dit", {})
    dit[("double_block", 0)] = theirs
    dit_hooks.add_block_hook(p, "funpack.x", 0, lambda a, e: e["original_block"](a))
    dit_hooks.add_attention_override(p, "funpack.x", lambda f, *a, **k: f(*a, **k))

    assert patching.strip(p, "funpack.") == 2
    to = p.model_options["transformer_options"]
    assert to["patches_replace"]["dit"][("double_block", 0)] is theirs
    assert "optimized_attention_override" not in to
    tiny_h3.run(p)
    assert foreign == [1]


def test_a_failing_block_hook_is_dropped_and_the_run_matches_an_unhooked_one(tiny_h3):
    from core import dit_hooks, patching
    base = tiny_h3.run()
    p = _clone(tiny_h3)
    dropped = patching.Dropped()
    guarded = patching.GuardedPatcher(p, "funpack.bad", dropped)

    def bad(args, extra):
        raise RuntimeError("boom")

    def bad_attn(func, *a, **k):
        raise RuntimeError("boom")

    dit_hooks.add_block_hook(guarded, "funpack.bad", 2, bad)
    dit_hooks.add_attention_override(guarded, "funpack.bad", bad_attn)
    out = tiny_h3.run(p)
    assert "funpack.bad" in dropped
    assert torch.allclose(out[0], base[0])


@pytest.mark.parametrize("spec,expect,problems", [
    ("1", [1], 0), ("0-2", [0, 1, 2], 0), ("3,1", [1, 3], 0),
    ("2-9", [2, 3], 1), ("x", [], 1), ("", [], 0), ("-1", [], 1),
])
def test_parse_blocks_reports_what_it_ignores(spec, expect, problems):
    from core import dit_hooks
    got, said = dit_hooks.parse_blocks(spec, 4)
    assert got == expect and len(said) == problems
