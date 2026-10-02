"""The phrase probe on the real tiny H3: it measures, and it changes nothing."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _needs_comfy(comfyui):
    """Imports comfy."""


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    from modules.sampling.phrase_probe import measure
    monkeypatch.setattr(measure, "SWITCH", tmp_path / "switch")
    monkeypatch.setattr(measure, "LATEST", tmp_path / "latest.json")


PHRASES = {"base": 0, "cond_len": 5, "spans": [(1, 3), (3, 5)]}


def _load(tiny):
    from comfy.patcher_extension import WrappersMP
    from modules.sampling.modifiers.nodes import FunPackLoadModifiers
    patched, status = FunPackLoadModifiers.execute(tiny.patcher, {}).result
    wraps = [w for ws in patched.wrappers.get(WrappersMP.APPLY_MODEL, {}).values() for w in ws]
    outer = [w for ws in patched.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).values() for w in ws]
    return patched, (wraps[-1] if wraps else None), outer


def _executor(tiny, patched, log):
    from conftest import packed_av

    def executor(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kw):
        to = dict(transformer_options or {})
        log.append(to)
        video, audio = tiny.run(patched, sigma=float(t.max()),
                                options={k: v for k, v in to.items() if k == "funpack_probe"})
        return packed_av(video, audio)[0]
    return executor


def _run(outer, fn):
    call = fn
    for o in reversed(outer):
        call = (lambda w, inner: (lambda: w(lambda: inner())))(o, call)
    return call()


def _go(tiny, patched, wrap, outer, to, steps=2):
    from conftest import packed_av
    x0, shapes = packed_av(tiny.video, tiny.audio)
    seen = []
    executor = _executor(tiny, patched, seen)
    s = torch.tensor([1.0, 0.5, 0.0])
    outs = _run(outer, lambda: [wrap(executor, x0, torch.tensor([[1.0, 0.5][i]]), None, None, None,
                                     {"sample_sigmas": s, "sigmas": s[i:i + 1], "cond_or_uncond": [0], **to},
                                     latent_shapes=shapes) for i in range(steps)])
    return outs, seen


def test_off_it_installs_nothing_and_stores_nothing(tiny_h3):
    from modules.sampling.phrase_probe import measure
    patched, wrap, outer = _load(tiny_h3)
    assert wrap is None and outer == [] and measure.latest() is None


def test_recording_measures_each_phrase_per_block_and_returns_the_untouched_answer(tiny_h3):
    from modules.sampling.phrase_probe import measure
    patched, _none, outer = _load(tiny_h3)
    base_outs, _ = _go(tiny_h3, patched, lambda ex, *a, **k: ex(*a, **k), outer,
                       {"funpack_markup_phrases": PHRASES})
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    outs, seen = _go(tiny_h3, patched, wrap, outer, {"funpack_markup_phrases": PHRASES})
    assert len(seen) == 2 * (1 + 2)                                  # 1 visible + 2 masked, per call
    probes = [bool(t.get("funpack_probe")) for t in seen]
    assert probes == [False, True, True] * 2                         # masked passes are flagged
    assert all(torch.equal(a, b) for a, b in zip(outs, base_outs))   # the clip is not changed
    res = measure.latest()
    n = len(tiny_h3.patcher.model.diffusion_model.blocks)
    assert res["model_calls"] == 2 and res["blocks"] == list(range(n))
    assert [p["tokens"] for p in res["phrases"]] == [2, 2]
    for p in res["phrases"]:
        assert len(p["read"]) == len(p["response"]) == n
        assert all(v is not None and 0 <= v <= 1 for v in p["read"])
        assert all(v is not None and v > 0 for v in p["response"])   # hiding a phrase moves the rows


def test_no_phrase_text_is_ever_in_the_saved_file(tiny_h3):
    from modules.sampling.phrase_probe import measure
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    _go(tiny_h3, patched, wrap, outer, {"funpack_markup_phrases": PHRASES})
    keys = set(measure.latest()) | {k for p in measure.latest()["phrases"] for k in p}
    assert keys == {"when", "blocks", "model_calls", "phrases", "phrase", "tokens", "read", "response"}


def test_a_run_with_no_phrases_says_so_and_costs_nothing(tiny_h3):
    from core import log
    from modules.sampling.phrase_probe import measure
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    log.new_run()
    _outs, seen = _go(tiny_h3, patched, wrap, outer, {})
    assert len(seen) == 2 and measure.latest() is None
    assert any("no phrases to measure" in (measure.problem or "") for _ in [0])
    assert any("Inactive" in e["message"] for e in log.history())


def test_a_probe_call_from_another_feature_is_not_probed_again(tiny_h3):
    from modules.sampling.phrase_probe import measure
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    _outs, seen = _go(tiny_h3, patched, wrap, outer,
                      {"funpack_markup_phrases": PHRASES, "funpack_probe": True})
    assert len(seen) == 2


def test_a_failing_masked_pass_costs_nothing(tiny_h3):
    from conftest import packed_av
    from modules.sampling.phrase_probe import measure
    measure.set_enabled(True)
    patched, wrap, outer = _load(tiny_h3)
    x0, shapes = packed_av(tiny_h3.video, tiny_h3.audio)
    good = _executor(tiny_h3, patched, [])

    def executor(x, t, *a, **kw):
        to = kw.get("transformer_options") or (a[3] if len(a) > 3 else {})
        if (to or {}).get("funpack_probe"):
            raise RuntimeError("masked pass broke")
        return good(x, t, *a, **kw)
    s = torch.tensor([1.0, 0.5, 0.0])
    out = _run(outer, lambda: wrap(executor, x0, torch.tensor([[1.0]]), None, None, None,
                                   {"sample_sigmas": s, "sigmas": s[0:1], "cond_or_uncond": [0],
                                    "funpack_markup_phrases": PHRASES}, latent_shapes=shapes))
    assert out.shape == x0.shape


def test_phrases_are_placed_by_the_markup_nodes_published_spans():
    from modules.sampling.phrase_probe import measure
    assert measure.phrases_of({"base": 10, "cond_len": 30, "spans": [(0, 3), (0, 3), (5, 8), (20, 30)]}) \
        == [(10, 13), (15, 18)]                                       # deduped; (30..40) does not fit
    assert measure.phrases_of(None) == [] and measure.phrases_of({"spans": "x"}) == []
