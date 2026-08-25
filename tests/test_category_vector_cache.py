"""The eight category descriptions are constant, so they are embedded once per encoder.

CATEGORY_DESCRIPTIONS is a class attribute; the vectors depend only on the encoder. Embedding
them every run cost eight full encoder passes to arrive at the same eight vectors — on H3,
eight passes of Qwen3-VL-32B.
"""
import gc
import sys

import pytest
import torch

sys.path.insert(0, ".")


class _Clip:
    """Counts how many times it was asked to encode."""
    def __init__(self):
        self.calls = 0


@pytest.fixture
def refiner():
    from conditioning import FunPackVideoRefinerV2
    FunPackVideoRefinerV2._V2_CATEGORY_VECTOR_CACHE.clear()
    return FunPackVideoRefinerV2()


@pytest.fixture(autouse=True)
def _stub_encode(monkeypatch):
    from conditioning import FunPackVideoRefinerV2

    def fake(self, clip, text, encode_cache=None, purpose="prompt", **kw):
        clip.calls += 1
        return torch.ones(1, 4, 8), {}, ""
    monkeypatch.setattr(FunPackVideoRefinerV2, "_v2_encode_prompt", fake)


def test_the_second_run_encodes_nothing(refiner):
    clip = _Clip()
    refiner._v2_category_vectors(clip, encode_cache={})
    first = clip.calls
    assert first == len(refiner.CATEGORY_DESCRIPTIONS)
    refiner._v2_category_vectors(clip, encode_cache={})      # a fresh run's cache
    assert clip.calls == first


def test_a_different_encoder_is_embedded_again(refiner):
    a, b = _Clip(), _Clip()
    refiner._v2_category_vectors(a, encode_cache={})
    refiner._v2_category_vectors(b, encode_cache={})
    assert b.calls == len(refiner.CATEGORY_DESCRIPTIONS)


def test_the_cache_does_not_keep_an_encoder_alive(refiner):
    """A held encoder is a text encoder pinned in VRAM. Saving eight passes is not worth it."""
    clip = _Clip()
    refiner._v2_category_vectors(clip, encode_cache={})
    entry = next(iter(type(refiner)._V2_CATEGORY_VECTOR_CACHE.values()))
    del clip
    gc.collect()
    assert entry[0]() is None


def test_a_recycled_id_is_not_mistaken_for_the_same_encoder(refiner):
    """ids are reused once an object is freed; a stale hit would classify a phrase against
    another encoder's geometry."""
    clip = _Clip()
    refiner._v2_category_vectors(clip, encode_cache={})
    stale_id = next(iter(type(refiner)._V2_CATEGORY_VECTOR_CACHE))
    del clip
    gc.collect()
    other = _Clip()
    type(refiner)._V2_CATEGORY_VECTOR_CACHE[id(other)] = \
        type(refiner)._V2_CATEGORY_VECTOR_CACHE.get(stale_id, (lambda: None, {}))
    refiner._v2_category_vectors(other, encode_cache={})
    assert other.calls == len(refiner.CATEGORY_DESCRIPTIONS)
