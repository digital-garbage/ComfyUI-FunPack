"""FunPack's own chunk-causal H3: the cache, the chunk geometry, and the re-classed DiT.

H3's stock DiT does one dense forward over the whole packed sequence. Chunk-causal generation
needs a forward over a SLICE that attends to a cache of the slices already finished. This is
that machinery, owned rather than borrowed — no third-party package has to be installed for
the feature to exist.

The safety property everything rests on: with no cache passed, every forward defers to its
parent, so an ordinary generation cannot be degraded by any of this.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h3_causal as hc


# ── the cache: what a chunk is allowed to remember ──────────────────────────

def _cache(**kw):
    kw.setdefault("num_layers", 2)
    return hc.ChunkKVCache(**kw)


def _kv(rows, value):
    return torch.full((1, 2, rows, 4), float(value)), torch.full((1, 2, rows, 4), float(value))


def test_an_empty_cache_has_nothing_to_read():
    assert _cache().read(0, 0) == (None, None)


def test_a_committed_chunk_is_readable():
    cache = _cache()
    k, v = _kv(3, 1.0)
    cache.write(0, 0, k, v)
    cache.finish_chunk(0)
    got_k, got_v = cache.read(0, 1)
    assert got_k.shape[-2] == 3 and got_v.shape[-2] == 3


def test_layers_do_not_share_their_cache():
    """Layer 7 reading layer 3's keys would be silent nonsense, not a crash."""
    cache = _cache(num_layers=4)
    cache.write(0, 0, *_kv(3, 1.0))
    cache.finish_chunk(0)
    assert cache.read(1, 1) == (None, None)


def test_chunks_are_read_back_in_order():
    cache = _cache()
    for index, value in enumerate((1.0, 2.0)):
        cache.write(0, index, *_kv(2, value))
        cache.finish_chunk(index)
    keys, _v = cache.read(0, 2)
    assert keys[0, 0, 0, 0] == 1.0 and keys[0, 0, 2, 0] == 2.0


def test_sink_chunks_are_never_evicted():
    """The prompt and the opening shot are what hold a character together over a long clip."""
    cache = _cache(sink=1, window=1)
    for index in range(6):
        cache.write(0, index, *_kv(2, index))
        cache.finish_chunk(index)
    assert 0 in cache.retained_indices(6)


def test_the_window_keeps_only_the_most_recent():
    cache = _cache(sink=1, window=2)
    for index in range(6):
        cache.write(0, index, *_kv(2, index))
        cache.finish_chunk(index)
    assert cache.retained_indices(6) == [0, 4, 5]


def test_a_zero_window_leans_entirely_on_the_sink():
    cache = _cache(sink=2, window=0)
    for index in range(5):
        cache.write(0, index, *_kv(2, index))
        cache.finish_chunk(index)
    assert cache.retained_indices(5) == [0, 1]


def test_no_window_remembers_everything():
    cache = _cache(sink=1, window=None)
    for index in range(5):
        cache.write(0, index, *_kv(2, index))
        cache.finish_chunk(index)
    assert cache.retained_indices(5) == [0, 1, 2, 3, 4]


def test_evicted_chunks_are_actually_dropped():
    """Retention that only hides rows would grow memory with clip length, which is the thing
    chunking exists to avoid."""
    cache = _cache(sink=1, window=1)
    for index in range(8):
        cache.write(0, index, *_kv(64, index))
        cache.finish_chunk(index)
    assert len(cache._store) <= 3


def test_a_negative_sink_is_refused():
    with pytest.raises(hc.CacheError):
        _cache(sink=-1)


def test_zero_layers_is_refused():
    with pytest.raises(hc.CacheError):
        hc.ChunkKVCache(0)


def test_clearing_forgets_everything():
    cache = _cache()
    cache.write(0, 0, *_kv(2, 1.0))
    cache.finish_chunk(0)
    cache.clear()
    assert cache.committed_chunks == 0 and cache.read(0, 1) == (None, None)


def test_offloaded_entries_leave_the_compute_device():
    """The retained K/V of a long clip is tens of GiB; holding it on the card is what makes a
    long clip impossible rather than merely slow."""
    cache = _cache(offload=True)
    cache.write(0, 0, *_kv(2, 1.0))
    assert cache._store[(0, 0)][0].device.type == "cpu"


# ── the chunk cut ───────────────────────────────────────────────────────────

def test_video_cuts_every_five_latents():
    bounds = hc.chunk_bounds(latent_t=12, audio_t=100)
    assert [(a, b) for a, b, _c, _d in bounds] == [(0, 5), (5, 10), (10, 12)]


def test_every_latent_lands_in_exactly_one_chunk():
    bounds = hc.chunk_bounds(latent_t=22, audio_t=180)
    video, audio = [], []
    for vs, ve, as_, ae in bounds:
        video.extend(range(vs, ve))
        audio.extend(range(as_, ae))
    assert video == list(range(22)) and audio == list(range(180))


def test_the_audio_cadence_falls_out_of_the_clock():
    """40 latents/s against a fixed 24 fps, 17 frames per chunk. The 28/29 alternation is a
    CONSEQUENCE — hard-coding it would drift silently if the grid ever moved."""
    frames = 17 * 11 + 5
    bounds = hc.chunk_bounds(latent_t=5 * 11 + 2, audio_t=int(round(frames / 24 * 40)))
    per_chunk = [ae - as_ for _a, _b, as_, ae in bounds]
    assert per_chunk[:4] == [28, 29, 28, 28]


def test_the_tail_chunk_is_kept_not_dropped():
    """H3's grid is 5k+2 latents, so there is ALWAYS a two-latent tail. Dropping it would cut
    the last five frames off every clip."""
    bounds = hc.chunk_bounds(latent_t=17, audio_t=140)
    assert bounds[-1][1] - bounds[-1][0] == 2


# ── stereo rows: the trap in packed audio ───────────────────────────────────

def test_a_chunks_audio_rows_are_two_spans_not_one():
    """pack_audio is channel-major over the WHOLE clip — all of the left channel, then all of
    the right. Treating a chunk's rows as one contiguous slice hands the model the wrong half
    of the stereo field, quietly."""
    rows = hc.chunk_rows(audio_t=100, audio_start=28, audio_stop=57)
    assert rows[:29] == list(range(28, 57))
    assert rows[29:] == list(range(128, 157))


def test_the_row_count_is_two_per_latent():
    rows = hc.chunk_rows(audio_t=100, audio_start=0, audio_stop=28)
    assert len(rows) == 56


def test_the_first_chunk_starts_at_both_channel_origins():
    rows = hc.chunk_rows(audio_t=50, audio_start=0, audio_stop=4)
    assert rows == [0, 1, 2, 3, 50, 51, 52, 53]


# ── re-classing, not wrapping ───────────────────────────────────────────────

def test_a_block_is_re_classed_in_place_and_keeps_its_submodules():
    """Wrapping renames the state-dict keys of everything inside, which has broken unpatching
    here before. Re-classing changes only which forward runs."""
    class FakeAttention:
        pass

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttention()
            self.norm1 = "the-real-norm"

    class CausalAttn(FakeAttention):
        pass

    class CausalBlock(FakeBlock):
        def __init__(self):
            raise AssertionError("re-classing must not re-run __init__")

    block = FakeBlock()
    attn, norm = block.attn, block.norm1
    out = hc._to_causal_block(block, CausalBlock, CausalAttn)
    assert out is block
    assert out.attn is attn and out.norm1 is norm
    assert isinstance(out, CausalBlock) and isinstance(out.attn, CausalAttn)
