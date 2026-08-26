"""Chunk plan and modulation, checked against the REAL ComfyUI MiniMax H3.

Run in a clean subprocess by `test_h3_causal_plan.py`, because the suite stubs `comfy` and
the real package cannot be imported alongside the stub. This is the test that catches a
ComfyUI update moving the packed layout out from under us.
"""
import sys

TEXT_LEN, LATENT_T, LAT_H, LAT_W, AUDIO_T = 120, 57, 32, 48, 320


def main(funpack, comfy_root):
    sys.path.insert(0, comfy_root)
    sys.path.insert(0, funpack)
    import torch
    from comfy.ldm.minimax.model import (VISUAL_COND_TIMESTEP, PackedLayout, _frame_grid,
                                         pack_audio)

    import h3_causal as hc

    frame_rows = _frame_grid(LAT_H, LAT_W)[0].shape[0]

    def plan_for(**kw):
        layout = PackedLayout(TEXT_LEN, LATENT_T, LAT_H, LAT_W, AUDIO_T, **kw)
        return hc.build_plan(layout, LATENT_T, AUDIO_T)

    def tiles(plan):
        rows = torch.cat([plan.prefix_rows] + [plan.chunk(i)[0] for i in range(plan.n_chunks)])
        return sorted(rows.tolist()) == list(range(plan.layout.seq_len))

    # 192 pixel frames is 57 video latents: eleven 17-frame chunks and a 5-frame tail.
    plan = plan_for()
    assert plan.n_chunks == 12, plan.n_chunks
    assert tiles(plan), "plain clip does not tile"
    assert [k for _, _, k in plan.prefix_runs] == ["text"]

    # a chunk's audio rows are two spans, and `pack_audio` on the chunk's own slice produces
    # them in exactly that order — which is what lets forward_chunk pack the slice directly
    rows, (audio_run, video_run) = plan.chunk(2)
    audio = rows[audio_run[0]:audio_run[1]].tolist()
    half = len(audio) // 2
    assert audio[half] - audio[0] == AUDIO_T, "the two stereo spans are not one clip apart"
    v_start, v_stop, a_start, a_stop = plan.bounds[2]
    packed = pack_audio(torch.zeros(1, 32, 2, a_stop - a_start))
    assert packed.shape[0] == len(audio), (packed.shape, len(audio))
    assert video_run[1] - video_run[0] == (v_stop - v_start) * frame_rows

    # conditioning rows keep the layout upstream packed for them
    refs = plan_for(refs=[{"kind": "image", "latent_h": LAT_H, "latent_w": LAT_W}])
    assert [k for _, _, k in refs.prefix_runs] == ["text", "ref_img"]
    assert tiles(refs), "ref clip does not tile"
    for i in range(refs.n_chunks):
        assert {k for _, _, k in refs.chunk(i)[1]} == {"audio", "video"}

    pins = plan_for(keyframes=[{"resolved_frame_index": 0}], frame_count=192)
    assert [k for _, _, k in pins.prefix_runs] == ["text", "cond"]
    assert tiles(pins), "pinned clip does not tile"

    # modulation: only the kinds present get a timestep row, and identical times share one
    model_cls = hc._causal_classes()[2]
    seg_t, t_row, unique = model_cls._modulation(None, {"audio", "video"}, 0.8, 0.6, {})
    assert len(unique) == 2 and t_row[seg_t["video"]] != t_row[seg_t["audio"]]
    _, t_row, unique = model_cls._modulation(None, {"audio", "video"}, 0.5, 0.5, {})
    assert len(unique) == 1 and len(t_row) == 1
    seg_t, _, _ = model_cls._modulation(None, {"video", "cond"}, 0.1, 0.1, {})
    assert abs(seg_t["cond"] - VISUAL_COND_TIMESTEP) < 1e-9

    # the text span splits by tag runs; everything else is one run per kind
    seg_t, t_row, _ = model_cls._modulation(None, {"text"}, 0.8, 0.6, {})
    out = model_cls._mod_segments(None, [(0, 6, "text")], seg_t, t_row,
                                  torch.tensor([1, 1, 0, 0, 0, 1]))
    assert [(a, b, row % 3) for a, b, row in out] == [(0, 2, 1), (2, 5, 0), (5, 6, 1)]

    # `_mod_scale_shift` writes each run in place, so a gap would leave rows unmodulated
    rows, runs = plan.chunk(1)
    seg_t, t_row, _ = model_cls._modulation(None, {k for _, _, k in runs}, 0.8, 0.6, {})
    cursor = 0
    for a, b, _row in model_cls._mod_segments(None, runs, seg_t, t_row, None):
        assert a == cursor
        cursor = b
    assert cursor == rows.shape[0]

    _forward_phase()
    print("OK")


def _install_rope_fallback():
    """A pure-torch stand-in for the fused RMSNorm+rope kernel, for hosts whose comfy_kitchen
    build lacks it. Both lanes use the same one, so the dense-vs-causal comparison stays fair.
    """
    import torch

    import comfy.quant_ops

    if hasattr(comfy.quant_ops.ck, "rms_rope_split_half_"):
        return

    def rms_rope(q, k, table, qw, kw, epsilon=1e-5, rot_dim=None):
        half = table.shape[-3]
        rot = rot_dim or half * 2
        cos = table[..., 0, 0].squeeze(0).squeeze(-2)[:, None, :]
        sin = table[..., 1, 0].squeeze(0).squeeze(-2)[:, None, :]
        for tensor, weight in ((q, qw), (k, kw)):
            x = tensor[0]
            scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + epsilon)
            x.copy_(x * scale.to(x.dtype) * weight)
            lo, hi = x[..., :rot // 2].clone(), x[..., rot // 2:rot].clone()
            x[..., :rot // 2] = lo * cos - hi * sin
            x[..., rot // 2:rot] = lo * sin + hi * cos
        return q, k

    comfy.quant_ops.ck.rms_rope_split_half_ = rms_rope


def _forward_phase():
    """The real DiT, at toy width, driven through both lanes."""
    import copy

    import torch

    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout

    import h3_causal as hc

    _install_rope_fallback()
    text_len, latent_t, lat_h, lat_w, audio_t = 7, 12, 8, 8, 65
    torch.manual_seed(0)
    stock = MiniMaxH3Model(
        hidden_size=192, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
        attention_head_dim=96, ffn_hidden_size=192, text_dim=64, timestep_input_dim=16,
        time_embed_hidden_size=192, time_embed_dim=32, rope_inv_freq_len=16,
        dtype=torch.float32, device=torch.device("cpu"),
        operations=comfy.ops.disable_weight_init)
    for _name, param in stock.named_parameters():
        param.data.normal_(0, 0.02)
    stock.rope.inv_freq.copy_(1.0 / (10000 ** (torch.arange(16).float() / 16)))
    stock.eval()

    causal = copy.deepcopy(stock)
    keys = set(causal.state_dict().keys())
    ok, note = hc.make_causal(type("Patcher", (), {"model": type("M", (), {})()})())
    assert not ok, note                                   # nothing to re-class on a bare object
    attention_cls, block_cls, model_cls = hc._causal_classes()
    for i in range(len(causal.blocks)):
        hc._to_causal_block(causal.blocks[i], block_cls, attention_cls)
    causal.__class__ = model_cls
    # Re-classing must not rename a single weight: wrapping used to, and unpatching crashed.
    assert set(causal.state_dict().keys()) == keys, "re-classing renamed state-dict keys"

    video = torch.randn(1, 24, latent_t, lat_h, lat_w)
    audio = torch.randn(1, 32, 2, audio_t)
    context = torch.randn(1, text_len, 64)
    layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t)
    payload = {"layout": layout, "seed": 0}

    # THE safety property: with no cache passed, the causal model is the stock one.
    with torch.no_grad():
        a = stock._forward([video, audio], torch.tensor([800.0]), context,
                           minimax_payload=payload)
        b = causal._forward([video, audio], torch.tensor([800.0]), context,
                            minimax_payload=payload)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "the dense lane changed"

    plan = hc.build_plan(layout, latent_t, audio_t)
    cache = hc.ChunkKVCache(len(causal.blocks), sink=2, window=2,
                            device=torch.device("cpu"), offload=False)
    with torch.no_grad():
        causal.prefill_text(context, plan, cache, minimax_payload=payload)
        assert cache._store[(0, 0)][0].shape[-2] == text_len, "the prefix cached the wrong rows"
        for i in range(plan.n_chunks):
            v_start, v_stop, a_start, a_stop = plan.bounds[i]
            chunk_v = video[:, :, v_start:v_stop]
            chunk_a = audio[..., a_start:a_stop]
            out_v, out_a = causal.forward_chunk(
                chunk_v, chunk_a, plan, i, cache, video_sigma=0.8, audio_sigma=0.4,
                context=context, minimax_payload=payload)
            assert out_v.shape == chunk_v.shape, (out_v.shape, chunk_v.shape)
            assert out_a.shape == chunk_a.shape, (out_a.shape, chunk_a.shape)
            assert torch.isfinite(out_v).all() and torch.isfinite(out_a).all()
            causal.forward_chunk(chunk_v, chunk_a, plan, i, cache, video_sigma=0.0,
                                 audio_sigma=0.0, context=context, update_cache=True,
                                 minimax_payload=payload)
            cache.finish_chunk(plan.cache_index(i))
            rows, _ = plan.chunk(i)
            assert cache._store[(0, plan.cache_index(i))][0].shape[-2] == rows.shape[0]

    # An i2v clip: the anchor is prefilled as its OWN cache chunk, reading the text already
    # cached at 0, so it can be evicted by `sink` without taking the prompt with it.
    pinned_layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                 keyframes=[{"resolved_frame_index": 0}], frame_count=39)
    pinned_plan = hc.build_plan(pinned_layout, latent_t, audio_t)
    assert pinned_plan.cond_rows.numel(), "the keyframe produced no conditioning rows"
    pinned_payload = {"layout": pinned_layout, "seed": 0,
                      "cond_video_latents": [torch.randn(1, 24, 1, lat_h, lat_w)]}
    pinned_cache = hc.ChunkKVCache(len(causal.blocks), sink=2, window=2,
                                   device=torch.device("cpu"), offload=False)
    with torch.no_grad():
        causal.prefill_text(context, pinned_plan, pinned_cache,
                            minimax_payload=pinned_payload)
        assert pinned_cache._store[(0, 0)][0].shape[-2] == text_len
        assert (pinned_cache._store[(0, 1)][0].shape[-2]
                == pinned_plan.cond_rows.shape[0]), "the anchor did not cache on its own"
        v_start, v_stop, a_start, a_stop = pinned_plan.bounds[0]
        out_v, out_a = causal.forward_chunk(
            video[:, :, v_start:v_stop], audio[..., a_start:a_stop], pinned_plan, 0,
            pinned_cache, video_sigma=0.8, audio_sigma=0.4, context=context,
            minimax_payload=pinned_payload)
        assert torch.isfinite(out_v).all() and torch.isfinite(out_a).all()
    # sink 1 releases the anchor and keeps the prompt; sink 2 keeps both
    pinned_cache.sink = 1
    assert pinned_cache.retained_indices(6) == [0], pinned_cache.retained_indices(6)
    pinned_cache.sink = 2
    assert pinned_cache.retained_indices(6) == [0, 1], pinned_cache.retained_indices(6)

    # A ComfyUI without the newer `time_shift_slope` must still build the lane. Importing it
    # like the rest made one missing symbol refuse the whole feature on 0.34.0, and the
    # refusal then blamed a setting — this is that regression.
    from comfy.ldm.minimax.model import time_shift_slope as _slope
    del sys.modules["comfy.ldm.minimax.model"].time_shift_slope
    try:
        old_attn, old_block, older = hc._causal_classes()
        assert hasattr(older, "forward_chunk"), "the lane refused itself without the slope"
        stripped = copy.deepcopy(stock)
        for i in range(len(stripped.blocks)):
            hc._to_causal_block(stripped.blocks[i], old_block, old_attn)
        stripped.__class__ = older
        older_cache = hc.ChunkKVCache(len(stripped.blocks), sink=2, window=2,
                                      device=torch.device("cpu"), offload=False)
        with torch.no_grad():
            stripped.prefill_text(context, plan, older_cache, minimax_payload=payload)
            v_start, v_stop, a_start, a_stop = plan.bounds[0]
            out_v, out_a = stripped.forward_chunk(
                video[:, :, v_start:v_stop], audio[..., a_start:a_stop], plan, 0, older_cache,
                video_sigma=0.8, audio_sigma=0.4, context=context, minimax_payload=payload)
        assert torch.isfinite(out_v).all() and torch.isfinite(out_a).all()
    finally:
        sys.modules["comfy.ldm.minimax.model"].time_shift_slope = _slope

    # a clip whose streams disagree about its length is refused by name, not deep in a block
    try:
        hc.build_plan(PackedLayout(text_len, latent_t, lat_h, lat_w, 20), latent_t, 20)
    except hc.CacheError as error:
        assert "does not cover the picture" in str(error), error
    else:
        raise AssertionError("a silent chunk was not refused")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
