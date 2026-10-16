"""Block selection: which key blocks each query block is allowed to see.

Vendored into FunPack from ComfyUI-H3-SLA-Attention, which vendored it from
LightX2V (Apache-2.0). The provenance note and change list below are the
original author's and still describe this file exactly.

Vendored from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/utils/sla_util_blhd.py
  https://github.com/ModelTC/LightX2V

This is the whole of what "SLA" does at inference time. Mean-pool Q into blocks,
mean-pool a smoothed K into blocks, score the two against each other with one
small matmul, and keep the top ``topk_ratio`` fraction of key blocks per query
block. No weights, nothing trained, nothing to load -- the published SLA LoRA
adapts the *model* to tolerate the resulting sparsity, it does not parameterise
this step.

Three changes against upstream, marked FIX:

1. ``other=0.0`` on the masked load. The masked lanes feed ``tl.sum`` two lines
   later, and Triton leaves them undefined, so the final (partial) block of the
   sequence pooled whatever was in memory. Upstream fixed exactly this in the
   BHSD twin of this file and did not port the fix here.
2. ``max(1, ...)`` on the top-k count, so a short sequence keeps one key block
   rather than zero. Also upstream's BHSD-only fix.
3. Smooth-k is folded into the pooled result instead of being materialised.
   Pooling is a mean over L and the correction is constant along L, so
   ``pool(k - mu) == pool(k) - mu`` exactly. Upstream builds the whole smoothed
   copy of K, which at 768p/15s is a needless ~1.3 GB allocation.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .sla_attention import block_ranges, minus


@triton.jit
def _compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_b * L * H * D + idx_h * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # FIX vs upstream: other=0.0 -- these lanes are summed below.
    x = tl.load(
        X + x_offset + offs_l[:, None] * (H * D) + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    """``(B, L, H, D)`` -> ``(B, H, ceil(L/BLK), D)``, mean over each L block."""
    assert x.is_contiguous()
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    # fp32, not x.dtype. The Triton reduction already accumulates in fp32; only
    # the store rounds. Upstream stores bf16, which quantises the block *scores*
    # and makes top-k pick different blocks than exact arithmetic would -- and
    # at 85-90% sparsity only 3-4 blocks per query survive, so one wrong pick is
    # a large error. Keeping fp32 here costs a few MB and no measurable time.
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=torch.float32)

    grid = (L_BLOCKS, B * H)
    _compress_kernel[grid](x, x_mean, L, H, D, BLK)
    return x_mean


# stabilize_motion: a block chosen last step gets 5% of its row's best score, enough to settle a near tie,
# never to beat a clearly better block. Only the 8 choices nearest the cut-off are kept between steps:
# keeping whole tables for 50 layers is gigabytes at 768p, for a nudge that only matters at the edge.
_STICKY = 0.05
_HISTORY = 8


def get_block_map(q, k, topk_ratio, BLKQ=128, BLKK=128, protect=(), refs=(), ref_keep=None,
                  prev=None, sticky_from=0, remember=False):
    """Return ``(lut, topk, history)``: the key blocks each query block should attend to.

    ``q``/``k`` are ``(B, L, H, D)`` contiguous. ``lut`` comes back as
    ``(B, H, ceil(LQ/BLKQ), topk)`` int32, contiguous, ready for the kernel.

    ``protect`` (token spans) is in every query block's selection. For H3 that is the language
    tokens and the audio: plain top-k starves audio, ~1% of the packed sequence, so nothing makes a
    query keep any of it. Pinned blocks come on top of the top-k budget rather than displacing
    video, so video coverage is unchanged.

    ``refs`` with ``ref_keep`` (0..1): each query block also keeps that share of each reference
    span, its best-scoring blocks, again on top of the budget.

    ``prev``: last step's ``history`` for this same layer, nudged up before top-k from query
    token ``sticky_from`` on (the target video), so a near tie does not flip step to step and
    show as a faint double exposure on fast motion. ``remember``: also return ``history``, what
    to pass as ``prev`` next step (None otherwise).
    """
    pooled_q = mean_pool(q, BLKQ)
    # Smooth-k (SageAttention's trick), folded in rather than materialised.
    mu = k.mean(dim=1, dtype=torch.float32)                  # (B, H, D)
    pooled_k = mean_pool(k, BLKK) - mu[:, :, None, :]

    # GQA, for completeness -- H3 is MHA so this is a no-op there.
    num_q_heads, num_kv_heads = pooled_q.shape[1], pooled_k.shape[1]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        pooled_k = pooled_k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)      # (B, H, NQ, NK)

    NQ, NK = pooled_score.shape[-2:]
    sticky = pooled_score[..., min(NQ, (max(0, int(sticky_from)) + BLKQ - 1) // BLKQ):, :]
    if prev is not None and prev.shape[:3] == sticky.shape[:3]:
        bonus = sticky.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) * _STICKY
        sticky.scatter_add_(-1, prev.long(), bonus.expand(*prev.shape))

    # FIX vs upstream: keep at least one key block.
    topk = max(1, min(NK, int(topk_ratio * NK)))

    # Ranking them above everything else is what pins them; widening topk by
    # the same amount is what stops them evicting the blocks top-k chose.
    pinned = block_ranges(protect, BLKK, NK)
    extra = 0
    if ref_keep:
        for a, b in minus(block_ranges(refs, BLKK, NK), pinned):
            keep = max(1, min(b - a, math.ceil(ref_keep * (b - a))))
            pooled_score.scatter_(-1, torch.topk(pooled_score[..., a:b], keep, dim=-1, sorted=False).indices + a, float("inf"))
            extra += keep
    for a, b in pinned:
        pooled_score[..., a:b] = float("inf")
        extra += b - a
    topk = min(NK, topk + extra)

    chosen = torch.topk(pooled_score, topk, dim=-1, sorted=False)
    lut = chosen.indices.to(torch.int32).contiguous()
    if not remember:
        return lut, topk, None
    rows = sticky.shape[-2]
    idx, val = chosen.indices[..., NQ - rows:, :], chosen.values[..., NQ - rows:, :]
    if topk > _HISTORY:      # the choices nearest the cut-off: the only ones that can flip
        edge = torch.topk(val, _HISTORY, dim=-1, largest=False, sorted=False).indices
        idx = torch.gather(idx, -1, edge)
    return lut, topk, idx.to(torch.int32).contiguous()
