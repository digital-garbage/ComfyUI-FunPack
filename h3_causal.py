"""FunPack's own chunk-causal MiniMax H3: a KV cache, and a DiT that reads it.

WHY THIS EXISTS
---------------
H3's stock DiT does ONE dense forward over the whole packed sequence. Chunk-causal generation
needs something it does not have: a forward over a slice of the clip that attends to a cache
of the slices already finished. That is the machinery here.

Every class is a SUBCLASS of its ComfyUI counterpart, with the same module names and the same
state dict, so:

* the weights load with no remapping and no conversion;
* a LoRA trained for the chunk-causal pattern attaches through ComfyUI's ordinary LoRA path
  (its adapter already reads PEFT `lora_A`/`lora_B` keys);
* with no cache passed, every forward defers to its parent and is bit-identical to stock H3.

That last point is the safety property. `cache=None` is the dense lane, unchanged, so nothing
here can degrade an ordinary generation.

GEOMETRY IS BORROWED, NOT REDERIVED
-----------------------------------
A chunk's RoPE positions must be the ones the dense layout would have given those same rows,
or the model is being told the clip is a different shape than it is. So the FULL clip's grid
is built with upstream's own `_frame_grid` / `_video_grid` / `_audio_grid` and then SLICED per
chunk. An upstream change to the grid is inherited rather than silently disagreed with.

THE LORA IS NOT OPTIONAL
------------------------
Chunk-causal attention is a pattern the base H3 weights have never seen — they were trained
with every row attending every other row. Running this without an adapter trained for it
produces output, and the output is out of distribution. The loader says so.
"""

from __future__ import annotations

import torch

#: video latents per chunk, and the pixel frames that many latents cover (1+4+4+4+4)
VIDEO_LATENTS_PER_CHUNK = 5
FRAMES_PER_CHUNK = 17
#: audio latents per pixel frame — 40/s against the model's fixed 24 fps
AUDIO_LATENTS_PER_FRAME = 40.0 / 24.0

#: AdaLN modality tags, from the dense model's own seg_tag
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


class CacheError(RuntimeError):
    """The cache was asked for something it cannot honestly answer."""


class ChunkKVCache:
    """Per-layer K/V of the chunks already finished, with sink + window eviction.

    Two kinds of memory, and they do different jobs:

    `sink` chunks are pinned from the very start and never evicted. Chunk 0 is the text, so a
    sink of 2 means the prompt plus the opening shot stay visible to every later chunk — this
    is what holds a character together across a long clip.

    `window` is the most recent chunks besides those, which is short-term continuity: motion
    and lighting carrying from one moment to the next.

    Anything in between is dropped, which is what keeps memory flat instead of growing with
    clip length. `window=None` keeps everything.
    """

    def __init__(self, num_layers, sink=2, window=2, device=None, offload=True):
        if int(num_layers) <= 0:
            raise CacheError(f"num_layers must be positive, got {num_layers!r}")
        if int(sink) < 0:
            raise CacheError(f"sink must be >= 0, got {sink!r}")
        if window is not None and int(window) < 0:
            raise CacheError(f"window must be None or >= 0, got {window!r}")
        self.num_layers = int(num_layers)
        self.sink = int(sink)
        self.window = None if window is None else int(window)
        self.device = device
        # The retained K/V of a long clip is measured in tens of GiB on the card. Holding it
        # on the host and copying one chunk back per layer trades bandwidth for the ability to
        # run at all.
        self.offload = bool(offload)
        self.committed_chunks = 0
        self._store = {}          # (layer, chunk_index) -> (k, v)

    def _keep(self, chunk_index, next_index):
        if chunk_index < self.sink:
            return True
        if self.window is None:
            return True
        return chunk_index > next_index - 1 - self.window

    def retained_indices(self, next_index):
        """Which committed chunks are visible to the chunk about to be generated."""
        return [i for i in range(self.committed_chunks) if self._keep(i, next_index)]

    def read(self, layer, next_index):
        """The retained K and V for one layer, in chunk order, or (None, None)."""
        parts = [self._store.get((layer, i)) for i in self.retained_indices(next_index)]
        parts = [p for p in parts if p is not None]
        if not parts:
            return None, None
        keys = torch.cat([p[0] for p in parts], dim=-2)
        values = torch.cat([p[1] for p in parts], dim=-2)
        if self.device is not None:
            keys, values = keys.to(self.device), values.to(self.device)
        return keys, values

    def write(self, layer, chunk_index, keys, values):
        """Store one layer's K/V for a finished chunk."""
        if self.offload:
            keys, values = keys.detach().to("cpu"), values.detach().to("cpu")
        else:
            keys, values = keys.detach(), values.detach()
        self._store[(layer, int(chunk_index))] = (keys, values)

    def finish_chunk(self, chunk_index):
        """Mark a chunk committed, and drop whatever is now out of both sink and window."""
        self.committed_chunks = max(self.committed_chunks, int(chunk_index) + 1)
        self.prune()

    def prune(self):
        keep = set(self.retained_indices(self.committed_chunks))
        for key in [k for k in self._store if k[1] not in keep]:
            del self._store[key]

    def clear(self):
        self._store.clear()
        self.committed_chunks = 0


def chunk_bounds(latent_t, audio_t):
    """(video_start, video_stop, audio_start, audio_stop) per chunk.

    Video cuts every ``VIDEO_LATENTS_PER_CHUNK`` latents. Audio is derived from the SHARED
    CLOCK — 40 latents per second against a fixed 24 fps, 17 pixel frames per chunk — so the
    familiar 28/29/28/28 cadence comes out as a consequence and stays correct if the grid ever
    moves, instead of being a remembered pattern that silently drifts.
    """
    latent_t, audio_t = int(latent_t), int(audio_t)
    if latent_t <= 0:
        return []
    bounds, cursor = [], 0
    starts = list(range(0, latent_t, VIDEO_LATENTS_PER_CHUNK))
    for index, video_start in enumerate(starts):
        video_stop = min(latent_t, video_start + VIDEO_LATENTS_PER_CHUNK)
        if index == len(starts) - 1:
            audio_stop = audio_t                       # the tail owns whatever is left
        else:
            done = FRAMES_PER_CHUNK * (index + 1)
            audio_stop = max(cursor, min(audio_t, int(round(done * AUDIO_LATENTS_PER_FRAME))))
        bounds.append((video_start, video_stop, cursor, audio_stop))
        cursor = audio_stop
    return bounds


def clip_grids(text_len, latent_t, latent_h, latent_w, audio_t):
    """The whole clip's video and audio position grids, and the rows per latent frame.

    Built with upstream's own helpers and sliced per chunk afterwards, so a chunk's RoPE
    angles are exactly the ones the dense layout would have given those rows. Re-deriving the
    geometry here instead would be a second source of truth for the clip's shape.
    """
    from comfy.ldm.minimax.model import _audio_grid, _frame_grid, _video_grid

    frame, w_grid = _frame_grid(int(latent_h), int(latent_w))
    video = _video_grid(int(latent_t), frame, float(text_len))
    audio = _audio_grid(float(text_len), int(audio_t),
                        float(w_grid[0]), float(w_grid[-1]))
    return video, audio, int(frame.shape[0])


def chunk_rows(audio_t, audio_start, audio_stop):
    """Which packed audio ROWS a chunk owns.

    `pack_audio` is channel-major over the WHOLE clip — all of the left channel, then all of
    the right — so a chunk's rows are two separate spans, not one. Treating them as one
    contiguous slice silently hands the model the wrong half of the stereo field.
    """
    left = list(range(int(audio_start), int(audio_stop)))
    right = [int(audio_t) + i for i in left]
    return left + right


def _causal_classes():
    """Build the three subclasses against whatever H3 this ComfyUI has.

    Deferred into a function because these subclass ``comfy.ldm.minimax.model``, which is not
    importable on a ComfyUI without H3 — and a node pack must still register there.
    """
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.minimax.model import (Attention, DiTBlock, MiniMaxH3Model,
                                         _mod_gate, _mod_scale_shift, optimized_attention)

    class CausalAttention(Attention):
        """Upstream attention, with the retained K/V prepended to this chunk's own.

        Q stays current-only: this chunk is what is being generated. K and V are the cache
        followed by this chunk, so the chunk sees all of itself and all of what was retained.
        There is no mask — that visibility IS the causal pattern, expressed by what is in the
        buffer rather than by hiding parts of a bigger one.

        Cached keys keep the absolute positions they were rotated with, so nothing is ever
        re-based. That is what makes the cache reusable across steps.
        """

        def forward(self, x, rope_freqs=None, transformer_options={},
                    cache=None, layer_idx=0, chunk_index=0, update_cache=False):
            if cache is None:
                # the dense lane, untouched
                return super().forward(x, rope_freqs=rope_freqs,
                                       transformer_options=transformer_options)
            s = x.shape[0]
            q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
            v = v.view(s, self.heads, self.head_dim)
            if rope_freqs is not None:
                q = q.view(1, s, self.heads, self.head_dim)
                k = k.view(1, s, self.heads, self.head_dim)
                qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
                rot = rope_freqs.shape[-3] * 2
                if comfy.model_management.in_training:
                    q, k = comfy.quant_ops.ck.rms_rope_split_half(
                        q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                else:
                    comfy.quant_ops.ck.rms_rope_split_half_(
                        q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
                q, k = q[0], k[0]
            else:
                q = self.q_norm(q.view(s, self.heads, self.head_dim))
                k = self.k_norm(k.view(s, self.heads, self.head_dim))
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            # Written BEFORE the cache is prepended: what this chunk contributes to later
            # chunks is its own K/V, never the concatenation it happened to attend over.
            if update_cache:
                cache.write(layer_idx, chunk_index, k, v)
            past_k, past_v = cache.read(layer_idx, chunk_index)
            if past_k is not None:
                k = torch.cat((past_k.to(dtype=k.dtype, device=k.device), k), dim=-2)
                v = torch.cat((past_v.to(dtype=v.dtype, device=v.device), v), dim=-2)
            out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True,
                                      transformer_options=transformer_options)
            return self.out_proj(out.squeeze(0))

    class CausalDiTBlock(DiTBlock):
        """Upstream block, with the cache threaded to its attention.

        A verbatim fork of the parent body — the norm / AdaLN / gate / MLP order is the
        checkpoint's contract and must not drift — differing only in the kwargs passed on.
        """

        def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={},
                    cache=None, layer_idx=0, chunk_index=0, update_cache=False):
            if cache is None:
                return super().forward(x, t_emb, mod_segments, rope_freqs,
                                       transformer_options=transformer_options)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
            h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
            attended = self.attn(h, rope_freqs=rope_freqs,
                                 transformer_options=transformer_options,
                                 cache=cache, layer_idx=layer_idx,
                                 chunk_index=chunk_index, update_cache=update_cache)
            x = _mod_gate(x, gate_msa, attended, mod_segments)
            h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
            return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)

    class CausalMiniMaxH3Model(MiniMaxH3Model):
        """Upstream H3 whose blocks can read a chunk cache. Same state dict, key for key.

        The blocks are rebuilt as causal ones after construction, ONE AT A TIME so peak memory
        stays at a single extra block rather than a second copy of all fifty. Each rebuilt
        block adopts the original's submodules by reference, so no weight is copied and no key
        is renamed — the hard rule this project already learned about wrapper modules.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for index in range(len(self.blocks)):
                self.blocks[index] = _to_causal_block(self.blocks[index],
                                                      CausalDiTBlock, CausalAttention)

    return CausalAttention, CausalDiTBlock, CausalMiniMaxH3Model


def _to_causal_block(block, block_cls, attention_cls):
    """Re-class one block and its attention in place, adopting every submodule by reference.

    NOT a wrapper module. Wrapping renames the state-dict keys of everything inside it, which
    has broken unpatching here before; re-classing changes only which `forward` runs.
    """
    block.attn.__class__ = attention_cls
    block.__class__ = block_cls
    return block


def make_causal(model):
    """Re-class an already-loaded H3 so its blocks can read a chunk cache. Returns (ok, note).

    This is why nothing here needs its own loader. The alternative — building the model
    through a different class — means reimplementing ComfyUI's whole diffusion-model load
    path, and attaching anything to the raw model before the ModelPatcher exists so its
    memory ledger counts it. Re-classing sidesteps all of that: no parameter is added, no key
    is renamed, nothing is copied. Only which `forward` runs changes.

    Idempotent, and a no-op on anything that is not H3.
    """
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None:
        return False, "no diffusion model to make causal"
    blocks = getattr(diffusion, "blocks", None)
    if not blocks:
        return False, "this model has no DiT blocks — not a MiniMax H3"
    if getattr(diffusion, "_funpack_causal", False):
        return True, "already chunk-causal"
    try:
        attention_cls, block_cls, model_cls = _causal_classes()
    except Exception as error:                        # noqa: BLE001
        return False, f"this ComfyUI has no MiniMax H3 to extend ({error})"
    if not isinstance(diffusion, model_cls.__bases__[0]):
        return False, "this model is not a MiniMax H3"
    for index in range(len(blocks)):
        _to_causal_block(blocks[index], block_cls, attention_cls)
    diffusion.__class__ = model_cls
    diffusion._funpack_causal = True
    return True, f"chunk-causal DiT: {len(blocks)} blocks re-classed in place"
