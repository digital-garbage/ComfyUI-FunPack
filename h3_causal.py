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
or the model is being told the clip is a different shape than it is. So ONE full-clip
`PackedLayout` is built by ComfyUI's own `extra_conds` and every chunk is a SELECTION of its
rows. An upstream change to the layout is inherited rather than silently disagreed with, and
the keyframe pins and reference blocks keep the rows upstream packed for them — which is what
makes i2v and r2v work on this lane instead of being refused by it.

WHAT IS OURS AND WHAT IS THE MODEL'S
------------------------------------
The chunk cut, the cache, the schedule and the step rule are all here: any scheduler, any step
count, and a step rule that can be set to plain flow euler so a chunked run can be compared
against a dense one without two variables moving. Nothing in this file needs a third-party
package installed.

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


class ChunkPlan:
    """Which packed rows each chunk owns, cut out of ONE full-clip PackedLayout.

    The layout is upstream's own, built for the WHOLE clip. Every chunk is then a selection of
    its rows, so a chunk's RoPE angles, modality tags and timestep classes are exactly the ones
    the dense forward would have given those same rows. Re-deriving the geometry per chunk
    would be a second source of truth for the clip's shape — and the first symptom of it
    drifting is a clip that looks subtly wrong rather than an exception.

    Two kinds of selection:

    `prefix` is everything before the target streams — the text, the keyframe cond rows and
    the reference blocks. It is cached ONCE, as cache chunk 0, and every media chunk reads it.
    That is why i2v and r2v survive here: the conditioning rows keep the layout upstream built
    for them instead of being re-packed into a chunk that has no place for them.

    `chunk(i)` is media chunk i: its audio rows followed by its video rows, in the packing
    order the target segments already use. Media chunk i is CACHE chunk i + 1, because the
    prefix took index 0.
    """

    def __init__(self, layout, bounds, audio_t):
        self.layout = layout
        self.bounds = list(bounds)
        self.audio_t = int(audio_t)
        segments = list(layout.segments)
        audio_seg = next((s for s in segments if s[2] == "audio"), None)
        video_seg = next((s for s in segments if s[2] == "video"), None)
        if audio_seg is None or video_seg is None:
            raise CacheError("this layout has no target audio/video segment to chunk")
        latent_t = self.bounds[-1][1] if self.bounds else 0
        if latent_t <= 0:
            raise CacheError("a clip with no video latents has nothing to chunk")
        # Rows per latent frame, ASKED OF THE LAYOUT rather than re-derived from the patch
        # size. The layout already answered this question when it packed the video segment,
        # and a second answer is a second thing that can drift.
        self.frame_rows = (video_seg[1] - video_seg[0]) // latent_t
        self._audio_base, self._video_base = audio_seg[0], video_seg[0]
        # Every chunk must own rows of BOTH streams: the final layer reads one contiguous
        # span of each, so a chunk with no audio is a StopIteration fifty blocks deep rather
        # than an answer. It only happens when the two streams disagree about the clip's
        # length, so say that instead.
        empty = [i for i, (vs, ve, a, b) in enumerate(self.bounds) if ve <= vs or b <= a]
        if empty:
            raise CacheError(
                f"the soundtrack does not cover the picture — {self.audio_t} audio latents "
                f"against {latent_t} video latents leaves chunk {empty[0]} silent. H3 runs "
                f"audio at 40/s against a fixed 24 fps; this clip is not on that clock.")
        self.prefix_rows = torch.arange(0, audio_seg[0], dtype=torch.long)
        self.prefix_runs = [(a, b, kind) for a, b, kind in segments if b <= audio_seg[0]]

    @property
    def n_chunks(self):
        return len(self.bounds)

    @staticmethod
    def cache_index(index):
        """Media chunk i lives at cache index i + 1; the prefix holds 0."""
        return int(index) + 1

    def chunk(self, index):
        """(row indices, local runs) for one media chunk."""
        v_start, v_stop, a_start, a_stop = self.bounds[int(index)]
        audio = [self._audio_base + r for r in chunk_rows(self.audio_t, a_start, a_stop)]
        video = range(self._video_base + v_start * self.frame_rows,
                      self._video_base + v_stop * self.frame_rows)
        rows = torch.tensor(list(audio) + list(video), dtype=torch.long)
        n_audio = len(audio)
        runs = [(0, n_audio, "audio"), (n_audio, rows.shape[0], "video")]
        return rows, [r for r in runs if r[1] > r[0]]


def build_plan(layout, latent_t, audio_t):
    """A ChunkPlan for a clip, with the chunk cuts derived from the shared clock."""
    return ChunkPlan(layout, chunk_bounds(latent_t, audio_t), audio_t)


#: how a chunk's next-sigma is reached from its prediction
STEP_RULES = ("consistency", "euler", "euler_ancestral")


def step(rule, x_t, prediction, sigma, sigma_next, noise, eta=1.0):
    """Advance one step, from a model prediction to the next sigma.

    `prediction` is x0. H3's head emits data-ward velocity and the caller converts with
    ``x0 = x_t + sigma * v`` before getting here — no negation; that is the model's own
    convention, and getting it wrong inverts the whole clip rather than degrading it.

    consistency
        RAVEN's transition: jump to x0 and re-noise to the next sigma with FRESH noise, never
        the noise already in x_t. This is what the RAVEN LoRA is distilled for.
    euler
        The ordinary flow step. Deterministic, and the same rule FunPack's other H3 sampling
        uses — so a chunked run can be compared against an unchunked one without the step rule
        being a second variable.
    euler_ancestral
        Euler with `eta` of the step's noise put back, in the RECTIFIED-FLOW form. The VP
        formula is wrong on a flow model and was a real bug here before.
    """
    sigma, sigma_next = float(sigma), float(sigma_next)
    if rule == "consistency":
        return (1.0 - sigma_next) * prediction + sigma_next * noise
    if rule not in ("euler", "euler_ancestral"):
        raise ValueError(f"unknown step rule {rule!r}; expected one of {STEP_RULES}")
    if sigma <= 0:
        return prediction
    derivative = (x_t - prediction) / sigma          # dx/dsigma on the flow ODE
    if rule == "euler" or sigma_next <= 0 or eta <= 0:
        return x_t + (sigma_next - sigma) * derivative
    # rectified-flow ancestral: shorten the step, then put back what was taken out
    down_ratio = 1.0 + (sigma_next / sigma - 1.0) * float(eta)
    sigma_down = sigma_next * down_ratio
    stepped = x_t + (sigma_down - sigma) * derivative
    renoise = max(0.0, sigma_next ** 2 - sigma_down ** 2) ** 0.5
    return stepped + noise * renoise


def audio_sigma_for(sigma, shift_video, shift_audio):
    """The audio stream's own sigma for a video sigma, through the flow time-shift.

    H3 runs the two streams on DIFFERENT shifted grids — 12 for video, 3 for audio — and the
    sampler only ever sees one array. Reusing the video sigma for the audio rows tells the
    model the soundtrack is at a noise level it is not at, and the error grows with the step
    size, which is why it bites hardest on the few-step schedules this lane is built for.

    Same remap ``h3_audio_clock`` uses: undo the video shift to recover the unshifted time,
    then apply the audio shift to it.
    """
    sigma = float(sigma)
    if sigma <= 0.0 or float(shift_video) == float(shift_audio):
        return sigma
    unshifted = sigma / (float(shift_video) - (float(shift_video) - 1.0) * sigma)
    return (float(shift_audio) * unshifted
            / (1.0 + (float(shift_audio) - 1.0) * unshifted))


def causal_rollout(*, chunks, sigmas, forward, commit, draw_noise,
                   video_noise, audio_noise, step_rule="consistency", eta=1.0,
                   known_chunks=0, known_video=None, known_audio=None,
                   on_chunk=None, cancel=None):
    """Generate a clip chunk by chunk, each one attending to the cache of the ones before it.

    Everything the model does arrives as a callable, so the loop is testable without ComfyUI,
    without the RAVEN package and without weights — and so the same loop can be driven by a
    different backend later without being rewritten.

    forward(video_xt, audio_xt, index, sigma)
        -> (video_velocity, audio_velocity) for that chunk at that sigma. Cache read-only.
    commit(video_x0, audio_x0, index)
        writes the finished chunk into the KV cache as clean context. Called ONCE per chunk,
        after its last step, and for the pre-known chunks too — a chunk that is not committed
        is a chunk the rest of the clip cannot see.
    draw_noise(shape)
        fresh noise. Called in a fixed order so a seeded run reproduces exactly.

    `known_chunks` leading chunks are taken from `known_video`/`known_audio` instead of being
    sampled. That is how an i2v anchor survives: chunk 0 comes from FunPack's ORDINARY dense
    path — where pins, references and region locks all work — and is only then committed here,
    so the causal lane is never asked to model conditioning rows it has no layout for.

    `sigmas` is FunPack's own schedule, descending and ending at 0. Any scheduler, any step
    count; the RAVEN adapter's distilled 4 is a default, not a constraint.
    """
    if step_rule not in STEP_RULES:
        raise ValueError(f"unknown step rule {step_rule!r}; expected one of {STEP_RULES}")
    sigmas = [float(s) for s in sigmas]
    if len(sigmas) < 2:
        raise ValueError("a schedule needs at least one step (two sigmas)")
    known = int(known_chunks)
    if known and (known_video is None or known_audio is None):
        raise ValueError("known_chunks was given without the latents to take them from")

    video_out = video_noise.clone()
    audio_out = audio_noise.clone()

    for index, (v_start, v_stop, a_start, a_stop) in enumerate(chunks):
        if cancel is not None:
            cancel(index)
        if index < known:
            # Already generated elsewhere. It still has to be COMMITTED, or every later
            # chunk continues from a clip that appears to start after it.
            video_x0 = known_video[:, :, v_start:v_stop]
            audio_x0 = known_audio[..., a_start:a_stop]
            video_out[:, :, v_start:v_stop] = video_x0
            audio_out[..., a_start:a_stop] = audio_x0
            commit(video_x0, audio_x0, index)
            if on_chunk is not None:
                on_chunk(index, video_x0, audio_x0, True)
            continue

        video_xt = video_noise[:, :, v_start:v_stop].clone()
        audio_xt = audio_noise[..., a_start:a_stop].clone()
        for position in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[position], sigmas[position + 1]
            video_v, audio_v = forward(video_xt, audio_xt, index, sigma)
            # H3's head is data-ward velocity; x0 = x_t + sigma * v, no negation.
            video_x0 = video_xt + sigma * video_v
            audio_x0 = audio_xt + sigma * audio_v
            # Both draws happen every step even when they are multiplied by zero at the end:
            # skipping one would shift the noise stream for every later chunk and a seeded
            # run would stop reproducing.
            video_eps = draw_noise(video_xt.shape)
            audio_eps = draw_noise(audio_xt.shape)
            video_xt = step(step_rule, video_xt, video_x0, sigma, sigma_next, video_eps, eta)
            audio_xt = step(step_rule, audio_xt, audio_x0, sigma, sigma_next, audio_eps, eta)

        video_out[:, :, v_start:v_stop] = video_xt
        audio_out[..., a_start:a_stop] = audio_xt
        commit(video_xt, audio_xt, index)
        if on_chunk is not None:
            on_chunk(index, video_xt, audio_xt, False)

    return video_out, audio_out


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
    from comfy.ldm.minimax.model import (AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP, Attention,
                                         DiTBlock, MiniMaxH3Model, _mod_gate, _mod_scale_shift,
                                         optimized_attention, pack_audio, patchify_video,
                                         rope_rotation_table, time_shift_slope, unpack_audio,
                                         unpatchify_video)

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

        # ── the two entry points the rollout drives ─────────────────────────────────

        def _modulation(self, kinds, t_v, t_a, payload):
            """seg_t / t_row / t_emb for the kinds actually present in this sequence.

            Upstream's own analytic table: text and video ride the video's time, audio rides
            the audio's, and the condition rows pin near 1 because that is the noise level
            they were augmented to. Only the kinds present get a timestep ROW, so a chunk with
            no cond rows does not carry an unused one into the AdaLN table.
            """
            vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
            aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
            seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                     "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                     "ref_audio": max(t_a, aud_aug)}
            unique_t = sorted({seg_t[k] for k in kinds})
            return seg_t, {t: i for i, t in enumerate(unique_t)}, unique_t

        def _t_emb(self, unique_t, device, dtype):
            t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
            if self.use_adaln_curves:
                table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
                pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
                i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
                return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
            return self.time_embedder(t_vals).to(dtype)

        def _mod_segments(self, runs, seg_t, t_row, text_tags):
            """Local (start, stop, adaln row) runs, splitting the text span by its tag runs.

            The text span mixes modality tags — a reference image's vision pads sit inside the
            text and carry the VIDEO tag — so it cannot be treated as one uniform run without
            modulating the picture rows as if they were words.
            """
            seg_tag = {"text": TEXT_TAG, "video": VIDEO_TAG, "audio": AUDIO_TAG,
                       "cond": VIDEO_TAG, "ref_img": VIDEO_TAG, "ref_audio": AUDIO_TAG}
            out = []
            for a, b, kind in runs:
                base = t_row[seg_t[kind]] * 3
                if kind == "text" and text_tags is not None:
                    tags = text_tags.view(-1).tolist()
                    run_start = 0
                    for i in range(1, b - a + 1):
                        if i == b - a or tags[i] != tags[run_start]:
                            out.append((a + run_start, a + i, base + int(tags[run_start])))
                            run_start = i
                else:
                    out.append((a, b, base + seg_tag[kind]))
            return out

        def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options,
                        cache, chunk_index, update_cache):
            for layer, block in enumerate(self.blocks):
                h = block(h, t_emb, mod_segments, rope_freqs,
                          transformer_options=transformer_options, cache=cache,
                          layer_idx=layer, chunk_index=chunk_index,
                          update_cache=update_cache)
            return h

        def _text_states(self, context, transformer_options):
            states = context[0]
            if states.shape[-1] == self.hidden_size:
                return states
            return self.token_refiner(self.condition_proj(states),
                                      transformer_options=transformer_options)

        def prefill_text(self, context, plan, cache, *, video_sigma=0.0, audio_sigma=0.0,
                         transformer_options={}, minimax_payload=None):
            """Run the prompt and the conditioning rows once, and cache them as chunk 0.

            The prefix is the text, the keyframe cond rows and the reference blocks — the part
            of the sequence that does not change from chunk to chunk. Caching it once is not
            only cheaper: every later chunk assumes the text keys it attends are the ones the
            model saw, and recomputing them per chunk would quietly make that false.

            Cached at sigma 0 by default, so the prefix is CLEAN context. The cond rows pin
            near 1 in upstream's own table regardless, and text is never noised, so this is the
            state those rows would be in at the end of any schedule.
            """
            payload = minimax_payload or {}
            rows = plan.prefix_rows
            if rows.numel() == 0:
                raise CacheError("this layout has no prefix rows to prefill")
            device = comfy.model_management.get_torch_device()
            if hasattr(context, "device"):
                device = context.device
            dtype = context.dtype
            t_v = float(1.0 - max(float(video_sigma), 1e-6))
            t_a = float(1.0 - max(float(audio_sigma), 1e-6))
            runs = plan.prefix_runs
            seg_t, t_row, unique_t = self._modulation({k for _, _, k in runs}, t_v, t_a, payload)
            text_states = self._text_states(context, transformer_options)
            cond_video = self._cond_video_rows(payload, device)
            cond_audio = self._cond_audio_rows(payload, device)

            h = torch.empty(int(rows.shape[0]), self.hidden_size, dtype=dtype, device=device)
            voff = aoff = 0
            for a, b, kind in runs:
                n = b - a
                if kind == "text":
                    h[a:b] = text_states
                elif kind in ("cond", "ref_img"):
                    if cond_video is None:
                        raise CacheError(f"the layout has {kind} rows but the payload carries "
                                         f"no cond_video_latents to fill them")
                    h[a:b] = self.video_patch_proj(cond_video[voff:voff + n]).to(dtype)
                    voff += n
                else:                                     # ref_audio
                    if cond_audio is None:
                        raise CacheError("the layout has ref_audio rows but the payload "
                                         "carries no cond_audio_latents to fill them")
                    h[a:b] = self.audio_patch_proj(cond_audio[aoff:aoff + n]).to(dtype)
                    aoff += n

            rope_freqs = rope_rotation_table(
                self.rope_freqs(plan.layout.position_ids[rows], device), dtype)
            self._run_blocks(h, self._t_emb(unique_t, device, dtype),
                             self._mod_segments(runs, seg_t, t_row,
                                                payload.get("text_token_tags")),
                             rope_freqs, transformer_options, cache, 0, True)
            cache.finish_chunk(0)

        def forward_chunk(self, video_latent, audio_latent, plan, index, cache, *,
                          video_sigma, audio_sigma, context=None, update_cache=False,
                          transformer_options={}, minimax_payload=None,
                          sigma_shift_video=None, sigma_shift_audio=None):
            """One media chunk's forward against the cache. Returns (video_v, audio_v).

            DATA-WARD velocity, which is what `x0 = x_t + sigma * v` wants — upstream's
            `_forward` negates on the way out because ComfyUI's flow sampler expects the other
            sign, and following it here would invert the whole clip rather than degrade it.

            The audio velocity carries the same `time_shift_slope` factor upstream applies, so
            the caller can integrate BOTH streams against the video sigma and still be on the
            audio's own shifted grid.
            """
            payload = minimax_payload or {}
            rows, runs = plan.chunk(index)
            device = video_latent.device
            dtype = context.dtype if context is not None else video_latent.dtype
            latent_t = video_latent.shape[2]
            lat_h, lat_w = video_latent.shape[3], video_latent.shape[4]
            sigma_v = max(float(video_sigma), 1e-6)
            t_v = float(1.0 - sigma_v)
            t_a = float(1.0 - max(float(audio_sigma), 1e-6))
            seg_t, t_row, unique_t = self._modulation({k for _, _, k in runs}, t_v, t_a, payload)

            video_embed = self.video_patch_proj(
                patchify_video(video_latent.to(torch.float32), self.patch_size)).to(dtype)
            audio_embed = self.audio_patch_proj(
                pack_audio(audio_latent.to(torch.float32))).to(dtype)
            h = torch.empty(int(rows.shape[0]), self.hidden_size, dtype=dtype, device=device)
            for a, b, kind in runs:
                h[a:b] = video_embed if kind == "video" else audio_embed

            rope_freqs = rope_rotation_table(
                self.rope_freqs(plan.layout.position_ids[rows], device), dtype)
            t_emb = self._t_emb(unique_t, device, dtype)
            h = self._run_blocks(h, t_emb,
                                 self._mod_segments(runs, seg_t, t_row, None),
                                 rope_freqs, transformer_options, cache,
                                 plan.cache_index(index), update_cache)

            video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in runs if k == "video")
            audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in runs if k == "audio")
            v, a = self.final_layer(h, t_emb, video_seg, audio_seg)
            video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2,
                                         self.latents_dim, self.patch_size)
            audio_out = unpack_audio(a)
            shift_v = float(self.sigma_shift_video if sigma_shift_video is None
                            else sigma_shift_video)
            shift_a = float(self.sigma_shift_audio if sigma_shift_audio is None
                            else sigma_shift_audio)
            slope = time_shift_slope(torch.tensor(sigma_v), shift_v, shift_a)
            return (video_out.to(video_latent.dtype),
                    (slope.to(audio_out.dtype) * audio_out).to(audio_latent.dtype))

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


# ── driving a whole clip ────────────────────────────────────────────────────────────


def _unbind_av(samples):
    """(video, audio) out of the nested container H3 latents travel in."""
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
        if len(parts) == 2:
            return parts[0], parts[1]
    raise CacheError("this latent is not an H3 video+audio pair")


def build_session(model, positive, latent, *, sink=2, window=2, device=None, offload=True):
    """Assemble everything one causal run needs, or explain why it cannot be assembled.

    Returns (session, reason). `session` is None when the run cannot be built, and `reason` is
    written for the person who has to fix it rather than for a log parser.

    The PAYLOAD is built by ComfyUI's own `extra_conds`, not by a parser of our own. That is
    deliberate: the keyframe rows, the reference blocks, the condition noise augmentation and
    the packed layout are then bit-for-bit the ones the dense path would have used, so an
    upstream change to any of them is inherited instead of silently disagreed with.
    """
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None:
        return None, "this model has no diffusion model to run."
    if not hasattr(diffusion, "forward_chunk"):
        return None, ("the loaded model is a stock bidirectional H3. Switch on `chunk_causal` "
                      "on FunPack Diffusion Model Loader, and load a LoRA trained for the "
                      "chunked pattern with FunPack LoRA Loader — it applies on its own, no "
                      "stack node needed.")
    blocks = getattr(diffusion, "blocks", None)
    if not blocks:
        return None, "the loaded model exposes no DiT blocks to cache."
    try:
        video, audio = _unbind_av(latent["samples"])
    except Exception as error:                                        # noqa: BLE001
        return None, f"this run is not one the causal lane can take: {error}"
    if video.shape[0] != 1:
        return None, "MiniMax H3 samples one clip at a time; this batch is larger."

    if device is None:
        device = getattr(model, "load_device", None) or video.device
    device = torch.device(device)
    try:
        base = model.model
        kwargs = dict(positive[0][1])
        kwargs["cross_attn"] = positive[0][0].to(device)
        kwargs["device"] = device
        kwargs["latent_shapes"] = [tuple(video.shape), tuple(audio.shape)]
        kwargs.setdefault("seed", 0)
        conds = base.extra_conds(**kwargs)
        payload = conds["minimax_payload"].cond
        context = conds["c_crossattn"].cond
    except Exception as error:                                        # noqa: BLE001
        return None, f"the conditioning could not be prepared for the causal lane: {error}"

    layout = payload.get("layout")
    if layout is None:
        return None, ("this run produced no packed layout — the causal lane needs the same "
                      "one the dense path builds.")
    try:
        plan = build_plan(layout, int(video.shape[2]), int(audio.shape[-1]))
    except CacheError as error:
        return None, str(error)
    if plan.n_chunks < 2:
        return None, (f"this clip is {plan.n_chunks} chunk long — there is nothing for a "
                      f"chunk cache to remember.")

    cache = ChunkKVCache(len(blocks), sink=int(sink),
                         window=None if window is None else int(window),
                         device=device, offload=bool(offload))
    return {
        "model": diffusion,
        "patcher": model,
        "plan": plan,
        "cache": cache,
        "context": context,
        "payload": payload,
        "device": device,
        "video_shape": tuple(video.shape),
        "audio_shape": tuple(audio.shape),
        "shift_video": float(getattr(diffusion, "sigma_shift_video", 12.0)),
        "shift_audio": float(getattr(diffusion, "sigma_shift_audio", 3.0)),
    }, ""


def run_session(session, *, sigmas, step_rule="consistency", eta=1.0, seed=0,
                known_chunks=0, known_video=None, known_audio=None,
                on_chunk=None, cancel=None, transformer_options=None):
    """Drive one causal rollout over an assembled session. Returns (video, audio)."""
    model, cache, plan = session["model"], session["cache"], session["plan"]
    device, payload = session["device"], session["payload"]
    options = dict(transformer_options or {})

    # LOAD FIRST. The ordinary path reaches the model through comfy.sample.sample_custom,
    # whose prepare_sampling calls load_models_gpu — and load_models_gpu is also what applies
    # a patcher's OBJECT PATCHES (partially_load -> patch_model). This lane calls the DiT
    # directly, so without this the weights may still be offloaded AND every object patch
    # FunPack installed (the AdaLN modality gains, the token-refiner edit) would silently not
    # be there. Once, for the whole rollout: the loop must not fight Comfy's offload decision
    # chunk by chunk.
    patcher = session.get("patcher")
    if patcher is not None:
        try:
            import comfy.model_management
            comfy.model_management.load_models_gpu([patcher], memory_required=0,
                                                   force_full_load=False)
        except Exception as error:                                    # noqa: BLE001
            print(f"[FunPack H3] causal lane could not pre-load the model ({error}); the "
                  f"rollout continues, but object patches may not be installed.")

    cache.clear()
    with torch.no_grad():
        model.prefill_text(session["context"], plan, cache,
                           transformer_options=options, minimax_payload=payload)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def draw_noise(shape):
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)

    video_noise = draw_noise(session["video_shape"])
    audio_noise = draw_noise(session["audio_shape"])
    shift_v, shift_a = session["shift_video"], session["shift_audio"]

    def forward(video_xt, audio_xt, index, sigma):
        # The two streams do NOT share a sigma. H3 denoises audio on its own shifted grid
        # (3 against the video's 12), and handing the audio rows the video's sigma is exactly
        # the error h3_audio_clock exists to correct — the longer the step, the worse it is.
        with torch.no_grad():
            return model.forward_chunk(
                video_xt, audio_xt, plan, index, cache,
                video_sigma=float(sigma),
                audio_sigma=float(audio_sigma_for(sigma, shift_v, shift_a)),
                context=session["context"], update_cache=False,
                transformer_options=options, minimax_payload=payload,
                sigma_shift_video=shift_v, sigma_shift_audio=shift_a)

    def commit(video_x0, audio_x0, index):
        # One extra forward per chunk, at sigma 0, purely to write CLEAN K/V. Caching the K/V
        # of the last denoising step instead would hand every later chunk a context that still
        # carries that step's noise.
        with torch.no_grad():
            model.forward_chunk(
                video_x0, audio_x0, plan, index, cache,
                video_sigma=0.0, audio_sigma=0.0, context=session["context"],
                update_cache=True, transformer_options=options, minimax_payload=payload,
                sigma_shift_video=shift_v, sigma_shift_audio=shift_a)
        cache.finish_chunk(plan.cache_index(index))

    return causal_rollout(
        chunks=plan.bounds, sigmas=sigmas, forward=forward, commit=commit,
        draw_noise=draw_noise, video_noise=video_noise, audio_noise=audio_noise,
        step_rule=step_rule, eta=eta, known_chunks=known_chunks,
        known_video=known_video, known_audio=known_audio, on_chunk=on_chunk, cancel=cancel)
