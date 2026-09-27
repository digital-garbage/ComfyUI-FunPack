"""The shadow pass itself: H3's block forward, run twice, with a NAG push between.

Ported from the community "H3 Shadow Negative" pack (MIT), Basic scope, via v4's
h3_shadow_negative.py. What changed from v4, and why:

* **The prompt length comes from the model call.** v4 read it from
  `transformer_options["c_crossattn"]`, which no ComfyUI version sets (checked
  locally and on upstream master, 2026-09-27) -- so every block fell straight
  back to the stock forward and the feature never ran. Here a DIFFUSION_MODEL
  wrapper records `context.shape[1]` for each call.
* **Progress through the schedule** is the exact step index (core's
  `current_step`), not a guess from a `timestep` key that was never present.
* **The shadow state resets per model call** in that same wrapper, instead of
  assuming block 0 is always hooked and always first.
"""

import torch


def nag_blend(pos, neg, scale, tau, alpha, channel_dim=1, eps=1e-6):
    """Push `pos` away from `neg`, norm-capped at `tau` x pos's own norm.

    scale 1 is a no-op; alpha blends guided (1) against untouched (0).
    """
    s, a = float(scale), float(alpha)
    if s == 1.0 or a == 0.0:
        return pos
    guided = pos * s - neg * (s - 1.0)
    pos_norm = torch.linalg.vector_norm(pos.float(), ord=2, dim=channel_dim, keepdim=True)
    guided_norm = torch.linalg.vector_norm(guided.float(), ord=2, dim=channel_dim, keepdim=True)
    ratio = guided_norm / (pos_norm + eps)
    cap = torch.clamp(float(tau) / ratio.clamp_min(eps), max=1.0).to(guided.dtype)
    return guided * cap * a + pos * (1.0 - a)


class State:
    """One run's worth: the negative text, and the shadow stream as it goes deeper."""

    def __init__(self, dm, video_scale, audio_scale, tau, alpha, start, end):
        self.dm = dm
        self.video_scale, self.audio_scale = float(video_scale), float(audio_scale)
        self.tau, self.alpha = float(tau), float(alpha)
        self.start, self.end = float(start), float(end)
        self.negative = None       # [L, text_dim] or [L, hidden]; set per sampling run
        self.text_len = None       # the POSITIVE context length; set per model call
        self.neg_h = None          # shadow stream; reset per model call


def _tag_is(row, tag):
    """A mod row is an int, or -- once an i2v pin or mask gives rows their own
    strength -- a per-token LongTensor. v4 called int() on the tensor, which
    raised, so it never ran on an i2v scene."""
    if torch.is_tensor(row):
        return bool(((row % 3) == tag).all())
    return int(row) % 3 == tag


def _mod_one(h, shift, scale, row):
    return h.mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))


def _gate_slice(x, residual, gate, a, b, row):
    x[a:b].addcmul_(residual[a:b], gate[row].to(x.dtype))


def _prepare_negative(state, x, to):
    neg = state.negative
    if not torch.is_tensor(neg):
        return None
    if neg.ndim == 3:
        neg = neg[0]
    if neg.ndim != 2 or neg.shape[0] == 0:
        return None
    neg = neg.to(device=x.device, dtype=x.dtype)
    if neg.shape[-1] != state.dm.hidden_size:
        neg = state.dm.token_refiner(state.dm.condition_proj(neg), transformer_options=to)
    return neg


def _presentation_plan(mod_segments, text_len):
    """(runs inside the text span, the user-prompt run) or (None, None).

    H3 tags prompt text 1 and presentation pads 0/2; the tokenizer puts the
    user's prompt after any media labels, so the LAST tag-1 run is the one the
    negative replaces. Everything else in the span stays the positive's.
    """
    runs = []
    for seg in mod_segments:
        a, b, _row = seg
        if a >= text_len:
            break
        if b > text_len:
            return None, None
        runs.append(seg)
    text_runs = [r for r in runs if _tag_is(r[2], 1)]
    if not runs or not text_runs:
        return None, None
    return runs, text_runs[-1]


def _negative_rope(state, rope, start, n, x):
    """RoPE for the replacement text at the prompt's own origin."""
    stop = start + n
    if stop <= state.text_len:
        return rope[:, start:stop]
    from comfy.ldm.minimax.model import rope_rotation_table
    pos = torch.zeros(n, 3, dtype=torch.float64)
    pos[:, 0] = torch.arange(start, stop, dtype=torch.float64)
    return rope_rotation_table(state.dm.rope_freqs(pos, x.device), x.dtype)


def forward(state, block, args):
    """The block, with the shadow push. None means "run the stock block"; the
    caller does that, so a layout this cannot prove costs nothing."""
    x, t_emb = args["img"], args["t_emb"]
    segs, rope, to = args["mod_segments"], args["rope_freqs"], args["transformer_options"]
    text_len = state.text_len
    if text_len is None or len(segs) < 3:
        return None, "the prompt length was not seen for this call"

    audio_a, audio_b, audio_row = segs[-2]
    video_a, video_b, video_row = segs[-1]
    if (not _tag_is(audio_row, 2) or not _tag_is(video_row, 0)
            or audio_b != video_a or video_b != x.shape[0]):
        return None, "could not prove where the target audio and video rows are"
    runs, user_run = _presentation_plan(segs, text_len)
    if runs is None:
        return None, "could not find the prompt inside the text span"

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h_pos = block.norm1(x)
    for a, b, row in segs:
        _mod_one(h_pos[a:b], shift_msa, scale_msa, row)
    attn_pos = block.attn(h_pos, rope_freqs=rope, transformer_options=to)

    neg_h = state.neg_h if state.neg_h is not None else _prepare_negative(state, x, to)
    if neg_h is None:
        return None, "there is no negative prompt to push away from"

    # The shadow sequence: the same presentation span with the user's prompt
    # swapped for the negative, then the positive's own audio/video rows.
    text_row = user_run[2]
    neg_norm = _mod_one(block.norm1(neg_h), shift_msa, scale_msa, text_row)
    neg_rope = _negative_rope(state, rope, user_run[0], int(neg_norm.shape[0]), h_pos)
    neg_norm = neg_norm[:neg_rope.shape[1]]
    h_parts, r_parts, cursor, neg_a = [], [], 0, 0
    for a, b, row in runs:
        if (a, b) == user_run[:2]:
            neg_a = cursor
            h_parts.append(neg_norm)
            r_parts.append(neg_rope)
            cursor += neg_norm.shape[0]
        else:
            h_parts.append(h_pos[a:b])
            r_parts.append(rope[:, a:b])
            cursor += b - a
    neg_b, suffix = neg_a + neg_norm.shape[0], cursor
    h_parts.append(h_pos[text_len:video_b])
    r_parts.append(rope[:, text_len:video_b])
    attn_neg = block.attn(torch.cat(h_parts, 0), rope_freqs=torch.cat(r_parts, 1),
                          transformer_options=to)

    def shadow(a, b):
        return attn_neg[suffix + a - text_len:suffix + b - text_len]

    if state.video_scale != 1.0:
        attn_pos[video_a:video_b] = nag_blend(attn_pos[video_a:video_b], shadow(video_a, video_b),
                                              state.video_scale, state.tau, state.alpha)
    if state.audio_scale != 1.0:
        attn_pos[audio_a:audio_b] = nag_blend(attn_pos[audio_a:audio_b], shadow(audio_a, audio_b),
                                              state.audio_scale, state.tau, state.alpha)

    out = x.clone()
    for a, b, row in segs:
        _gate_slice(out, attn_pos, gate_msa, a, b, row)
    h2 = block.norm2(out)
    for a, b, row in segs:
        _mod_one(h2[a:b], shift_mlp, scale_mlp, row)
    mlp_pos = block.mlp(h2)
    for a, b, row in segs:
        _gate_slice(out, mlp_pos, gate_mlp, a, b, row)

    # Advance the shadow stream through this block too, so the next block's
    # push is against the negative text at the same depth.
    neg_x = neg_h[:neg_b - neg_a].clone()
    neg_x.addcmul_(attn_neg[neg_a:neg_b], gate_msa[text_row].to(neg_x.dtype))
    neg2 = _mod_one(block.norm2(neg_x), shift_mlp, scale_mlp, text_row)
    neg_x.addcmul_(block.mlp(neg2), gate_mlp[text_row].to(neg_x.dtype))
    state.neg_h = neg_x
    return {"img": out}, None
