"""H3 shadow negative: give the negative prompt a job at CFG 1, via attention repulsion.

MiniMax H3 runs at CFG 1.0 always (see project_h3_sampling memory), so the negative
CONDITIONING FunPack already threads through the sampler is otherwise computed and
discarded -- comfy's CFG guider skips the uncond branch entirely at cond_scale==1.0.
[[negative_erase]] already gives that text a job by projecting its direction out of the
POSITIVE conditioning once, before sampling starts. This is a different, heavier lever:
a second "shadow" copy of the negative text is run through the SAME DiT blocks, block by
block, alongside the real (positive) stream, and the real stream's attention output is
pushed away from the shadow's, NAG-style (normalized, norm-capped so it cannot run away).

Ported from the "H3 Shadow Negative" ComfyUI-H3NegativeLab pack (MIT), Basic-node scope
only: one shared negative drives both video and audio, T2V/FL2VA/REF2VA with best-effort
reference preservation. NOT ported: the "Advanced exact REF2VA" mode (needs a
minimax_payload/layout object this codebase's comfy fork was never confirmed to expose
the same way) and its weighted-text-encode node (FunPack already has (phrase:weight)
markup for H3 in h3_token_weights.py).

Does NOT compose with FunPack's other per-block dit patches (REINS/Q-steer/av_decouple/
block-repeat) -- it replaces each block's forward outright, same as upstream. See
_install_h3_shadow_negative in samplers.py for the "last installed wins, say so loudly"
policy this shares with the rest of this codebase's dit-patch stack.
"""

import torch


def _as_float(x, default=0.0):
    try:
        if isinstance(x, torch.Tensor):
            return float(x.flatten()[0])
        return float(x)
    except Exception:
        return float(default)


def sigma_progress(timestep, sigmas):
    """0 at the first denoise step, 1 at the last real one, or None if unresolvable."""
    if sigmas is None:
        return None
    try:
        cur = _as_float(timestep)
        vals = sigmas.detach().flatten().tolist() if isinstance(sigmas, torch.Tensor) else list(sigmas)
        if len(vals) < 2:
            return None
        steps = len(vals) - 1
        idx = min(range(steps), key=lambda i: abs(float(vals[i]) - cur))
        return idx / float(max(1, steps - 1)) if steps > 1 else 0.0
    except Exception:
        return None


def nag_blend(pos, neg, scale, tau, alpha, channel_dim=1, eps=1e-6):
    """Normalized extrapolation of `pos` away from `neg` (NAG-form).

    scale=1.0 is a no-op (returns pos unchanged); >1 pushes away from neg. The guided
    result's norm is capped at `tau` times the positive's own norm, per channel-axis
    position, so a token that would blow up under extrapolation is clamped back rather
    than corrupting the stream. `alpha` blends between the positive-only and guided
    result (0 = ignore this call's math entirely, 1 = fully guided).
    """
    s = float(scale)
    a = float(alpha)
    if s == 1.0 or a == 0.0:
        return pos
    guided = pos * s - neg * (s - 1.0)
    pos_norm = torch.linalg.vector_norm(pos.float(), ord=2, dim=channel_dim, keepdim=True)
    guided_norm = torch.linalg.vector_norm(guided.float(), ord=2, dim=channel_dim, keepdim=True)
    ratio = guided_norm / (pos_norm + eps)
    cap = torch.clamp(float(tau) / ratio.clamp_min(eps), max=1.0).to(guided.dtype)
    guided = guided * cap
    return guided * a + pos * (1.0 - a)


class ShadowState:
    """Shared, mutable state for one _sample_chunk call's worth of shadow-negative blocks.

    `neg_h` is the shadow branch's own running hidden state -- it has to go through the
    SAME sequence of blocks the positive stream does (each block's own attn+mlp, in
    order) to mean anything as "what the model would produce for the negative text at
    this depth". Reset to None at block 0 of every forward pass (block 0 always runs
    first and only once per model() call -- see _install_h3_av_decouple's own use of the
    same fact), rebuilt fresh from `negative_context` on first use afterwards.
    """
    __slots__ = ("dm", "negative_context", "video_scale", "audio_scale", "tau", "alpha",
                 "start_percent", "end_percent", "neg_h", "pos_text_len",
                 "warned_layout", "warned_fallback", "hard_disabled")

    def __init__(self, dm, negative_context, video_scale, audio_scale, tau, alpha,
                start_percent, end_percent):
        self.dm = dm
        self.negative_context = negative_context
        self.video_scale = float(video_scale)
        self.audio_scale = float(audio_scale)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.neg_h = None
        self.pos_text_len = None
        self.warned_layout = False
        self.warned_fallback = False
        self.hard_disabled = False


def _prepare_negative(state, x, transformer_options):
    neg = state.negative_context
    if not isinstance(neg, torch.Tensor):
        return None
    if neg.ndim == 3:
        neg = neg[0]
    if neg.ndim != 2 or neg.shape[0] == 0:
        return None
    neg = neg.to(device=x.device, dtype=x.dtype)
    if neg.shape[-1] != state.dm.hidden_size:
        neg = state.dm.condition_proj(neg)
        neg = state.dm.token_refiner(neg, transformer_options=transformer_options)
    return neg


def _mod_one(h, shift, scale, row):
    return h.mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))


def _gate_slice(x, residual, gate, a, b, row):
    x[a:b].addcmul_(residual[a:b], gate[row].to(x.dtype))


def _presentation_plan(mod_segments, pos_text_len):
    """(runs, user_prompt_run) for H3's Qwen presentation span, or (None, None).

    MiniMax H3 tags text as modality 1, visual presentation pads as 0/2. The tokenizer
    appends the user's prompt after any picture/video/audio labels, so the final tag-1
    run inside the context span is the one this replaces with the negative text; earlier
    labels and all visual presentation rows stay identical to the positive.
    """
    runs = []
    for seg in mod_segments:
        a, b, row = seg
        if a >= pos_text_len:
            break
        if b > pos_text_len:
            return None, None
        runs.append(seg)
    text_runs = [r for r in runs if (r[2] % 3) == 1]
    if not runs or not text_runs:
        return None, None
    return runs, text_runs[-1]


def _negative_text_rope(state, rope, start_pos, neg_len, x):
    """RoPE for variable-length replacement text at the original prompt's own origin."""
    stop = start_pos + neg_len
    if stop <= state.pos_text_len:
        return rope[:, start_pos:stop]
    try:
        from comfy.ldm.minimax.model import rope_rotation_table
        pos = torch.zeros(neg_len, 3, dtype=torch.float64)
        pos[:, 0] = torch.arange(start_pos, stop, dtype=torch.float64)
        freqs = state.dm.rope_freqs(pos, x.device)
        return rope_rotation_table(freqs, x.dtype)
    except Exception:
        if not state.warned_layout:
            state.warned_layout = True
            print("[FunPackSceneChain] H3 shadow negative: negative text is longer than "
                  "the positive prompt span and dynamic RoPE synthesis is unavailable on "
                  "this core -- truncating the negative to fit.")
        keep = max(1, state.pos_text_len - start_pos)
        return rope[:, start_pos:start_pos + keep]


def _build_shadow_attention_input(state, block, neg_h, h_pos, rope, presentation_runs,
                                  user_run, suffix_start, suffix_end,
                                  shift_msa, scale_msa, transformer_options):
    text_row = user_run[2]
    neg_norm = block.norm1(neg_h)
    _mod_one(neg_norm, shift_msa, scale_msa, text_row)

    neg_rope = _negative_text_rope(state, rope, user_run[0], int(neg_norm.shape[0]), h_pos)
    if neg_rope.shape[1] != neg_norm.shape[0]:
        neg_norm = neg_norm[:neg_rope.shape[1]]

    h_parts, r_parts, cursor = [], [], 0
    neg_a = neg_b = None
    for a, b, row in presentation_runs:
        if (a, b, row) == user_run:
            neg_a, neg_b = cursor, cursor + neg_norm.shape[0]
            h_parts.append(neg_norm)
            r_parts.append(neg_rope)
            cursor = neg_b
        else:
            h_parts.append(h_pos[a:b])
            r_parts.append(rope[:, a:b])
            cursor += b - a

    suffix_alt_start = cursor
    h_parts.append(h_pos[suffix_start:suffix_end])
    r_parts.append(rope[:, suffix_start:suffix_end])
    h_alt = torch.cat(h_parts, dim=0)
    rope_alt = torch.cat(r_parts, dim=1)
    return h_alt, rope_alt, neg_a, neg_b, suffix_alt_start, text_row


def make_block_hook(block, state, index):
    """One H3 double_block replacement for the shadow-negative branch. `index` is this
    block's position in dm.blocks -- 0 resets the shared shadow state at the start of
    every forward pass (block 0 always runs first, exactly once per model() call)."""

    def _hook(args, extra_options):
        if state.hard_disabled:
            return extra_options["original_block"](args)
        if index == 0:
            state.neg_h = None
        try:
            return _shadow_forward(state, block, args, extra_options)
        except Exception as e:
            if not state.warned_fallback:
                state.warned_fallback = True
                print(f"[FunPackSceneChain] H3 shadow negative: block {index} failed "
                      f"({type(e).__name__}: {str(e)[:120]}); falling back to stock H3 "
                      f"blocks for the rest of this run. Core H3 internals may have "
                      f"changed since this was ported.")
            state.hard_disabled = True
            return extra_options["original_block"](args)

    return _hook


def _shadow_forward(state, block, args, extra_options):
    x = args["img"]
    t_emb = args["t_emb"]
    mod_segments = args["mod_segments"]
    rope = args["rope_freqs"]
    to = args["transformer_options"]

    sigmas = to.get("sample_sigmas", to.get("sigmas"))
    progress = sigma_progress(args.get("timestep", to.get("timestep")), sigmas)
    active = state.alpha > 0 and (state.video_scale != 1.0 or state.audio_scale != 1.0)
    if progress is not None:
        active = active and state.start_percent <= progress <= state.end_percent
    if not active or len(mod_segments) < 3:
        return extra_options["original_block"](args)

    state.pos_text_len = int(x.shape[0])  # placeholder, corrected below once c_crossattn known
    cross_attn = to.get("c_crossattn")
    pos_text_len = (int(cross_attn.shape[1])
                    if isinstance(cross_attn, torch.Tensor) and cross_attn.ndim >= 3
                    else None)
    if pos_text_len is None:
        return extra_options["original_block"](args)
    state.pos_text_len = pos_text_len

    audio_a, audio_b, audio_row = mod_segments[-2]
    video_a, video_b, video_row = mod_segments[-1]
    if (int(audio_row) % 3) != 2 or (int(video_row) % 3) != 0 or audio_b != video_a \
            or video_b != x.shape[0]:
        if not state.warned_layout:
            state.warned_layout = True
            print("[FunPackSceneChain] H3 shadow negative: could not prove the final "
                  "audio/video target segments on this core's H3 block layout -- "
                  "bypassing shadow guidance.")
        return extra_options["original_block"](args)

    presentation_runs, user_run = _presentation_plan(mod_segments, pos_text_len)
    if presentation_runs is None:
        if not state.warned_layout:
            state.warned_layout = True
            print("[FunPackSceneChain] H3 shadow negative: could not identify the H3 "
                  "Qwen presentation span -- bypassing shadow guidance.")
        return extra_options["original_block"](args)

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

    h_pos = block.norm1(x)
    for a, b, row in mod_segments:
        _mod_one(h_pos[a:b], shift_msa, scale_msa, row)
    attn_pos = block.attn(h_pos, rope_freqs=rope, transformer_options=to)

    neg_h = state.neg_h
    if neg_h is None:
        neg_h = _prepare_negative(state, x, to)
        if neg_h is None:
            return extra_options["original_block"](args)

    h_alt, rope_alt, neg_a, neg_b, alt_suffix_start, text_row = _build_shadow_attention_input(
        state, block, neg_h, h_pos, rope, presentation_runs, user_run,
        pos_text_len, video_b, shift_msa, scale_msa, to)
    attn_neg = block.attn(h_alt, rope_freqs=rope_alt, transformer_options=to)
    a_span = (alt_suffix_start + (audio_a - pos_text_len), alt_suffix_start + (audio_b - pos_text_len))
    v_span = (alt_suffix_start + (video_a - pos_text_len), alt_suffix_start + (video_b - pos_text_len))

    if state.video_scale != 1.0:
        na, nb = v_span
        attn_pos[video_a:video_b] = nag_blend(attn_pos[video_a:video_b], attn_neg[na:nb],
                                              state.video_scale, state.tau, state.alpha)
    if state.audio_scale != 1.0:
        na, nb = a_span
        attn_pos[audio_a:audio_b] = nag_blend(attn_pos[audio_a:audio_b], attn_neg[na:nb],
                                              state.audio_scale, state.tau, state.alpha)

    x_out = x.clone()
    for a, b, row in mod_segments:
        _gate_slice(x_out, attn_pos, gate_msa, a, b, row)
    h2 = block.norm2(x_out)
    for a, b, row in mod_segments:
        _mod_one(h2[a:b], shift_mlp, scale_mlp, row)
    mlp_pos = block.mlp(h2)
    for a, b, row in mod_segments:
        _gate_slice(x_out, mlp_pos, gate_mlp, a, b, row)

    # Advance the shadow branch through this block too, same as the positive stream, so
    # the NEXT block's repulsion target reflects the negative text at the same depth.
    neg_x = neg_h[:neg_b - neg_a].clone()
    attn_shadow = attn_neg[neg_a:neg_b]
    neg_x.addcmul_(attn_shadow, gate_msa[text_row].to(neg_x.dtype))
    neg2 = block.norm2(neg_x)
    _mod_one(neg2, shift_mlp, scale_mlp, text_row)
    neg_x.addcmul_(block.mlp(neg2), gate_mlp[text_row].to(neg_x.dtype))
    state.neg_h = neg_x

    return {"img": x_out}
