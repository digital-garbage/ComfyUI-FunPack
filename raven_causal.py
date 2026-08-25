"""Chunk-causal rollout for MiniMax H3, on FunPack's own schedule and step rule.

WHAT THIS IS
------------
RAVEN's contribution is not a sampler. It is a way of *shaping the sequence*: instead of
denoising the whole clip at once, the clip is cut into time chunks, each chunk is carried to
completion, committed into a KV cache as clean context, and the next chunk is generated
attending to that cache. The model gets real memory of what it already drew, so a long clip
stays one continuous shot instead of a stitched chain.

That shaping is separable from the four-step consistency sampler RAVEN ships with it, and this
module keeps them apart so each can be tested on its own:

* the CHUNKING and the cache come from the RAVEN package (``prefill_text`` / ``forward_chunk``);
* the SCHEDULE is FunPack's — any scheduler, any step count, including bong_tangent;
* the STEP RULE is selectable — RAVEN's consistency transition, or plain flow euler, or
  ancestral euler.

WHY THE FIRST CHUNK IS DENSE
----------------------------
RAVEN's own sampler refuses keyframe pins (``fl2va``) and reference blocks (``ref2va``): its
causal layout models text and target rows only, so an i2v anchor has nowhere to go. Rather
than extend that layout, chunk 0 is sampled through FunPack's ORDINARY dense H3 path — where
pins, references, region locks, AdaLN gains and everything else already work — and only THEN
committed into the cache as clean context. Chunks 1..N continue causally from it.

So the anchor lands where an anchor belongs (the opening frames), every conditioning feature
FunPack has keeps working, and the causal lane is only ever asked to do the thing it was
trained for: continue from cached context.

REQUIREMENTS
------------
The RAVEN LoRA is not optional. The chunk-causal attention pattern is what it was trained to
read; the base H3 weights have never seen a KV cache. Without the LoRA this produces output,
and the output is out of distribution. The caller is told, once, rather than discovering it.
"""

from __future__ import annotations

import importlib
import os
import sys

import torch

#: video latents per chunk, and the clock span that many latents cover. Re-derived here so a
#: probe can check them against the installed package instead of trusting a remembered number.
VIDEO_LATENTS_PER_CHUNK = 5
CHUNK_T_SPAN = 85.0 / 3.0

#: how a chunk's next-sigma is reached from its prediction
STEP_RULES = ("consistency", "euler", "euler_ancestral")

_PROBE = {"state": None}


def locate_raven():
    """Import ``raven_streaming``, from ComfyUI's custom_nodes if it is not already on the path.

    Returns (module, reason). The module is None when it cannot be had, and `reason` says why
    in the terms the user can act on — this is an optional capability, not a dependency.
    """
    if _PROBE["state"] is not None:
        return _PROBE["state"]
    try:
        module = importlib.import_module("raven_streaming")
        _PROBE["state"] = (module, "")
        return _PROBE["state"]
    except ImportError:
        pass
    # Not on sys.path: ComfyUI does not add a custom node's directory for other packs to
    # import from. Find it the way a user would describe it — the sibling directory.
    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes = os.path.dirname(here)
    for name in sorted(os.listdir(custom_nodes)) if os.path.isdir(custom_nodes) else []:
        candidate = os.path.join(custom_nodes, name)
        if not os.path.isdir(os.path.join(candidate, "raven_streaming")):
            continue
        if candidate not in sys.path:
            sys.path.append(candidate)
        try:
            module = importlib.import_module("raven_streaming")
            _PROBE["state"] = (module, "")
            return _PROBE["state"]
        except Exception as error:                       # found, but will not import
            _PROBE["state"] = (None, f"found {name} but it did not import ({error})")
            return _PROBE["state"]
    _PROBE["state"] = (
        None,
        "ComfyUI-MiniMax-H3-RAVEN-Streaming is not installed. The causal rollout borrows its "
        "chunk cache; install it into custom_nodes and restart ComfyUI.",
    )
    return _PROBE["state"]


#: pixel frames one full chunk covers: FRAME_PER_TOKEN over 5 latents is 1+4+4+4+4
FRAMES_PER_CHUNK = 17
#: audio latents per pixel frame — 40 latents/s against the model's fixed 24 fps
AUDIO_LATENTS_PER_FRAME = 40.0 / 24.0


def chunk_bounds(latent_t, audio_t):
    """Plan the chunk cut: (video_start, video_stop, audio_start, audio_stop) per chunk.

    Video cuts every ``VIDEO_LATENTS_PER_CHUNK`` latents. Audio is NOT cut by a remembered
    29/28/28 cadence — it is derived from the shared clock, so a chunk owns the audio latents
    whose time falls inside its video span and the cadence comes out as a consequence.

    The LIVE rollout uses the RAVEN package's own layout object, because that is what its
    ``forward_chunk`` indexes. This exists to plan and report a run before the package is
    touched (how many chunks, how long), and to be checkable against theirs.
    """
    latent_t, audio_t = int(latent_t), int(audio_t)
    if latent_t <= 0:
        return []
    bounds = []
    audio_cursor = 0
    starts = list(range(0, latent_t, VIDEO_LATENTS_PER_CHUNK))
    for index, video_start in enumerate(starts):
        video_stop = min(latent_t, video_start + VIDEO_LATENTS_PER_CHUNK)
        if index == len(starts) - 1:
            audio_stop = audio_t                      # the tail owns whatever is left
        else:
            frames_done = FRAMES_PER_CHUNK * (index + 1)
            audio_stop = min(audio_t, int(round(frames_done * AUDIO_LATENTS_PER_FRAME)))
            audio_stop = max(audio_stop, audio_cursor)
        bounds.append((video_start, video_stop, audio_cursor, audio_stop))
        audio_cursor = audio_stop
    return bounds


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


def build_session(model, positive, latent, *, sink=2, window=2, storage="cpu_pinned",
                  device=None, compute_dtype=None):
    """Assemble everything one causal run needs, or explain why it cannot be assembled.

    Returns (session, reason). `session` is None when the run cannot be built, and `reason` is
    written for the person who has to fix it, not for a log parser.

    The pieces are the RAVEN package's own — its contracts parse the sockets, its layout cuts
    the chunks, its cache holds the K/V, its causal model does the per-chunk forward. FunPack
    supplies only the schedule and the step rule, which is the whole point of the split.
    """
    module, reason = locate_raven()
    if module is None:
        return None, reason
    try:
        from raven_streaming import cache as cache_mod
        from raven_streaming import contracts
    except Exception as error:                       # installed but not importable
        return None, f"the RAVEN package is present but its modules did not load ({error})"
    try:
        resolved = contracts.resolve_model(model)
        conditioning = contracts.parse_conditioning(positive)
        request = contracts.parse_latent(latent, warn_experimental=False)
        layout = request.layout(conditioning.text_len, warn_experimental=False)
    except Exception as error:
        return None, f"this run is not one the causal lane can take: {error}"

    diffusion = resolved.diffusion_model
    if not hasattr(diffusion, "forward_chunk") or not hasattr(diffusion, "prefill_text"):
        return None, ("the loaded model is a stock bidirectional H3. The causal lane needs the "
                      "chunk-causal DiT — load it through RAVEN Model Loader, which also "
                      "attaches the RAVEN LoRA the causal attention pattern was trained for.")
    blocks = getattr(diffusion, "blocks", None)
    if not blocks:
        return None, "the loaded model exposes no DiT blocks to cache"

    if device is None:
        device = resolved.load_device if resolved.load_device is not None else request.device
    if compute_dtype is None:
        # One dtype for the prefill AND every chunk: the cache is filled in one and read in
        # the other otherwise, and the attention module refuses that mid-rollout.
        compute_dtype = getattr(diffusion, "dtype", None) or request.dtype

    kv = cache_mod.ChunkKVCache(len(blocks), sink=int(sink),
                                window=None if window is None else int(window),
                                storage=str(storage))
    return {
        "module": module,
        "model": diffusion,
        "patcher": resolved.patcher,
        "conditioning": conditioning,
        "request": request,
        "layout": layout,
        "cache": kv,
        "device": torch.device(device),
        "compute_dtype": compute_dtype,
    }, ""


def run_session(session, *, sigmas, step_rule="consistency", eta=1.0, seed=0,
                known_chunks=0, known_video=None, known_audio=None,
                on_chunk=None, cancel=None, transformer_options=None):
    """Drive one causal rollout over an assembled session. Returns (video, audio).

    The text is written into the cache as chunk 0, alone and once. Folding it into the first
    media chunk would let text rows attend media rows, and every later chunk assumes the text
    keys it cached are the ones the model saw.
    """
    layout = session["layout"]
    model = session["model"]
    cache = session["cache"]
    device = session["device"]
    compute_dtype = session["compute_dtype"]
    conditioning = session["conditioning"]
    options = transformer_options or {}

    model.prefill_text(
        conditioning.cross_attn.to(device=device, dtype=compute_dtype),
        cache=cache,
        transformer_options=options,
        text_token_tags=(None if conditioning.token_tags is None
                         else conditioning.token_tags.to(device)),
        compute_dtype=compute_dtype,
    )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def draw_noise(shape):
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)

    request = session["request"]
    video_noise = draw_noise(layout.video_latent_shape(request.video_channels)
                             if hasattr(layout, "video_latent_shape") else known_video.shape)
    audio_noise = draw_noise(layout.audio_latent_shape(request.audio_channels)
                             if hasattr(layout, "audio_latent_shape") else known_audio.shape)

    chunks = [(c.video_start, c.video_stop, c.audio_start, c.audio_stop) for c in layout.chunks]

    def forward(video_xt, audio_xt, index, sigma):
        return model.forward_chunk(
            video_latent=video_xt, audio_latent=audio_xt, layout=layout, chunk_index=index,
            cache=cache, role="noise", video_sigma=float(sigma), audio_sigma=float(sigma),
            update_cache=False, transformer_options=options, compute_dtype=compute_dtype,
        )

    def commit(video_x0, audio_x0, index):
        model.forward_chunk(
            video_latent=video_x0, audio_latent=audio_x0, layout=layout, chunk_index=index,
            cache=cache, role="clean",
            video_eps=draw_noise(video_x0.shape), audio_eps=draw_noise(audio_x0.shape),
            update_cache=True, transformer_options=options, compute_dtype=compute_dtype,
        )

    return causal_rollout(
        chunks=chunks, sigmas=sigmas, forward=forward, commit=commit, draw_noise=draw_noise,
        video_noise=video_noise, audio_noise=audio_noise, step_rule=step_rule, eta=eta,
        known_chunks=known_chunks, known_video=known_video, known_audio=known_audio,
        on_chunk=on_chunk, cancel=cancel,
    )
