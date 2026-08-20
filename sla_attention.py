"""SLA: block-sparse attention for MiniMax H3, as a loader choice.

ComfyUI has no sparse-attention backend for H3, which is why lightx2v's SLA turbo
LoRA gives no speedup on its own: the LoRA is the *adaptation* to sparsity, not the
acceleration. This module supplies the missing half — mean-pool Q and a smoothed K
into blocks, score them with one small matmul, and attend only the top
``1 - sparsity`` fraction of key blocks per query block. Nothing is trained and
nothing is loaded; the published SLA files contain only ordinary LoRA tensors.

Ported from ComfyUI-H3-SLA-Attention (MIT), whose kernel and block map are vendored
from LightX2V (Apache-2.0) — see sla_kernel.py / sla_block_map.py for the change
lists. What is FunPack's here: this file, the H3 model check, and the loader wiring,
so turning it on is picking a value in the diffusion model loader rather than adding
and wiring a node.

The hook is ``transformer_options["optimized_attention_override"]``, which
``wrap_attn`` consults and H3's one attention call site reaches. The legacy
``set_model_attn1_patch`` is the SD-UNet path a DiT never consults: a patch installed
there reports success and silently does nothing, which is why the invocation counter
below exists and why a run that never sparsified says so in the log.
"""
import logging

SLA_NAME = "sla_h3"

_H3_HEAD_DIM = 128
_H3_MODEL_NAMES = ("MiniMaxH3Model",)

# Validated on an RTX 5090 at 768p/15s (ComfyUI-H3-SLA-Attention's own measurements):
# 3.7x the attention throughput of stock ComfyUI at sparsity 0.90 / block 64.
SLA_DEFAULTS = {
    "sparsity_ratio": 0.90,
    "block_size": 64,
    "min_seq_len": 8192,
    "dense_last_steps": 0,
    "protect_audio": True,
    "enabled": True,
}


def _torch():
    import torch
    return torch


def sla_available():
    """True when this machine could actually run the kernel.

    Triton and CUDA, nothing else — the ladder in sla_kernel.py handles differing
    shared-memory limits per architecture. Offered as a choice only where it can run,
    like every other backend in the loader's list.
    """
    try:
        import torch
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return bool(torch.cuda.is_available())


def is_h3_model(model):
    """Whether this patcher holds a MiniMax H3 diffusion model.

    SLA is only sound on H3: the sparsity ratio it runs at is the one the turbo LoRA
    was distilled to tolerate, and H3's packed [text | cond | audio | video] sequence
    is what the prefix pinning is for. Head shape alone would also match LTX, and
    silently sparsifying LTX attention is a quality loss with no LoRA compensating it.
    """
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    return type(diffusion).__name__ in _H3_MODEL_NAMES


def new_state():
    return {
        "calls": 0,        # sparse invocations this run
        "dense": 0,        # fall-throughs this run
        "step": 0,
        "seq": 0,
        "kept": 0,
        "blocks": 0,
        "pinned": 0,
        "backend": None,   # what we displaced
        "failed": None,    # first kernel failure, if any
    }


def _summarise(state, sparsity, blkq, blkk):
    """One line per sampling run. Never one per block — there are 50 of those."""
    if state["calls"] == 0:
        logging.warning(
            "[FunPack SLA] installed but never invoked — attention was NOT sparsified "
            "(%d dense fall-throughs). Check that the MODEL reaching the sampler is the "
            "one this loader produced.", state["dense"])
        return
    real = 1.0 - (state["kept"] / state["blocks"]) if state["blocks"] else 0.0
    logging.info(
        "[FunPack SLA] %d calls | S=%d | blocks %d/%d kept (%.1f%% sparse, asked %.0f%%) "
        "| %d pinned | BLK=%dx%d | %d dense fall-throughs | displaced %s",
        state["calls"], state["seq"], state["kept"], state["blocks"], real * 100.0,
        sparsity * 100.0, state["pinned"], blkq, blkk, state["dense"],
        state["backend"] or "?")
    if state["failed"] is not None:
        logging.warning("[FunPack SLA] kernel fell back to dense at least once: %s",
                        state["failed"])


def make_override(state, sparsity_ratio, blkq, blkk, min_seq_len, protect_audio=True):
    """The `optimized_attention_override` wrap_attn hands every attention call to."""
    torch = _torch()
    ok_dtypes = (torch.bfloat16, torch.float16)
    topk_ratio = 1.0 - sparsity_ratio

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            state["dense"] += 1
            return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape, **kwargs)

        if state["backend"] is None:
            state["backend"] = getattr(func, "__name__", repr(func))

        to = kwargs.get("transformer_options") or {}

        # Anything that is not the packed H3 self-attention goes straight through. The
        # min_seq_len guard is what keeps the 2-block token refiner (S = text length)
        # and low-resolution runs dense, where selection costs more than it saves.
        if (
            not skip_reshape
            or mask is not None
            or q.ndim != 4
            or q.shape[-1] != _H3_HEAD_DIM
            or q.dtype not in ok_dtypes
            or q.shape[2] < min_seq_len
            or to.get("_funpack_sla_dense", False)
        ):
            return dense()

        try:
            # Imported here, not at module scope: triton is absent on plenty of machines
            # and importing it eagerly would take the whole module — and the loader —
            # down with it.
            try:
                from .sla_block_map import get_block_map
                from .sla_kernel import block_sparse_attention
            except ImportError:
                from sla_block_map import get_block_map
                from sla_kernel import block_sparse_attention

            B, H, S, D = q.shape

            # [1, H, S, D] -> [1, S, H, D]. H3 builds q/k/v as [S, H, D] and transposes
            # for the call, so this transposes back onto the original memory and the
            # copy is a no-op. A BHSD kernel would cost a real ~1.3 GB copy per tensor.
            qb, kb, vb = (t.transpose(1, 2) for t in (q, k, v))
            if not qb.is_contiguous():
                qb, kb, vb = qb.contiguous(), kb.contiguous(), vb.contiguous()

            # Pin the [text | cond | audio] prefix into every query's selection. Audio is
            # ~1% of the packed sequence, so plain top-k routinely drops all of it and the
            # soundtrack degrades while the video still looks fine. 0 when the layout is
            # unavailable, which disables the protection rather than guessing at it.
            prefix = int(to.get("_funpack_sla_prefix", 0) or 0) if protect_audio else 0
            if prefix >= S:
                prefix = 0

            lut, topk = get_block_map(qb, kb, topk_ratio, blkq, blkk, protect_upto=prefix)
            out = block_sparse_attention(qb, kb, vb, lut, topk, blkq, blkk)

            state["calls"] += 1
            state["seq"] = S
            state["kept"] = topk
            state["blocks"] = (S + blkk - 1) // blkk
            state["pinned"] = (prefix + blkk - 1) // blkk

            if skip_output_reshape:
                return out.transpose(1, 2)
            return out.reshape(B, S, H * D)

        except Exception as exc:  # noqa: BLE001 - a bad kernel must not kill the run
            if state["failed"] is None:
                state["failed"] = "%s: %s" % (exc.__class__.__name__, exc)
                logging.debug("[FunPack SLA] kernel failed", exc_info=True)
            return dense()

    return override


def make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps):
    """DIFFUSION_MODEL wrapper: per-step state, and the end-of-run summary.

    Registered once and then reused — ComfyUI caches node outputs, so this closure
    outlives a single sampling run. The step counter therefore has to reset itself, or
    every run after the first drifts permanently into the trailing-dense window and
    silently stops sparsifying.
    """

    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        n_steps = max(1, len(to.get("sample_sigmas", [])) - 1)

        if state["step"] >= n_steps:      # new run
            state["step"] = 0
            state["calls"] = 0
            state["dense"] = 0
            state["failed"] = None
        state["step"] += 1

        # PackedLayout.segments is [(start, stop, kind), …] over
        # [text | cond/ref | audio | video]; the video start is therefore the length of
        # everything that must stay exactly attended. It lives on the payload, which
        # never reaches the attention call site, so the wrapper is the only place it can
        # be picked up.
        prefix = 0
        layout = minimax_payload.get("layout") if minimax_payload else None
        for seg in getattr(layout, "segments", ()) or ():
            if len(seg) == 3 and seg[2] == "video":
                prefix = int(seg[0])
                break
        to["_funpack_sla_prefix"] = prefix
        to["_funpack_sla_dense"] = bool(
            dense_last_steps > 0 and state["step"] > n_steps - dense_last_steps)

        # Forward minimax_payload only when H3 actually supplied one: every other
        # diffusion model would raise TypeError on the unexpected kwarg, turning a
        # graceful no-op into a crash mid-sampling.
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        out = executor.original(x, timestep, context,
                                transformer_options=transformer_options, **kwargs)

        if state["step"] >= n_steps:
            _summarise(state, sparsity_ratio, blkq, blkk)
        return out

    return wrapper


def install_sla(model, sparsity_ratio=None, block_size=None, min_seq_len=None,
                dense_last_steps=None, protect_audio=None, enabled=None):
    """Give `model` block-sparse H3 attention. Returns (model, status line).

    Weights are untouched: this installs an attention override and a per-step wrapper
    on a clone. A model that is not MiniMax H3 comes back unchanged and says so —
    the sparsity is only safe at the ratio H3's turbo LoRA was distilled to tolerate.
    """
    def pick(value, key):
        return SLA_DEFAULTS[key] if value is None else value

    sparsity_ratio = float(pick(sparsity_ratio, "sparsity_ratio"))
    blkq = int(pick(block_size, "block_size"))
    min_seq_len = int(pick(min_seq_len, "min_seq_len"))
    dense_last_steps = int(pick(dense_last_steps, "dense_last_steps"))
    protect_audio = bool(pick(protect_audio, "protect_audio"))

    if not bool(pick(enabled, "enabled")):
        # A dense baseline without touching the backend choice, so an A/B is one click
        # and the settings you were testing are still on the node when you switch back.
        return model, "attention: sla_h3 off (dense baseline)"
    if not is_h3_model(model):
        return model, ("attention: sla_h3 requested but this is not a MiniMax H3 model — "
                       "left on the launched backend")
    if not sla_available():
        return model, ("attention: sla_h3 requested but this machine has no CUDA+Triton — "
                       "left on the launched backend")

    # BLKK=64 is not a typo. On sm_120 the 128x128 tile needs 160 KB of shared memory
    # against a ~99 KB limit and cannot launch at all; 128x64 both fits and measured
    # fastest. LightX2V picks the same split for its sage2 path off sm90.
    blkk = 64 if blkq == 128 else blkq

    state = new_state()
    patched = model.clone()
    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = make_override(
        state, sparsity_ratio, blkq, blkk, min_seq_len, protect_audio)
    patched.model_options["transformer_options"] = to
    patched.add_wrapper_with_key(
        "diffusion_model", "funpack_sla_state",
        make_wrapper(state, sparsity_ratio, blkq, blkk, dense_last_steps))

    return patched, (f"attention: sla_h3 | sparsity={sparsity_ratio:.2f} BLK={blkq}x{blkk} "
                     f"min_seq_len={min_seq_len} dense_last_steps={dense_last_steps} "
                     f"protect_audio={protect_audio}")
