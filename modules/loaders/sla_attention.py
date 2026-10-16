"""SLA: block-sparse attention for MiniMax H3, as a loader choice.

Not a module: see common.py's own note -- a plain file at the domain level is a
helper the loaders import, and ships no node of its own.

ComfyUI has no sparse-attention backend for H3, which is why lightx2v's SLA turbo
LoRA gives no speedup on its own: the LoRA is the *adaptation* to sparsity, not the
acceleration. This module supplies the missing half — mean-pool Q and a smoothed K
into blocks, score them with one small matmul, and attend only the top
``1 - sparsity`` fraction of key blocks per query block. Nothing is trained and
nothing is loaded; the published SLA files contain only ordinary LoRA tensors.

Ported from ComfyUI-H3-SLA-Attention (MIT; refreshed against its 2026-09-21 version),
whose kernel and block map are vendored from LightX2V (Apache-2.0) — see sla_kernel.py /
sla_block_map.py for the change lists. What is FunPack's here: this file, the H3 model
check, and the loader wiring, so turning it on is picking a value in the diffusion model
loader rather than adding and wiring a node.

Two engines run the sparse calls: comfy_kitchen's compiled ``sol_attn`` (int8, the default
where installed) and the vendored Triton kernel (block size, stabilize_motion and the exact
Light reference tier are its alone). That pack's Triton int8-QK and tail additions are not
ported: it marks both untested on hardware, and sol_attn does both natively.

The hook is ``transformer_options["optimized_attention_override"]``, which
``wrap_attn`` consults and H3's one attention call site reaches. The legacy
``set_model_attn1_patch`` is the SD-UNet path a DiT never consults: a patch installed
there reports success and silently does nothing, which is why the invocation counter
below exists and why a run that never sparsified says so in the log.
"""
import logging

from .._core import dit_hooks as _dit_hooks, patching, traits as _traits

SLA_NAME = "sla_h3"

_H3_HEAD_DIM = 128
# Qualified on purpose, per core/traits.py's has_block: "a bare name matches on
# the class name alone... actively wrong" for a false positive that silently
# sparsifies a model this was never validated on -- exactly the LTX-collision
# risk this file's own docstring already warns about, just not guarded against
# strongly enough before this fix (v4 matched the bare name; found in extensive_
# testing round 1 on this port). Kept as a local string matching modules/models/
# minimax_h3's own MODEL_CLASS rather than importing that module -- it pulls in
# comfy_extras at import time, which this file deliberately does not require
# just to be importable on a machine with no ComfyUI on the path. Keep the two
# in sync if the H3 model class is ever renamed.
_H3_MODEL_CLASS = "comfy.ldm.minimax.model.MiniMaxH3Model"

# Validated on an RTX 5090 at 768p/15s (ComfyUI-H3-SLA-Attention's own measurements):
# 3.7x the attention throughput of stock ComfyUI at sparsity 0.90 / block 64.
SLA_DEFAULTS = {
    "sparsity_ratio": 0.90,
    "block_size": 64,
    "min_seq_len": 8192,
    "dense_last_steps": 0,
    "protect_audio": True,
    "enabled": True,
    "engine": "comfy_kitchen",
    "dense_steps": "1",
    "method": "sla",
    "tau": 1.3,
    "references": "off",
    "tail": False,
    "stabilize_motion": False,
}
ENGINES = ("comfy_kitchen", "triton")
# sla: each query block keeps the top (1 - sparsity) of key blocks, what SLA turbo LoRAs were distilled against.
# sol-attn: comfy_kitchen picks per head and block by a threshold (tau), training-free. vsa: FastVideo's cube
# tiling, for models trained for it -- ComfyUI's own Model Sparse Attention runs it, handed the settings here.
METHODS = ("sla", "sol-attn", "vsa")
REFERENCES = ("off", "light", "heavy")
_LIGHT_KEEP = 0.15          # "light": each reference span keeps its best 15% of blocks for every query
_CK_BLOCK = 64              # sol_attn's fixed key block

try:        # ComfyUI with its compiler (PR #16148): a long-lived tensor made mid-graph must be made outside it
    from comfy.model_prefetch import pause_malloc_graph
except ImportError:
    import contextlib
    pause_malloc_graph = contextlib.nullcontext


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


def ck_available():
    """comfy_kitchen's sol_attn (comfy-kitchen >= 0.2.32) compiled for this GPU."""
    try:
        import comfy_kitchen as ck
        return bool(ck.sol_attn_is_available())
    except Exception:  # noqa: BLE001
        return False


def is_h3_model(model):
    """Whether this patcher holds a MiniMax H3 diffusion model.

    SLA is only sound on H3: the sparsity ratio it runs at is the one the turbo LoRA
    was distilled to tolerate, and H3's packed [text | cond | audio | video] sequence
    is what the span protection is for. Head shape alone would also match LTX, and
    silently sparsifying LTX attention is a quality loss with no LoRA compensating it
    -- which is also why the class name is matched qualified (module + name), not
    bare: a bare name is exactly as blind to an unrelated same-named class as it is
    to LTX's different head shape.

    has_block's contract is "does this model CONTAIN one", not "IS this model
    one" -- true today because nothing in v5 wraps diffusion_model in anything
    that could nest an unrelated MiniMaxH3Model submodule (checked: comfy's own
    model_base.py assigns the raw architecture instance directly, and no
    composite/multi-model pipeline exists in this codebase yet). Revisit this
    gate if that ever changes -- the same assumption underpins minimax_h3's own
    equally consequential is_h3() gate, so it would need fixing in both places.
    """
    return _traits.has_block(model, _H3_MODEL_CLASS)


def block_ranges(spans, blk, nk):
    """Token spans [(start, stop)] -> merged half-open key-block ranges, rounded outward."""
    out = []
    for a, b in sorted((min(nk, max(0, int(a)) // blk), min(nk, (int(b) + blk - 1) // blk)) for a, b in spans if b > a):
        if a >= b:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out

def minus(ranges, cut):
    """Block ranges with every block in `cut` taken out."""
    for c0, c1 in cut:
        ranges = [p for a, b in ranges for p in ((a, min(b, c0)), (max(a, c1), b)) if p[0] < p[1]]
    return ranges



def parse_steps(spec):
    """"1,3-4" -> {0, 2, 3}: steps as a person counts them (1 is the first) -> the sampler's 0-based
    indices. A token that is not such a number (0, a letter, past step 1000) is skipped and named."""
    steps, bad = set(), []
    for tok in str(spec or "").replace(" ", "").split(","):
        if not tok or tok == "0":          # blank, or "0": no dense steps (the tooltip and the commit say so)
            continue
        a, _, b = tok.partition("-")
        if a.isdigit() and (not b or b.isdigit()) and 1 <= min(int(a), int(b or a)) and max(int(a), int(b or a)) <= 1000:
            lo, hi = sorted((int(a), int(b or a)))
            steps.update(range(lo - 1, hi))
        else:
            bad.append(tok)
    return frozenset(steps), bad


def _runs(numbers):
    """[1, 2, 3, 6] -> "1-3, 6"."""
    out = []
    for n in sorted(numbers):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def spans(payload):
    """H3's packed layout -> (video start, spans to keep exact, visual reference spans).

    Exact: the language tokens and every audio stream (target, reference, conditioning) --
    audio is ~1% of the sequence, so plain top-k routinely drops all of it and the
    soundtrack degrades while the picture looks fine. Visual references (the vision tokens
    inside the text span, conditioning frames, reference images) stay sparse unless
    `references` asks otherwise. No layout: nothing is protected, rather than guessed at.
    """
    layout = payload.get("layout") if payload else None
    tags = payload.get("text_token_tags") if payload else None
    video, keep, refs = 0, [], []
    for seg in getattr(layout, "segments", ()) or ():
        if len(seg) != 3:
            continue
        a, b, kind = int(seg[0]), int(seg[1]), seg[2]
        if kind == "text":
            try:
                t = [int(x) for x in (tags.reshape(-1).tolist() if hasattr(tags, "reshape") else tags)]
            except TypeError:
                t = None
            if t is None or len(t) != b - a:          # untagged: all of it is language
                keep.append((a, b))
                continue
            i = 0
            for j in range(1, len(t) + 1):            # tag runs: 0 = a vision token, else language
                if j == len(t) or t[j] != t[i]:
                    (refs if t[i] == 0 else keep).append((a + i, a + j))
                    i = j
        elif kind in ("audio", "ref_audio", "cond_audio"):
            keep.append((a, b))
        elif kind in ("cond", "ref_img"):
            refs.append((a, b))
        elif kind == "video" and not video:
            video = a
    return video, keep, refs


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
        "layer": 0,        # this step's call count: the Nth call is the same layer every step
        "history": {},     # layer -> last step's choices, for stabilize_motion
        "order": None,     # comfy_kitchen: (S, protected spans, key order, sink blocks), same every layer
        "last": None,      # last step index read from the schedule
        "dense_at": set(), # steps (counted from 1) that ran at full attention this run
        "n": 0,            # this run's step count
        "summed": False,   # this run's summary is logged
    }


def _summarise(state, cfg):
    """One line per sampling run. Never one per block — there are 50 of those."""
    if state["calls"] == 0:
        logging.warning(
            "[FunPack SLA] installed but never invoked — attention was NOT sparsified "
            "(%d dense fall-throughs). Check that the MODEL reaching the sampler is the "
            "one this loader produced.", state["dense"])
        return
    real = 1.0 - (state["kept"] / state["blocks"]) if state["blocks"] else 0.0
    logging.info(
        "[FunPack SLA] %s | %d calls | S=%d | blocks %d/%d kept (%.1f%% sparse, asked %.0f%%) "
        "| %d pinned | %d dense fall-throughs | displaced %s",
        cfg["engine"], state["calls"], state["seq"], state["kept"], state["blocks"], real * 100.0,
        cfg["sparsity_ratio"] * 100.0, state["pinned"], state["dense"], state["backend"] or "?")
    logging.info("[FunPack SLA] full attention on %s of %d", f"steps {_runs(state['dense_at'])}"
                 if state["dense_at"] else "no step", state["n"])
    if state["failed"] is not None:
        logging.warning("[FunPack SLA] kernel fell back to dense at least once: %s",
                        state["failed"])


def _mean_pool_torch(x, blk):
    """``(B, L, H, D)`` -> ``(B, H, ceil(L/blk), D)`` fp32, the mean over each block: the same numbers as the
    Triton pool in sla_block_map, in plain torch, so Light works without Triton."""
    import torch
    B, L, H, D = x.shape
    pad = (-L) % blk
    x = torch.nn.functional.pad(x.float().transpose(1, 2), (0, 0, 0, pad))      # (B, H, L+pad, D)
    counts = torch.nn.functional.pad(torch.ones(L, device=x.device), (0, pad)).reshape(1, 1, -1, blk, 1).sum(dim=3)
    return x.reshape(B, H, -1, blk, D).sum(dim=3) / counts      # the last, short block averages its real rows only


def _ck_attention(state, cfg, qb, kb, vb, keep, refs, scale):
    """comfy_kitchen's sol_attn: one 64-token key-block range (`sink_blocks`) is always exact. Attention
    does not care about key order, so when the protected blocks are not one run, K and V are put in an
    order where they are -- a copy per call, needed only then. -> (output, blocks, pinned)."""
    import comfy_kitchen as ck

    torch = _torch()
    S = kb.shape[1]
    nk = (S + _CK_BLOCK - 1) // _CK_BLOCK
    light = cfg["references"] == "light"
    cached = state["order"]
    if cached and cached[0] == S and cached[1] == keep and not light:
        _, _, idx, sink = cached
    else:
        pinned = block_ranges(keep, _CK_BLOCK, nk)
        if light:       # one set for every query here: ranked by the average query, not each one (Triton does that)
            score = (_mean_pool_torch(qb, _CK_BLOCK).mean(dim=2, keepdim=True)
                     @ (_mean_pool_torch(kb, _CK_BLOCK) - kb.mean(dim=1, dtype=torch.float32)[:, :, None, :]).transpose(-1, -2))
            score = score.mean(dim=(0, 1, 2))
            for a, b in minus(block_ranges(refs, _CK_BLOCK, nk), pinned):
                best = torch.topk(score[a:b], max(1, round(_LIGHT_KEEP * (b - a)))).indices + a
                pinned += [(i, i + 1) for i in best.tolist()]
            pinned = block_ranges([(a * _CK_BLOCK, b * _CK_BLOCK) for a, b in pinned], _CK_BLOCK, nk)
        if len(pinned) <= 1:
            idx, sink = None, list(pinned[0]) if pinned else [0, 0]
        else:
            first = [i for a, b in pinned for i in range(a, b)]
            taken = set(first)
            blocks = first + [i for i in range(nk) if i not in taken]
            with pause_malloc_graph():
                idx = torch.cat([torch.arange(i * _CK_BLOCK, min(S, (i + 1) * _CK_BLOCK), device=kb.device)
                                 for i in blocks])
            sink = [0, len(first)]
        if not light:
            state["order"] = (S, keep, idx, sink)
    if idx is not None:
        kb, vb = kb.index_select(1, idx), vb.index_select(1, idx)
    cast = qb.dtype != torch.bfloat16           # the compiled kernel takes bf16 only
    q, k, v = ((t.to(torch.bfloat16) if cast else t) for t in (qb, kb, vb))
    if cfg["method"] == "sol-attn":       # threshold routing: the pooled tail is part of the method
        out = ck.sol_attn(q, k, v, tau=cfg["tau"], scale=scale, sink_blocks=sink, sink_q=[0, 0])
    else:
        out = ck.sol_attn(q, k, v, scale=scale, sink_blocks=sink, sink_q=[0, 0],
                          topk_ratio=1.0 - cfg["sparsity_ratio"], tail=cfg["tail"])
    return (out.to(qb.dtype) if cast else out), nk, sink[1] - sink[0]


def make_override(state, cfg, dense_fn=None, dense_label=None):
    """The `optimized_attention_override` wrap_attn hands every attention call to.

    `cfg` is install_sla's resolved settings. `dense_fn` is another override (same
    signature) handling everything SLA does not: the text refiner, masked calls, the dense
    steps, any non-H3 model. There is only one override slot, so without this, choosing SLA
    would silently discard the backend the user picked and drop those calls onto whatever
    ComfyUI was launched with. Sparse where sparsity applies, the chosen backend everywhere else.
    """
    torch = _torch()
    ok_dtypes = (torch.bfloat16, torch.float16)
    topk_ratio = 1.0 - cfg["sparsity_ratio"]
    blkq = cfg["block_size"]
    # BLKK=64 is not a typo. On sm_120 the 128x128 tile needs 160 KB of shared memory
    # against a ~99 KB limit and cannot launch at all; 128x64 both fits and measured
    # fastest. LightX2V picks the same split for its sage2 path off sm90.
    blkk = 64 if blkq == 128 else blkq
    heavy = cfg["references"] == "heavy"
    ref_keep = _LIGHT_KEEP if cfg["references"] == "light" else None

    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        def dense():
            state["dense"] += 1
            run = (lambda *a, **kw: dense_fn(func, *a, **kw)) if dense_fn is not None else func
            return run(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                       skip_reshape=skip_reshape,
                       skip_output_reshape=skip_output_reshape, **kwargs)

        if state["backend"] is None:
            state["backend"] = dense_label or getattr(func, "__name__", repr(func))

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
            or q.shape[2] < cfg["min_seq_len"]
            or to.get("_funpack_sla_dense", False)
        ):
            return dense()

        try:
            B, H, S, D = q.shape
            layer = state["layer"]
            state["layer"] = layer + 1

            # [1, H, S, D] -> [1, S, H, D]. H3 builds q/k/v as [S, H, D] and transposes
            # for the call, so this transposes back onto the original memory and the
            # copy is a no-op. A BHSD kernel would cost a real ~1.3 GB copy per tensor.
            # Each checked on its own: newer ComfyUI hands K over in another layout than Q (its fused
            # norm+RoPE), and assuming they match failed every call into dense.
            qb, kb, vb = (t if t.is_contiguous() else t.contiguous() for t in (x.transpose(1, 2) for x in (q, k, v)))

            video, keep, refs = to.get("_funpack_sla_spans") or (0, (), ())
            keep = tuple(keep if cfg["protect_audio"] else ()) + tuple(refs if heavy else ())

            if cfg["engine"] == "comfy_kitchen":
                out, blocks, pinned = _ck_attention(state, cfg, qb, kb, vb, keep, refs, D ** -0.5)
                kept = max(1, round(topk_ratio * blocks)) + pinned
            else:
                try:
                    from .sla_block_map import get_block_map
                    from .sla_kernel import block_sparse_attention
                except ImportError:
                    from sla_block_map import get_block_map
                    from sla_kernel import block_sparse_attention
                stable = cfg["stabilize_motion"]
                lut, kept, history = get_block_map(
                    qb, kb, topk_ratio, blkq, blkk, protect=keep, refs=() if heavy else refs,
                    ref_keep=ref_keep, prev=state["history"].get(layer) if stable else None,
                    sticky_from=video, remember=stable)
                if stable:      # one buffer per layer, refilled in place: what a captured graph can replay
                    held = state["history"].get(layer)
                    if held is None or held.shape != history.shape:
                        with pause_malloc_graph():
                            held = state["history"][layer] = torch.empty_like(history)
                    held.copy_(history)
                out = block_sparse_attention(qb, kb, vb, lut, kept, blkq, blkk)
                blocks = (S + blkk - 1) // blkk
                pinned = sum(b - a for a, b in block_ranges(keep, blkk, blocks))

            state["calls"] += 1
            state["seq"] = S
            state["kept"] = min(kept, blocks)
            state["blocks"] = blocks
            state["pinned"] = pinned

            if skip_output_reshape:
                return out.transpose(1, 2)
            return out.reshape(B, S, H * D)

        except Exception as exc:  # noqa: BLE001 - a bad kernel must not kill the run
            if state["failed"] is None:
                state["failed"] = "%s: %s" % (exc.__class__.__name__, exc)
                # Warning, not debug: a silent fall to dense is close enough to sparse speed
                # to go unnoticed from timing alone.
                logging.warning("[FunPack SLA] %s kernel failed, dense from here this run: %s",
                                cfg["engine"], state["failed"], exc_info=True)
            return dense()

    return override


def make_wrapper(state, cfg):
    """DIFFUSION_MODEL wrapper: per-step state, and the end-of-run summary.

    Registered once and then reused — ComfyUI caches node outputs, so this closure
    outlives a single sampling run. The step counter therefore has to reset itself, or
    every run after the first drifts permanently into the trailing-dense window and
    silently stops sparsifying.
    """
    dense_steps = cfg["dense_steps"]
    last = cfg["dense_last_steps"]

    def wrapper(executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        to = transformer_options
        new_run = lambda: (state.update(step=0, calls=0, dense=0, failed=None, order=None, last=None, summed=False),
                           state["history"].clear(), state["dense_at"].clear())
        counted = not _dit_hooks.probing(to)
        if to.get("sigmas") is not None and to.get("sample_sigmas") is not None:
            # The schedule says which step this is, so a cancelled run or a sampler calling the model twice
            # a step (heun, res_2s) cannot shift the dense steps. A call between scheduled sigmas (a
            # sampler's second evaluation) belongs to the step it is inside.
            where = _dit_hooks.current_step(to)
            n_steps = where[1] if where else max(1, len(to["sample_sigmas"]) - 1)
            if where and counted:
                idx = where[0]
                if state["last"] is not None and (idx < state["last"] or (state["summed"] and idx <= state["last"])):
                    new_run()
                state["last"], state["step"] = idx, idx + 1
        else:       # no schedule to read (a caller that is not ComfyUI's sampler): count calls
            n_steps = max(1, len(to.get("sample_sigmas", [])) - 1)
            # A throwaway call (a probe, late-branch's weakened copy) is not a step of the
            # schedule: counting it would slide the dense steps onto the wrong call.
            if counted and state["step"] >= n_steps:
                new_run()
            if counted:
                state["step"] += 1
        state["layer"] = 0

        # The layout lives on the payload, which never reaches the attention call site,
        # so the wrapper is the only place it can be picked up.
        video, keep, refs = spans(minimax_payload)
        to["_funpack_sla_spans"] = (video, tuple(keep), tuple(refs))
        to["_funpack_sla_dense"] = bool(
            (last > 0 and state["step"] > n_steps - last) or (state["step"] - 1) in dense_steps)
        state["n"] = n_steps
        if to["_funpack_sla_dense"] and counted:
            state["dense_at"].add(state["step"])

        # Forward minimax_payload only when H3 actually supplied one: every other
        # diffusion model would raise TypeError on the unexpected kwarg, turning a
        # graceful no-op into a crash mid-sampling.
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        try:
            # The executor itself, not executor.original: that skips every wrapper added after this one.
            out = executor(x, timestep, context, transformer_options=transformer_options, **kwargs)
        except Exception:
            state["history"].clear()            # an OOM must not leave the next attempt less memory
            raise

        if counted and state["step"] >= n_steps and not state["summed"]:
            _summarise(state, cfg)
            state["summed"] = True
            state["history"].clear()            # only needed step to step, not while the model sits cached
        return out

    return wrapper


def install_sla(model, sparsity_ratio=None, block_size=None, min_seq_len=None,
                dense_last_steps=None, protect_audio=None, enabled=None,
                dense_fn=None, dense_label=None, engine=None, dense_steps=None,
                references=None, tail=None, stabilize_motion=None, method=None, tau=None):
    """Give `model` block-sparse H3 attention. Returns (model, status line, installed).

    Weights are untouched: this installs an attention override and a per-step wrapper on
    a clone. A model that is not MiniMax H3 comes back unchanged and says so — the
    sparsity is only safe at the ratio H3's turbo LoRA was distilled to tolerate.

    `installed` is False whenever SLA did not take, so the caller can fall back to
    installing the chosen backend on its own rather than leaving the model with nothing.
    """
    given = dict(sparsity_ratio=sparsity_ratio, block_size=block_size, min_seq_len=min_seq_len,
                 dense_last_steps=dense_last_steps, protect_audio=protect_audio, enabled=enabled,
                 engine=engine, dense_steps=dense_steps, references=references, tail=tail,
                 stabilize_motion=stabilize_motion, method=method, tau=tau)
    cfg = {key: SLA_DEFAULTS[key] if value is None else value for key, value in given.items()}

    if not bool(cfg["enabled"]):
        # A dense baseline without touching the backend choice, so an A/B is one click
        # and the settings you were testing are still on the node when you switch back.
        return model, "SLA: off (dense baseline)", False
    if not is_h3_model(model):
        return model, ("SLA: skipped — not a MiniMax H3 model"), False
    triton_ok, ck_ok = sla_available(), ck_available()
    if not (triton_ok or ck_ok):
        return model, ("SLA: skipped — this machine has neither comfy_kitchen's sol_attn nor CUDA+Triton"), False

    notes = []
    if cfg["method"] not in METHODS:
        cfg["method"] = SLA_DEFAULTS["method"]
    if cfg["method"] == "vsa":
        return _install_vsa(model, cfg, dense_fn, dense_label)
    cfg.update(sparsity_ratio=float(cfg["sparsity_ratio"]), block_size=int(cfg["block_size"]),
               min_seq_len=int(cfg["min_seq_len"]), dense_last_steps=int(cfg["dense_last_steps"]),
               protect_audio=bool(cfg["protect_audio"]), tail=bool(cfg["tail"]),
               stabilize_motion=bool(cfg["stabilize_motion"]),
               references=cfg["references"] if cfg["references"] in REFERENCES else "off")
    if cfg["engine"] not in ENGINES:
        cfg["engine"] = SLA_DEFAULTS["engine"]
    if cfg["engine"] == "comfy_kitchen" and not ck_ok:
        cfg["engine"] = "triton"
        notes.append("comfy_kitchen's sol_attn is not here (needs comfy-kitchen >= 0.2.32): the Triton kernel runs")
    elif cfg["engine"] == "triton" and not triton_ok:
        cfg["engine"] = "comfy_kitchen"
        notes.append("no Triton here: comfy_kitchen's sol_attn runs")
    if cfg["method"] == "sol-attn" and cfg["engine"] != "comfy_kitchen":
        if ck_ok:
            cfg["engine"] = "comfy_kitchen"
            notes.append("sol-attn is comfy_kitchen's: that engine runs")
        else:
            cfg["method"] = "sla"
            notes.append("sol-attn needs comfy_kitchen's sol_attn (comfy-kitchen 0.2.32+), not here: sla runs instead")
    cfg["tau"] = float(cfg["tau"])
    cfg["dense_steps"], bad = parse_steps(cfg["dense_steps"])
    if bad:
        notes.append(f"dense_steps: ignored {', '.join(bad)} (steps count from 1: '1' is the first, '1-2' the first two)")
    if cfg["engine"] == "comfy_kitchen":
        ignored = [n for n, on in (("block_size", cfg["block_size"] != 64), ("stabilize_motion", cfg["stabilize_motion"])) if on]
        if ignored:
            notes.append(f"{', '.join(ignored)}: Triton engine only, ignored (sol_attn routes 64-token blocks, per step)")
    elif cfg["tail"]:
        notes.append("tail: comfy_kitchen engine only, ignored")

    state = new_state()
    patched = patching.clone(model)
    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = make_override(state, cfg, dense_fn=dense_fn, dense_label=dense_label)
    patched.model_options["transformer_options"] = to
    patched.add_wrapper_with_key("diffusion_model", "funpack_sla_state", make_wrapper(state, cfg))

    blk = cfg["block_size"] if cfg["engine"] == "triton" else 64
    how = f"tau={cfg['tau']:.2f}" if cfg["method"] == "sol-attn" else f"sparsity={cfg['sparsity_ratio']:.2f}"
    return patched, (f"SLA on | {cfg['method']} on {cfg['engine']} | {how} BLK={blk} "
                     f"min_seq_len={cfg['min_seq_len']} dense_last_steps={cfg['dense_last_steps']} "
                     f"full attention on steps {_runs(i + 1 for i in cfg['dense_steps']) or 'none'} and the last {cfg['dense_last_steps']} protect_audio={cfg['protect_audio']} "
                     f"references={cfg['references']} tail={cfg['tail']} stabilize_motion={cfg['stabilize_motion']} "
                     f"| dense calls -> {dense_label or 'as launched'}"
                     + "".join(f"\nSLA: {n}" for n in notes)), True


def _install_vsa(model, cfg, dense_fn, dense_label):
    """VSA through ComfyUI's own Model Sparse Attention (cube tiling and the trained coarse branch are
    its), the chosen backend underneath for every dense call. Its own sigma window keeps the first
    20% of the schedule dense; dense_steps and the span protection are this file's and do not apply."""
    try:
        from comfy_extras.nodes_sparse_attention import apply_block_sparse_attention
    except ImportError:
        return model, "SLA: vsa needs ComfyUI's Model Sparse Attention (ComfyUI 0.39 or later), not here", False
    if not ck_available():
        return model, "SLA: vsa needs comfy_kitchen's sol_attn (comfy-kitchen 0.2.32+), not here", False
    base = patching.clone(model)
    if dense_fn is not None:        # Model Sparse Attention hands what it does not take to the override it found
        to = base.model_options.get("transformer_options", {}).copy()
        to["optimized_attention_override"] = dense_fn
        base.model_options["transformer_options"] = to
    keep = 1.0 - float(cfg["sparsity_ratio"])
    try:
        patched = apply_block_sparse_attention(
            base, tau=float(cfg["tau"]), topk_ratio=keep, vsa=True, start_percent=0.2, end_percent=1.0,
            min_tokens=int(cfg["min_seq_len"]), dense_blocks=set(), sink_conditioning="exact_kv_and_rows",
            extra_tokens=0, verbose=False)
    except (TypeError, ValueError) as exc:      # another ComfyUI's version of it, or a model it refuses
        return model, f"SLA: vsa could not be set up by this ComfyUI's Model Sparse Attention: {exc}", False
    notes = [f"SLA on | vsa (ComfyUI's Model Sparse Attention) | keeps {keep:.0%} of video cubes, first 20% of "
             f"the schedule dense, min_seq_len={int(cfg['min_seq_len'])} | dense calls -> {dense_label or 'as launched'}"]
    try:
        trained = getattr(patched.get_model_object("diffusion_model").blocks[0].attn, "to_gate_compress", None) is not None
    except Exception:  # noqa: BLE001 - a description must never fail the load
        trained = True
    if not trained:
        notes.append("SLA: this model has no VSA layers (to_gate_compress): it was not trained for VSA, so it runs "
                     "without VSA's coarse branch and quality may drop. sla or sol-attn suits it better.")
    ignored = [n for n, on in (("dense_steps", cfg["dense_steps"].strip() not in ("", "1")), ("references", cfg["references"] != "off"),
                               ("stabilize_motion", cfg["stabilize_motion"]), ("tail", cfg["tail"])) if on]
    if ignored:
        notes.append(f"SLA: {', '.join(ignored)}: not used by vsa")
    return patched, "\n".join(notes), True
