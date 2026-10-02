"""Block hooks and attention overrides that stack, come off again, and cannot end a run.

ComfyUI gives a transformer two extension points that are NOT lists:

* `transformer_options["patches_replace"]["dit"][("double_block", i)]` -- one
  callable per block, which replaces that block's call.
* `transformer_options["optimized_attention_override"]` -- one callable for every
  attention call in the model.

One slot each. A second modifier that simply assigns the slot silently deletes the
first one's effect, and v4 learned that the slow way (temperature vanishing under
REINS, with no error). So everything here CHAINS: a hook is written as if it were
alone -- it calls `extra["original_block"](args)`, or `func(q, k, v, ...)` -- and
what it calls is really the rest of the chain. No hook ever needs to know another
exists.

Model-agnostic on purpose. The two slot names are ComfyUI conventions shared by
several DiTs; which models actually honour them is a trait the MODEL module
announces (`dit_block_hooks`), never something this file decides.

Removal: every installed callable carries `patching.TAG` and a pointer to what it
wrapped, so `patching.strip` unwinds ours and puts back whatever was underneath --
including someone else's hook we chained through.
"""

from . import patching

PREV = "_funpack_prev"
BLOCK = "double_block"


def _transformer_options(patcher) -> dict:
    options = patcher.model_options
    to = options.get("transformer_options")
    if not isinstance(to, dict):
        to = {}
        options["transformer_options"] = to
    return to


def _guard(patcher, fn, neutral):
    """Guarded when the patcher can guard (it is always a GuardedPatcher inside a
    run); a bare patcher -- tests, a direct call -- gets the hook unchanged."""
    wrap = getattr(patcher, "guarded", None)
    return wrap(fn, neutral) if callable(wrap) else fn


def _mark(fn, key, prev):
    patching.tag(fn, key)
    setattr(fn, PREV, prev)
    return fn


def add_block_hook(patcher, key: str, index: int, hook) -> None:
    """Put `hook(args, extra)` around block `index`, on top of whatever is there.

    `extra["original_block"]` inside the hook runs the rest of the chain, ending in
    the real block. A failing hook is dropped for the rest of the run and the block
    runs as though it had never been installed.
    """
    to = _transformer_options(patcher)
    replace = to.setdefault("patches_replace", {})
    dit = replace.setdefault("dit", {})
    slot = (BLOCK, int(index))
    inner = dit.get(slot)

    def below(extra):
        # `funpack_below` tells a hook that REPLACES the block (rather than
        # calling original_block) that doing so would skip someone's hook.
        if inner is None:
            return {**extra, "funpack_below": False}
        original = extra["original_block"]
        return {**extra, "funpack_below": True,
                "original_block": lambda a: inner(a, {**extra, "original_block": original})}

    def neutral(args, extra):
        return below(extra)["original_block"](args)

    guarded = _guard(patcher, lambda args, extra: hook(args, below(extra)), neutral)

    def chained(args, extra):
        return guarded(args, extra)

    dit[slot] = _mark(chained, key, inner)


def add_attention_override(patcher, key: str, override) -> None:
    """Put `override(func, q, k, v, heads, **kw)` around every attention call.

    Calling `func(...)` inside it runs the rest of the chain, ending in the real
    backend. The override sees plain tensors: ComfyUI unwraps its tensor
    containers before an override is called.
    """
    to = _transformer_options(patcher)
    inner = to.get("optimized_attention_override")

    def rest(func):
        if inner is None:
            return func
        return lambda *a, **kw: inner(func, *a, **kw)

    def neutral(func, *args, **kwargs):
        return rest(func)(*args, **kwargs)

    guarded = _guard(patcher, lambda func, *a, **kw: override(rest(func), *a, **kw), neutral)

    def chained(func, *args, **kwargs):
        return guarded(func, *args, **kwargs)

    to["optimized_attention_override"] = _mark(chained, key, inner)


def block_count(model) -> int:
    """How many hookable blocks the model has, or 0 when it cannot be read."""
    root = getattr(getattr(model, "model", None), "diffusion_model", None)
    blocks = getattr(root, "blocks", None)
    try:
        return len(blocks)
    except TypeError:
        return 0


def parse_blocks(spec, n_blocks: int):
    """"40" | "38-42" | "10,40,44" | "" -> (sorted indices, [problems]).

    Out-of-range and unparseable entries are reported, not silently dropped: a
    typo that turns a setting into a no-op is exactly the inert knob this project
    refuses to ship, so the caller says what was ignored.
    """
    out, problems = set(), []
    for part in str(spec or "").replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part[1:]:
                a, b = part.split("-", 1)
                lo, hi = sorted((int(a), int(b)))
            else:
                lo = hi = int(part)
        except ValueError:
            problems.append(f"{part!r} is not a block number or range")
            continue
        wanted = set(range(lo, hi + 1))
        kept = {i for i in wanted if 0 <= i < n_blocks}
        if kept != wanted:
            problems.append(f"{part!r} reaches past this model's blocks 0-{n_blocks - 1}")
        out |= kept
    return sorted(out), problems


def current_step(transformer_options):
    """(step index, step count) of the denoise call in progress, or None.

    Core puts the whole schedule in `sample_sigmas` and this call's sigma in
    `sigmas`; the index is exact, so a step window never has to guess a sigma
    threshold per schedule.
    """
    import torch
    to = transformer_options or {}
    sched, cur = to.get("sample_sigmas"), to.get("sigmas")
    if sched is None or cur is None:
        return None
    cur = cur.reshape(-1)[0].to(sched.dtype)
    hit = torch.nonzero(torch.isclose(sched, cur, rtol=1e-4))
    if not len(hit):
        return None
    return int(hit[0]), int(sched.shape[0]) - 1


PROBE = "funpack_probe"
# Set on late-branch guidance's weakened copy (which also carries PROBE). Hooks that must not
# run on it, or must tell it from a probe, read this.
WEAK_BRANCH = "funpack_weak_branch"


def weak_branch(transformer_options) -> bool:
    return bool((transformer_options or {}).get(WEAK_BRANCH))


def probing(transformer_options) -> bool:
    """A throwaway model call (a candidate scored and discarded, e.g. the
    first-step seed search). Nothing may be learned from one: a feature that
    captures from it banks a step the clip never took."""
    return bool((transformer_options or {}).get(PROBE))


def last_step(transformer_options) -> bool:
    """Whether this denoise call is the schedule's final step -- True when that
    cannot be told, so a capture taken "on the last step" still happens. Never
    during a probe."""
    if probing(transformer_options):
        return False
    where = current_step(transformer_options)
    return where is None or where[0] >= where[1] - 1


def late_half(transformer_options, ahead: int = 0) -> float:
    """0 over the first half of the steps, rising to 1 at the last: the gate
    every learning feature that steers late shares. By step POSITION, so it
    holds on any schedule -- v4's sigma-based gate was open on 0 of 12 steps
    of H3's default schedule, silently. 0 when the step cannot be told.

    `ahead` reads a LATER step's gate: an edit carried into step i+1's input
    (core/input_steer.py) is gated by the step it lands on."""
    where = current_step(transformer_options)
    if where is None:
        return 0.0
    index, total = where
    return max(0.0, 2.0 * (index + ahead) / max(1, total) - 1.0)


def row_span(mask):
    """A [S] bool mask of ONE contiguous run -> (start, stop), or None.

    Slicing a span is a view; indexing with a mask copies every row, which on
    a 37k-token video is hundreds of MB per call. One host sync, so callers
    cache the answer per sequence length.
    """
    if mask is None:
        return None
    hit = mask.nonzero()
    if not len(hit):
        return None
    lo, hi = int(hit[0]), int(hit[-1]) + 1
    return (lo, hi) if hi - lo == len(hit) else None


def target_rows(mod_segments, seq_len, device, stream):
    """[seq_len] bool mask of the model's TARGET `stream` ("video" | "audio")
    rows, or None when no model module can prove it for this call.

    Asked of whichever model module announces `target_rows`, so core never
    learns a model's sequence layout; the first provider with an answer wins.
    """
    from . import registry
    for _spec, provider in registry.current().providers("target_rows"):
        try:
            mask = provider(mod_segments, seq_len, device, stream)
        except Exception:                        # noqa: BLE001
            continue
        if mask is not None:
            return mask
    return None


def without_compiler(patcher, key: str) -> None:
    """Run this model's sampling with ComfyUI's model compiler off.

    The compiler (aimdo malloc-graph) records what each block allocates and
    assumes it never changes. A hook that REPLACES a block's forward with one
    allocating differently -- a repeated block, a second attention stream --
    crashes it with "aimdo memory compile error" (v4, on rentals: block repeat,
    REINS injection, shadow negative). Comfy-Org's own fix is
    --disable-comfy-compiler; it is read live on every scope, so switching it
    for the one sampling call is enough and the next run keeps the compiler.

    Attached to the MODEL as an OUTER_SAMPLE wrapper, so it holds for any
    sampler, not only FunPack's.
    """
    from comfy.patcher_extension import WrappersMP

    def outer(executor, *args, **kwargs):
        import comfy.cli_args
        cli = comfy.cli_args.args
        had = hasattr(cli, "disable_comfy_compiler")
        prior = getattr(cli, "disable_comfy_compiler", None)
        cli.disable_comfy_compiler = True
        try:
            return executor(*args, **kwargs)
        finally:
            if had:
                cli.disable_comfy_compiler = prior
            else:
                try:
                    del cli.disable_comfy_compiler
                except AttributeError:
                    pass

    patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)
