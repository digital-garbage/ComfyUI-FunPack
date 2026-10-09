"""SLA block-sparse attention: the ComfyUI override contract, and the H3 gate.

The contract half matters more than it looks. The failure this guards against is
not a crash -- it is a patch that installs cleanly, logs success and never runs,
because it hooked an API the model does not consult. So these drive the override
exactly the way `wrap_attn` does, with H3's real argument shape, and assert which
path fired.

Ported from FunPack v4's own suite, itself ported from ComfyUI-H3-SLA-Attention
(MIT). The CUDA half is skipped without a GPU; everything else runs anywhere
torch imports.
"""
import types

import pytest
import torch

from modules.loaders import sla_attention as sla

H, D = 56, 128          # MiniMax H3: 56 heads, head_dim 128

CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False


def _backend(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
             skip_output_reshape=False, **kwargs):
    """Stand-in for ComfyUI's undecorated attention backend."""
    if not skip_reshape:
        b, s, _ = q.shape
        q, k, v = (t.view(b, s, heads, -1).transpose(1, 2) for t in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if skip_output_reshape:
        return o
    b, _, s, _ = q.shape
    return o.transpose(1, 2).reshape(b, s, -1)


def _cfg(**kw):
    """install_sla's resolved settings, on the Triton engine (the one that runs without comfy_kitchen)."""
    cfg = {**sla.SLA_DEFAULTS, "engine": "triton", "dense_steps": frozenset()}
    cfg.update(kw)
    return cfg


def _call(override, q, k, v, **kw):
    # wrap_attn hands the override the UNDECORATED backend as arg 0, then q/k/v/heads
    # with H3's kwargs: mask is always None, skip_reshape True, and skip_output_reshape
    # is not passed at all.
    opts = dict(mask=None, skip_reshape=True, transformer_options={},
                _inside_attn_wrapper=True)
    opts.update(kw)
    return override(_backend, q, k, v, H, **opts)


# ── the override contract ─────────────────────────────────────────────────────

def test_a_short_sequence_stays_dense():
    """H3's text refiner is a few hundred tokens and must never be sparsified."""
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=8192))
    q = torch.randn(1, H, 512, D)
    out = _call(ov, q, q.clone(), q.clone())
    assert (state["calls"], state["dense"]) == (0, 1)
    assert out.shape == (1, 512, H * D)


def test_masked_attention_stays_dense():
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=0))
    q = torch.randn(1, H, 256, D)
    _call(ov, q, q.clone(), q.clone(), mask=torch.zeros(1, 1))
    assert state["calls"] == 0


def test_float32_never_reaches_the_kernel():
    """The dtype guard catches it first, so nothing is recorded as a kernel failure."""
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=0))
    q = torch.randn(1, H, 256, D)
    out = _call(ov, q, q.clone(), q.clone())
    assert (state["calls"], state["dense"]) == (0, 1)
    assert state["failed"] is None
    assert out.shape == (1, 256, H * D)


def test_the_run_records_which_backend_it_displaced():
    """So the log can say what dense fall-throughs will actually use."""
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=8192))
    _call(ov, *(torch.randn(1, H, 256, D),) * 3)
    assert state["backend"] == "_backend"


def test_a_kernel_failure_costs_speed_not_the_run():
    """bf16 on CPU passes every guard and then cannot launch a CUDA kernel -- the
    closest stand-in for a driver or Triton mismatch on a real machine."""
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=0))
    q = torch.randn(1, H, 256, D, dtype=torch.bfloat16)
    out = _call(ov, q, q.clone(), q.clone())
    assert out.shape == (1, 256, H * D)
    assert state["failed"] is not None
    assert (state["calls"], state["dense"]) == (0, 1)


# ── the per-step wrapper ──────────────────────────────────────────────────────

def _run(wrapper, n_steps, payload=None):
    """One sampling run: n_steps forwards through the wrapper. Returns the per-step
    value of the force-dense flag."""
    seen = []

    def Ex(*a, **kw):
        seen.append(kw["transformer_options"]["_funpack_sla_dense"])
        return None

    to = {"sample_sigmas": [0.0] * (n_steps + 1)}
    for _ in range(n_steps):
        wrapper(Ex, None, None, None, transformer_options=to, minimax_payload=payload)
    return seen


def test_the_step_counter_resets_between_runs():
    """ComfyUI caches node outputs, so this closure outlives one run. Without the reset
    every later run sits permanently inside the trailing-dense window and silently stops
    sparsifying."""
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=1))
    first = _run(w, 4)
    assert first == [False, False, False, True]
    assert _run(w, 4) == first


def test_dense_last_steps_zero_never_forces_dense():
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=0))
    assert _run(w, 4) == [False] * 4


def test_the_protected_spans_are_read_from_the_packed_layout():
    """Language and every audio stream stay exact; vision tokens and visual references do not.
    The layout lives on minimax_payload, which never reaches the attention call site, so the
    wrapper is the only place it can be picked up."""
    layout = types.SimpleNamespace(segments=[
        (0, 10, "text"), (10, 300, "cond"), (300, 400, "cond_audio"), (400, 500, "ref_img"),
        (500, 520, "ref_audio"), (520, 800, "audio"), (800, 9000, "video")])
    tags = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])        # language, vision pads, language
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=0))
    to = {"sample_sigmas": [0.0] * 5}
    w(lambda *a, **kw: None, None, None, None, transformer_options=to,
      minimax_payload={"layout": layout, "text_token_tags": tags})
    video, keep, refs = to["_funpack_sla_spans"]
    assert video == 800
    assert keep == ((0, 3), (6, 10), (300, 400), (500, 520), (520, 800))
    assert refs == ((3, 6), (10, 300), (400, 500))


def test_untagged_text_is_all_language_and_no_layout_protects_nothing():
    layout = types.SimpleNamespace(segments=[(0, 10, "text"), (10, 50, "video")])
    assert sla.spans({"layout": layout}) == (10, [(0, 10)], [])
    assert sla.spans({"layout": layout, "text_token_tags": [1, 0]}) == (10, [(0, 10)], [])   # wrong length
    assert sla.spans(None) == (0, [], [])


def test_dense_steps_are_0_based_and_stack_with_the_last_steps():
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=1, dense_steps=frozenset({0, 2})))
    assert _run(w, 5) == [True, False, True, False, True]
    assert sla.parse_steps("0, 2-4,x,6-5") == (frozenset({0, 2, 3, 4, 5, 6}), ["x"])


def test_the_wrapper_calls_the_next_wrapper_not_the_bare_model():
    """executor.original skips every wrapper registered after SLA's -- Shadow Negative's
    per-call hook among them, which silently did nothing with SLA on."""
    from comfy.patcher_extension import WrapperExecutor
    seen = []

    def later(executor, *a, **kw):
        seen.append("later")
        return executor(*a, **kw)

    state = sla.new_state()
    chain = [sla.make_wrapper(state, _cfg()), later]
    ex = WrapperExecutor.new_executor(lambda *a, **kw: seen.append("model"), chain)
    ex.execute(None, None, None, transformer_options={"sample_sigmas": [0.0] * 3})
    assert seen == ["later", "model"]


def test_a_non_h3_model_is_not_handed_a_minimax_kwarg():
    """Every other diffusion model would raise TypeError on the unexpected kwarg -- a
    crash mid-sampling instead of the graceful no-op it should be."""
    seen = {}

    def Ex(x, timestep, context, transformer_options=None, **kw):
        seen.update(kw)
        return None

    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=0))
    w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 2})
    assert "minimax_payload" not in seen


# ── the H3 gate ───────────────────────────────────────────────────────────────

class _Patcher:
    def __init__(self, class_name, module="comfy.ldm.minimax.model"):
        # has_block (core/traits.py) matches a QUALIFIED class name -- module and
        # name both -- so the stub needs a real __module__ and a modules() walk
        # the way an nn.Module provides, not just a bare type() with a name.
        cls = type(class_name, (), {"__module__": module, "modules": lambda self: [self]})
        inner = cls()
        self.model = types.SimpleNamespace(diffusion_model=inner)
        self.model_options = {}
        self.wrappers = []

    def clone(self):
        out = _Patcher("x")
        out.model, out.model_options = self.model, dict(self.model_options)
        return out

    def add_wrapper_with_key(self, *a):
        self.wrappers.append(a)


def test_sla_refuses_a_model_that_is_not_h3_and_says_so():
    """Head shape alone matches LTX too, and sparsifying LTX attention is a quality loss
    with no LoRA compensating for it. Silence would read as "it is on"."""
    model = _Patcher("LTXVModel")
    out, note, installed = sla.install_sla(model)
    assert out is model and installed is False
    assert "not a MiniMax H3 model" in note


def test_a_same_named_class_from_a_different_module_is_not_treated_as_h3():
    """A bare class-name match would treat any unrelated MiniMaxH3Model -- built by a
    different custom node, or a future FunPack model reusing the name by accident --
    as the real thing, and sparsify it at H3's ratio with H3's prefix-pinning logic
    on a model that was never validated for either. The qualified check requires the
    DEFINING module too, not just the name."""
    model = _Patcher("MiniMaxH3Model", module="some_other_package.model")
    out, note, installed = sla.install_sla(model)
    assert out is model and installed is False
    assert "not a MiniMax H3 model" in note


def test_sla_reports_when_the_machine_cannot_run_it():
    model = _Patcher("MiniMaxH3Model")
    out, note, installed = sla.install_sla(model)
    if sla.sla_available():
        assert installed is True and out is not model and "sparsity=0.90" in note
    else:
        assert installed is False and out is model and "CUDA+Triton" in note


def test_the_defaults_are_the_validated_ones():
    """0.90 / 64 / protect on: the settings the port was measured at."""
    assert sla.SLA_DEFAULTS == {"sparsity_ratio": 0.90, "block_size": 64,
                                "min_seq_len": 8192, "dense_last_steps": 0,
                                "protect_audio": True, "enabled": True,
                                "engine": "comfy_kitchen", "dense_steps": "0", "method": "sla", "tau": 1.3,
                                "references": "off",
                                "tail": False, "stabilize_motion": False}


def test_turning_sla_off_leaves_the_settings_in_place():
    """A dense A/B baseline must not cost the settings being tested."""
    model = _Patcher("MiniMaxH3Model")
    out, note, installed = sla.install_sla(model, sparsity_ratio=0.85, enabled=False)
    assert out is model and installed is False
    assert "off (dense baseline)" in note


# ── composing with a chosen backend ───────────────────────────────────────────

def test_dense_calls_go_to_the_chosen_backend_not_the_launched_one():
    """There is one override slot. Without this, choosing SLA would silently discard the
    backend the user picked and drop the text refiner onto whatever ComfyUI launched with."""
    seen = []

    def chosen(func, q, k, v, heads, **kw):
        seen.append("chosen")
        return func(q, k, v, heads, **kw)

    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=8192),
                           dense_fn=chosen, dense_label="sage3")
    q = torch.randn(1, H, 512, D)              # short -> dense fall-through
    out = _call(ov, q, q.clone(), q.clone())
    assert seen == ["chosen"]
    assert state["backend"] == "sage3"         # what the log will name
    assert out.shape == (1, 512, H * D)


def test_without_a_chosen_backend_dense_still_runs():
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=8192))
    q = torch.randn(1, H, 512, D)
    assert _call(ov, q, q.clone(), q.clone()).shape == (1, 512, H * D)
    assert state["dense"] == 1


# ── the kernel itself ─────────────────────────────────────────────────────────

pytestmark_cuda = pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + Triton")


@pytestmark_cuda
def test_keeping_every_block_is_just_attention():
    from modules.loaders.sla_block_map import get_block_map
    from modules.loaders.sla_kernel import block_sparse_attention

    S = 4096
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    lut, topk, _ = get_block_map(q, k, 1.0, 64, 64)
    got = block_sparse_attention(q, k, v, lut, topk, 64, 64)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
    assert not torch.isnan(got).any()
    rel = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    assert rel < 1e-2


@pytestmark_cuda
def test_every_query_block_keeps_the_whole_protected_prefix():
    """This is the audio fix: audio is ~1% of the packed sequence, so plain top-k
    routinely drops all of it and the soundtrack degrades while the video looks fine."""
    from modules.loaders.sla_block_map import get_block_map

    S, prefix = 16384, 2048
    torch.manual_seed(0)
    q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
    plain_lut, plain_topk, _ = get_block_map(q, k, 0.10, 64, 64)
    lut, topk, _ = get_block_map(q, k, 0.10, 64, 64, protect=[(0, prefix)])

    n_pinned = prefix // 64
    assert topk == plain_topk + n_pinned          # widened, so it displaces no video
    got = lut.long().sort(dim=-1).values[..., :n_pinned]
    want = torch.arange(n_pinned, device=lut.device)
    assert torch.equal(got, want.expand_as(got))
    # and the failure it exists to prevent
    covered = (plain_lut.long() < n_pinned).sum(-1).float().mean().item()
    assert covered < n_pinned


@pytestmark_cuda
def test_the_override_fires_and_returns_h3s_shape():
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(min_seq_len=1024))
    S = 8192
    q, k, v = (torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = _call(ov, q, k, v)
    assert (state["calls"], state["dense"]) == (1, 0)
    assert out.shape == (1, S, H * D)
    assert not torch.isnan(out).any()


def test_a_throwaway_call_is_not_a_step_so_the_dense_window_stays_on_the_real_last_step():
    """Late-branch guidance's weakened copy (and a seed-search probe) call the model between
    real steps. Counting them slid the dense window onto the wrong call and left the real
    final step sparse."""
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_last_steps=1))
    real, weak = [], []

    def Ex(*a, **kw):
        (weak if kw["transformer_options"].get("funpack_probe") else real).append(
            kw["transformer_options"]["_funpack_sla_dense"])
        return None

    for i in range(4):
        w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 5}, minimax_payload=None)
        if i < 3:
            w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 5, "funpack_probe": True},
              minimax_payload=None)
    assert real == [False, False, False, True]


def test_block_ranges_merge_and_round_outward():
    from modules.loaders.sla_attention import block_ranges, minus
    assert block_ranges([(0, 3), (6, 10), (300, 400), (130, 100)], 64, 100) == [(0, 1), (4, 7)]
    assert block_ranges([(0, 10_000)], 64, 5) == [(0, 5)]
    assert minus([(0, 10)], [(2, 4), (8, 20)]) == [(0, 2), (4, 8)]


@pytestmark_cuda
def test_light_references_keep_a_share_and_stabilize_remembers_only_the_edge():
    from modules.loaders.sla_block_map import get_block_map
    torch.manual_seed(0)
    S = 8192
    q, k = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    plain_lut, plain_topk, _ = get_block_map(q, k, 0.10, 64, 64)
    lut, topk, hist = get_block_map(q, k, 0.10, 64, 64, refs=[(0, 64 * 20)], ref_keep=0.15,
                                    sticky_from=64 * 20, remember=True)
    assert topk == plain_topk + 3                                   # ceil(0.15 * 20) on top of the budget
    assert ((lut.long() < 20).sum(-1) >= 3).all()
    assert hist.shape == (1, H, S // 64 - 20, 8)
    again, _, _ = get_block_map(q, k, 0.10, 64, 64, prev=hist, sticky_from=64 * 20)
    assert again.shape == plain_lut.shape


def test_install_reports_the_engine_and_what_it_ignores(monkeypatch):
    monkeypatch.setattr(sla, "sla_available", lambda: True)
    monkeypatch.setattr(sla, "ck_available", lambda: False)
    out, note, installed = sla.install_sla(_Patcher("MiniMaxH3Model"), dense_steps="0,x", tail=True)
    assert installed and "| sla on triton |" in note
    assert "sol_attn is not here" in note and "ignored x" in note and "tail: comfy_kitchen engine only" in note
    monkeypatch.setattr(sla, "ck_available", lambda: True)
    _, note, _ = sla.install_sla(_Patcher("MiniMaxH3Model"), block_size=32, stabilize_motion=True)
    assert "| sla on comfy_kitchen |" in note and "block_size, stabilize_motion: Triton engine only" in note


def test_the_comfy_kitchen_engine_puts_scattered_protected_blocks_first(monkeypatch):
    """sol_attn keeps ONE key-block range exact; attention ignores key order, so K/V are reordered."""
    import sys
    calls = []

    def sol_attn(q, k, v, scale=None, sink_blocks=None, sink_q=None, topk_ratio=0.0, tail=True):
        calls.append((k[0, :, 0, 0].tolist(), sink_blocks, round(topk_ratio, 2), tail))
        return q
    monkeypatch.setitem(sys.modules, "comfy_kitchen", types.SimpleNamespace(sol_attn=sol_attn))
    S = 64 * 6
    q = torch.zeros(1, H, S, D, dtype=torch.bfloat16)
    k = torch.arange(S, dtype=torch.bfloat16).repeat_interleave(H * D).view(1, S, H, D).transpose(1, 2)
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(engine="comfy_kitchen", min_seq_len=0))
    for keep in (((0, 64), (256, 320)), ((64, 192),)):
        _call(ov, q, k, k.clone(), transformer_options={"_funpack_sla_spans": (320, keep, ())})
    first, second = calls
    assert first[1] == [0, 2] and first[0][:128:64] == [0.0, 256.0] and first[2:] == (0.1, False)
    assert second[1] == [1, 3] and second[0][64] == 64.0             # one run: used where it is, no copy
    assert state["calls"] == 2 and state["failed"] is None


def test_dense_steps_follow_the_schedule_through_a_cancel_and_a_two_call_sampler():
    """Counting calls drifted for good after a cancelled render, and a sampler calling the
    model twice a step ended the 'run' halfway."""
    sched = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    state = sla.new_state()
    w = sla.make_wrapper(state, _cfg(dense_steps=frozenset({0, 1})))
    seen = []
    ex = lambda *a, **kw: seen.append(kw["transformer_options"]["_funpack_sla_dense"])
    call = lambda s: w(ex, None, None, None, transformer_options={"sample_sigmas": sched, "sigmas": torch.tensor([s])})
    for s in (1.0, 0.8, 0.6):           # cancelled after three steps
        call(s)
    seen.clear()
    for s in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):     # a full run, two calls a step
        call(s)
    assert seen == [True, True, True, True, False, False, False, False, False, False]


def test_sol_attn_routes_by_threshold_and_vsa_hands_over_to_comfyui(monkeypatch):
    import sys
    calls = []
    monkeypatch.setitem(sys.modules, "comfy_kitchen", types.SimpleNamespace(
        sol_attn=lambda q, k, v, **kw: calls.append(kw) or q))
    state = sla.new_state()
    ov = sla.make_override(state, _cfg(engine="comfy_kitchen", method="sol-attn", tau=1.7, min_seq_len=0))
    q = torch.zeros(1, H, 128, D, dtype=torch.bfloat16)
    _call(ov, q, q.clone(), q.clone(), transformer_options={"_funpack_sla_spans": (64, ((0, 64),), ())})
    assert calls[0]["tau"] == 1.7 and "topk_ratio" not in calls[0]

    monkeypatch.setattr(sla, "sla_available", lambda: True)
    monkeypatch.setattr(sla, "ck_available", lambda: False)
    _, note, _ = sla.install_sla(_Patcher("MiniMaxH3Model"), method="sol-attn")
    assert "sla runs instead" in note and "| sla on triton |" in note

    monkeypatch.setattr(sla, "ck_available", lambda: True)
    got = {}

    def apply(model, **kw):
        got.update(kw, override=model.model_options["transformer_options"].get("optimized_attention_override"))
        return model
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sparse_attention",
                        types.SimpleNamespace(apply_block_sparse_attention=apply))
    patcher = _Patcher("MiniMaxH3Model")
    patcher.get_model_object = lambda name: types.SimpleNamespace(
        blocks=[types.SimpleNamespace(attn=types.SimpleNamespace(to_gate_compress=None))])
    patcher.clone = lambda: patcher
    backend = lambda func, *a, **kw: func(*a, **kw)
    _, note, installed = sla.install_sla(patcher, method="vsa", sparsity_ratio=0.9, dense_fn=backend)
    assert installed and got["vsa"] is True and abs(got["topk_ratio"] - 0.1) < 1e-9
    assert got["override"] is backend                 # dense calls still go to the chosen backend
    assert "no VSA layers" in note
