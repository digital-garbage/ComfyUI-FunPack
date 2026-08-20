"""SLA block-sparse attention: the ComfyUI override contract, and the H3 gate.

The contract half matters more than it looks. The failure this guards against is not a
crash — it is a patch that installs cleanly, logs success and never runs, because it
hooked an API the model does not consult. So these drive the override exactly the way
``wrap_attn`` does, with H3's real argument shape, and assert which path fired.

Ported from ComfyUI-H3-SLA-Attention's own suite (MIT). The CUDA half is skipped
without a GPU; everything else runs anywhere torch imports.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in ("comfy", "comfy.sd", "comfy.utils", "folder_paths"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

import sla_attention as sla  # noqa: E402

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
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=8192)
    q = torch.randn(1, H, 512, D)
    out = _call(ov, q, q.clone(), q.clone())
    assert (state["calls"], state["dense"]) == (0, 1)
    assert out.shape == (1, 512, H * D)


def test_masked_attention_stays_dense():
    state = sla.new_state()
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=0)
    q = torch.randn(1, H, 256, D)
    _call(ov, q, q.clone(), q.clone(), mask=torch.zeros(1, 1))
    assert state["calls"] == 0


def test_float32_never_reaches_the_kernel():
    """The dtype guard catches it first, so nothing is recorded as a kernel failure."""
    state = sla.new_state()
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=0)
    q = torch.randn(1, H, 256, D)
    out = _call(ov, q, q.clone(), q.clone())
    assert (state["calls"], state["dense"]) == (0, 1)
    assert state["failed"] is None
    assert out.shape == (1, 256, H * D)


def test_the_run_records_which_backend_it_displaced():
    """So the log can say what dense fall-throughs will actually use."""
    state = sla.new_state()
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=8192)
    _call(ov, *(torch.randn(1, H, 256, D),) * 3)
    assert state["backend"] == "_backend"


def test_a_kernel_failure_costs_speed_not_the_run():
    """bf16 on CPU passes every guard and then cannot launch a CUDA kernel — the
    closest stand-in for a driver or Triton mismatch on a real machine."""
    state = sla.new_state()
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=0)
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

    class Ex:
        @staticmethod
        def original(*a, **kw):
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
    w = sla.make_wrapper(state, 0.90, 64, 64, dense_last_steps=1)
    first = _run(w, 4)
    assert first == [False, False, False, True]
    assert _run(w, 4) == first


def test_dense_last_steps_zero_never_forces_dense():
    state = sla.new_state()
    w = sla.make_wrapper(state, 0.90, 64, 64, dense_last_steps=0)
    assert _run(w, 4) == [False] * 4


def test_the_protected_prefix_is_read_from_the_packed_layout():
    """The video segment's start is the length of what must stay exactly attended. It
    lives on minimax_payload, which never reaches the attention call site, so the
    wrapper is the only place it can be picked up."""
    layout = types.SimpleNamespace(segments=[
        (0, 512, "text"), (512, 800, "cond"), (800, 2000, "audio"), (2000, 114785, "video")])
    state = sla.new_state()
    w = sla.make_wrapper(state, 0.90, 64, 64, dense_last_steps=0)
    to = {"sample_sigmas": [0.0] * 5}

    class Ex:
        @staticmethod
        def original(*a, **kw):
            return None

    w(Ex, None, None, None, transformer_options=to, minimax_payload={"layout": layout})
    assert to["_funpack_sla_prefix"] == 2000


def test_no_layout_disables_the_protection_rather_than_guessing():
    state = sla.new_state()
    w = sla.make_wrapper(state, 0.90, 64, 64, dense_last_steps=0)
    to = {"sample_sigmas": [0.0] * 5}

    class Ex:
        @staticmethod
        def original(*a, **kw):
            return None

    w(Ex, None, None, None, transformer_options=to, minimax_payload=None)
    assert to["_funpack_sla_prefix"] == 0


def test_a_non_h3_model_is_not_handed_a_minimax_kwarg():
    """Every other diffusion model would raise TypeError on the unexpected kwarg — a
    crash mid-sampling instead of the graceful no-op it should be."""
    seen = {}

    class Ex:
        @staticmethod
        def original(x, timestep, context, transformer_options=None, **kw):
            seen.update(kw)
            return None

    state = sla.new_state()
    w = sla.make_wrapper(state, 0.90, 64, 64, dense_last_steps=0)
    w(Ex, None, None, None, transformer_options={"sample_sigmas": [0.0] * 2})
    assert "minimax_payload" not in seen


# ── the H3 gate ───────────────────────────────────────────────────────────────

class _Patcher:
    def __init__(self, class_name):
        inner = type(class_name, (), {})()
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
    out, note = sla.install_sla(model)
    assert out is model
    assert "not a MiniMax H3 model" in note


def test_sla_reports_when_the_machine_cannot_run_it():
    model = _Patcher("MiniMaxH3Model")
    out, note = sla.install_sla(model)
    if sla.sla_available():
        assert out is not model and "sparsity=0.90" in note
    else:
        assert out is model and "CUDA+Triton" in note


def test_the_defaults_are_the_validated_ones():
    """0.90 / 64 / protect on: the settings the port was measured at."""
    assert sla.SLA_DEFAULTS == {"sparsity_ratio": 0.90, "block_size": 64,
                                "min_seq_len": 8192, "dense_last_steps": 0,
                                "protect_audio": True, "enabled": True}


def test_turning_sla_off_leaves_the_settings_in_place():
    """A dense A/B baseline must not cost the settings being tested."""
    model = _Patcher("MiniMaxH3Model")
    out, note = sla.install_sla(model, sparsity_ratio=0.85, enabled=False)
    assert out is model
    assert "off (dense baseline)" in note


# ── the kernel itself ─────────────────────────────────────────────────────────

pytestmark_cuda = pytest.mark.skipif(not (CUDA and HAS_TRITON), reason="needs CUDA + Triton")


@pytestmark_cuda
def test_keeping_every_block_is_just_attention():
    from sla_block_map import get_block_map
    from sla_kernel import block_sparse_attention

    S = 4096
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    lut, topk = get_block_map(q, k, 1.0, 64, 64)
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
    from sla_block_map import get_block_map

    S, prefix = 16384, 2048
    torch.manual_seed(0)
    q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16)
    plain_lut, plain_topk = get_block_map(q, k, 0.10, 64, 64)
    lut, topk = get_block_map(q, k, 0.10, 64, 64, protect_upto=prefix)

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
    ov = sla.make_override(state, 0.90, 64, 64, min_seq_len=1024)
    S = 8192
    q, k, v = (torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    out = _call(ov, q, k, v)
    assert (state["calls"], state["dense"]) == (1, 0)
    assert out.shape == (1, S, H * D)
    assert not torch.isnan(out).any()
