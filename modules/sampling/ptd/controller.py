"""Running a PTD adapter on a native MiniMax-H3 model: one expert set per hop, switched by hooks.

Adapters are low-rank additions to linear outputs (forward hooks), so ordinary LoRAs, which are weight
patches, stack with them in either order. The bank stays in system RAM; each layer copies only the
current hop's factors to the GPU, and int8 row scales are folded into the small factors.

Hooks live for one model call and are removed even when the call fails. Block arithmetic follows PTD's
reference order (bf16 products rounded one by one, as the experts were trained) unless `fuse_blocks`
asks for ComfyUI's fused native blocks.

Controller, hooks, block maths and the model-call wrapper follow ComfyUI-PTD (EPFL VITA Lab, Apache-2.0),
with the int8 factors, in-place residual and DynamicVRAM handling of ComfyUI-PTD-Compact (Nynxz,
Apache-2.0) and the output-head fusion of ComfyUI-MiniMaxH3-PTD (taobao-content-ai, Apache-2.0).
"""

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
import copy
from dataclasses import dataclass
import math
import threading
import weakref

import torch
import torch.nn.functional as F

from . import schedule

_FORWARD = ContextVar("funpack_ptd_forward", default=None)
_LOCKS = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()


def _model_lock(model):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(model, threading.RLock())


def _check_interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


# --- block arithmetic in PTD's reference order ------------------------------------------------------

def _modulated(normed, shift, scale, segments):
    result = torch.empty_like(normed)
    for start, stop, row in segments:
        result[start:stop] = normed[start:stop] * (1.0 + scale[row].to(normed.dtype)) + shift[row].to(normed.dtype)
    return result


def _residual(state, update, gate, segments):
    """`state + update * gate`, added in place. The product is rounded on its own (no addcmul) as PTD was
    trained; adding into the stream keeps ComfyUI's allocation compiler from seeing a block output that
    outlives its block."""
    for start, stop, row in segments:
        state[start:stop].add_(update[start:stop] * gate[row].to(state.dtype))
    return state


def _block_math(block, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
    options = {} if transformer_options is None else transformer_options
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = block.adaln_proj(t_emb)
    normed = _modulated(block.norm1(x), shift_a, scale_a, mod_segments)
    update = (block.attn if attention is None else attention)(normed, rope_freqs=rope_freqs, transformer_options=options)
    x = _residual(x, update, gate_a, mod_segments)
    normed = _modulated(block.norm2(x), shift_m, scale_m, mod_segments)
    return _residual(x, block.mlp(normed), gate_m, mod_segments)


def _split_half_rotation(x, table):
    cos = table[0, :, 0, :, 0, 0].repeat(1, 2).unsqueeze(1).to(x.dtype)
    sin = table[0, :, 0, :, 1, 0].repeat(1, 2).unsqueeze(1).to(x.dtype)
    width = cos.shape[-1]
    leading = x[..., :width]
    first, second = leading.chunk(2, dim=-1)
    # The two products and their sum stay separate: a fused RMS-RoPE would do the rotation in fp32 first.
    result = leading * cos + torch.cat((-second, first), dim=-1) * sin
    return torch.cat((result, x[..., width:]), dim=-1)


def _attention_math(attention, x, rope_freqs=None, transformer_options=None):
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    options = {} if transformer_options is None else transformer_options
    streams = attention.qkv_proj(x).chunk(3, dim=-1)
    q, k, v = [value.reshape(x.shape[0], attention.heads, attention.head_dim) for value in streams]
    q, k = attention.q_norm(q), attention.k_norm(k)
    if rope_freqs is not None:
        q, k = _split_half_rotation(q, rope_freqs), _split_half_rotation(k, rope_freqs)
    packed = [AttentionTensorContainer(value.transpose(0, 1).unsqueeze(0)) for value in (q, k, v)]
    result = optimized_attention(*packed, attention.heads, skip_reshape=True,
                                 preferred_attention=attention.comfy_attention, transformer_options=options)
    return attention.out_proj(result.squeeze(0))


def _scope_forward(module, function, frame, originals):
    original = module.forward
    originals.append((module, "forward" in vars(module), original))

    def forward(*args, **kwargs):
        if _FORWARD.get() is not frame:
            return original(*args, **kwargs)
        return function(module, *args, **kwargs)

    module.forward = forward


@contextmanager
def _without_comfy_compiler():
    """Switch off ComfyUI's allocation compiler (and its CUDA graphs) for one call. Used only for a run in
    which the compiler failed: its recording does not expect the extra tensors PTD's block maths allocates."""
    try:
        from comfy.cli_args import args
    except ImportError:
        yield
        return
    saved = getattr(args, "disable_comfy_compiler", False), getattr(args, "disable_cuda_graphs", False)
    args.disable_comfy_compiler = args.disable_cuda_graphs = True
    try:
        yield
    finally:
        args.disable_comfy_compiler, args.disable_cuda_graphs = saved


@dataclass(frozen=True)
class Hop:
    index: int
    sigma: float


class Controller:
    """The adapter bank of one loaded file, bound to one diffusion model. Hop selection is context-local;
    calls that share a backbone are serialised."""

    def __init__(self, native, adapter):
        self.native = native
        self.adapter = adapter
        self.hops = adapter.hops
        self.grid = torch.tensor(adapter.grid, dtype=torch.float64)
        self._lock = _model_lock(native)
        self._hop = ContextVar(f"funpack_ptd_hop_{id(self)}", default=None)
        self.fuse_blocks = False
        self.allocation_compiler = True

    @contextmanager
    def activate(self, hop, sigma):
        if isinstance(hop, bool) or not isinstance(hop, int) or not 0 <= hop < self.hops:
            raise ValueError(f"PTD hop must be in [0, {self.hops})")
        expected = float(self.grid[hop])
        if abs(float(sigma) - expected) > 2e-6:
            raise ValueError(f"Hop {hop} must run at its trained time {expected:.6f}, not {float(sigma):.6f}")
        token = self._hop.set(Hop(hop, expected))
        try:
            yield self
        finally:
            self._hop.reset(token)

    def current(self):
        hop = self._hop.get()
        if hop is None:
            raise RuntimeError("PTD adapters are on this model but no PTD hop is running: the model was called "
                               "outside the PTD sampler")
        return hop

    def _linear_hook(self, frame, experts):
        # q/k/v share one input: their A factors run as one matmul. Each expert's delta is added into the
        # fresh linear output in place; int8 row scales are folded into the small factors.
        qkv = sorted((e for e in experts if e.qkv_index is not None), key=lambda e: e.qkv_index)
        plain = [e for e in experts if e.qkv_index is None]
        ranks = [e.a.shape[0] for e in qkv]

        def factors(e, dev, dt):
            a = e.a.to(dev, non_blocking=True).to(dt)
            b = e.b.to(dev, non_blocking=True).to(dt)
            if e.sa is not None:
                a = a * e.sa.to(dev, dt)[:, None]
            if e.sb is not None:
                b = b * e.sb.to(dev, dt)[:, None]
            return a, b

        def hook(module, args, kwargs, output):
            if _FORWARD.get() is not frame:
                return output
            _check_interrupt()
            x = args[0] if args else kwargs.get("input")
            dev, dt = x.device, output.dtype      # the layer's compute dtype
            x = x.to(dt)
            for e in plain:
                a, b = factors(e, dev, dt)
                output.add_(F.linear(F.linear(x, a), b))
            if qkv:
                pairs = [factors(e, dev, dt) for e in qkv]
                hs = F.linear(x, torch.cat([a for a, _ in pairs])).split(ranks, dim=-1)
                width = output.shape[-1] // 3
                for e, (_, b), h in zip(qkv, pairs, hs):
                    output[..., e.qkv_index * width:(e.qkv_index + 1) * width].add_(F.linear(h, b))
            return output
        return hook

    @contextmanager
    def forward_hooks(self, hop):
        """Scope the adapters, the PTD clocks and (unless fuse_blocks) PTD's block maths to one model call."""
        native = self.native
        with self._lock, ExitStack() as stack:
            if not self.allocation_compiler:
                stack.enter_context(_without_comfy_compiler())
            frame = object()
            token = _FORWARD.set(frame)
            handles, forced, forwards = [], [], []
            try:
                if not self.fuse_blocks:
                    for block in native.blocks:
                        if all(hasattr(block, n) for n in ("norm1", "norm2", "adaln_proj", "attn", "mlp")):
                            _scope_forward(block, _block_math, frame, forwards)
                            _scope_forward(block.attn, _attention_math, frame, forwards)
                self._time_hooks(frame, hop, handles)
                heads = self.adapter.heads[hop.index] if self.adapter.heads else {}
                for name, (weight, bias) in heads.items():
                    parent, _, attr = name.rpartition(".")
                    handles.append(getattr(native.get_submodule(parent), attr).register_forward_hook(
                        self._fused_head(frame, weight, bias)))
                for name, by_hop in self.adapter.bank.items():
                    module = native.get_submodule(name)
                    # Native SwiGLU may fuse activation and fc2, bypassing Module.__call__ and its hooks;
                    # ComfyUI's public flag routes the call through the plain linear.
                    if hasattr(module, "comfy_force_forward") or name.endswith(".mlp.fc2"):
                        forced.append((module, "comfy_force_forward" in vars(module),
                                       getattr(module, "comfy_force_forward", None)))
                        module.comfy_force_forward = True
                    handles.append(module.register_forward_hook(
                        self._linear_hook(frame, self.adapter.bank[name][hop.index]), with_kwargs=True))
                factor = self.adapter.audio[hop.index]

                def audio_clock(module, args, output):
                    if _FORWARD.get() is not frame:
                        return output
                    # Scaled before native H3 casts to the latent dtype, like PTD's fp32 output heads.
                    return output * torch.as_tensor(factor, dtype=output.dtype, device=output.device)

                handles.append(native.final_layer.audio_out.register_forward_hook(audio_clock))
                _check_interrupt()
                yield
                _check_interrupt()
            finally:
                for handle in reversed(handles):
                    handle.remove()
                for module, had, value in reversed(forced):
                    if had:
                        module.comfy_force_forward = value
                    else:
                        del module.comfy_force_forward
                for module, had, original in reversed(forwards):
                    if had:
                        module.forward = original
                    else:
                        del module.forward
                _FORWARD.reset(token)

    @staticmethod
    def _fused_head(frame, weight, bias):
        def hook(module, args, output):
            if _FORWARD.get() is not frame:
                return output
            x = args[0]
            return F.linear(x.to(output.dtype), weight.to(x.device, output.dtype),
                            None if bias is None else bias.to(x.device, output.dtype))
        return hook

    def _time_hooks(self, frame, hop, handles):
        native = self.native
        if not hasattr(native, "time_embedder"):      # pruned (AdaLN-curve) base: no time embedder to steer
            return
        remembered = {}

        def exact_times(module, args):
            if _FORWARD.get() is not frame:
                return args
            times = args[0]
            if not isinstance(times, torch.Tensor) or times.ndim != 1:
                raise ValueError("PTD expects a vector of native H3 time labels")
            video_time = 1.0 - hop.sigma
            candidates = torch.tensor([video_time, 1.0 - schedule.audio_sigma(hop.sigma), max(video_time, .999)],
                                      device=times.device, dtype=torch.float32).unique()
            distance, indices = (times[:, None] - candidates[None]).abs().min(dim=1)
            if not bool((distance <= 2e-6).all()):
                raise ValueError("Native H3 time labels do not match the active PTD clocks")
            # Native H3 rounds sigma to fp32 first; PTD subtracts in float64. Keep the table order.
            return (candidates[indices],) + args[1:]

        def remember(module, args, output):
            if _FORWARD.get() is frame:
                remembered["fp32"] = output

        def precise_adaln_input(module, args):
            if _FORWARD.get() is not frame:
                return args
            if "fp32" not in remembered:
                raise RuntimeError("PTD AdaLN ran before the time embedder")
            # Diffusers applies SiLU to the fp32 time embedding before casting; native H3 casts first.
            return (F.silu(remembered["fp32"]).to(args[0].dtype),) + args[1:]

        handles.append(native.time_embedder.register_forward_pre_hook(exact_times))
        handles.append(native.time_embedder.register_forward_hook(remember))
        for layer in (*native.blocks, native.final_layer):
            adaln = getattr(layer, "adaln_proj", None)
            if adaln is not None and getattr(adaln, "apply_silu", True):
                handles.append(adaln.linear.register_forward_pre_hook(precise_adaln_input))


class ModelCall:
    """The model-function wrapper: PTD carries physical latents, so the native shifts are neutralised for the
    call and the adapters are switched on for it."""

    def __init__(self, controller, previous=None):
        self.controller = controller
        self.previous = previous

    def __call__(self, model_function, kwargs):
        hop = self.controller.current()
        sigma = kwargs.get("timestep")
        if not isinstance(sigma, torch.Tensor) or sigma.numel() == 0:
            raise ValueError("PTD needs a tensor sigma from the sampler")
        if not bool(torch.allclose(sigma, torch.full_like(sigma, hop.sigma), rtol=0.0, atol=2e-6)):
            raise ValueError("The model sigma does not match the active PTD hop")
        forwarded = dict(kwargs)
        conditions = dict(kwargs.get("c", {}))
        options = dict(conditions.get("transformer_options", {}))
        options["minimax_h3_sigma_shift_video"] = 1.0
        options["minimax_h3_sigma_shift_audio"] = schedule.native_audio_shift(hop.sigma)
        conditions["transformer_options"] = options
        payload = dict(conditions.get("minimax_payload", {}))
        payload["audio_scale"] = 1.0
        conditions["minimax_payload"] = payload
        forwarded["c"] = conditions
        with self.controller.forward_hooks(hop):
            if self.previous is not None:
                return self.previous(model_function, forwarded)
            return model_function(forwarded["input"], forwarded["timestep"], **conditions)

    def to(self, device):
        if self.previous is not None and hasattr(self.previous, "to"):
            self.previous = self.previous.to(device)
        return self

    def cleanup(self, **kwargs):
        if self.previous is not None and hasattr(self.previous, "cleanup"):
            self.previous.cleanup(**kwargs)

    def models(self):
        if self.previous is not None and hasattr(self.previous, "models"):
            return self.previous.models()
        return []


def attach(patcher, controller, key, tag):
    """Put `controller` on `patcher` (a clone the caller owns): raw text embeddings so each hop's refiner
    adapters run, PTD's audio shift, and the model-call wrapper. Everything is tagged `key` for removal."""
    native = controller.native
    if not all(hasattr(native, n) for n in ("blocks", "token_refiner", "condition_proj", "final_layer",
                                            "sigma_shift_video", "sigma_shift_audio")):
        raise ValueError("PTD needs a native MiniMax-H3 model")
    sampling = copy.deepcopy(patcher.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift") or not hasattr(sampling, "shift"):
        raise ValueError("PTD needs the native MiniMax-H3 audio/video flow sampling")
    sampling.audio_shift = sampling.shift
    patcher.add_object_patch("model_sampling", tag(sampling, key))

    text_dim = native.condition_proj.in_features

    def raw_text(text_states):
        if text_states.shape[-1] != text_dim:
            raise ValueError("PTD needs raw MiniMax-H3 text conditioning, not pre-refined embeddings")
        return text_states

    patcher.add_object_patch("diffusion_model.preprocess_text_embeds", tag(raw_text, key))
    previous = patcher.model_options.get("model_function_wrapper")
    wrapper = tag(ModelCall(controller, previous), key)
    wrapper._funpack_prev = previous
    patcher.set_model_unet_function_wrapper(wrapper)
    patcher.model_options[CONTROLLER_KEY] = tag(controller, key)


CONTROLLER_KEY = "funpack_ptd_controller"
