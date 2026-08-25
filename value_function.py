"""Online value function for reward-guided sampling.

Trains a small ENSEMBLE of MLPs incrementally on user-rated generations (pairwise ranking,
not absolute-reward regression — see train_on). At inference, provides an ensemble-pessimistic
∂reward/∂conditioning as a per-step steering gradient: gradient-ascent-style steering is prone
to walking into regions a single undertrained value function scores confidently but wrongly
(confirmed live in this project — an earlier version collapsed to a degenerate "static
character" attractor). Independently bootstrap-trained ensemble members disagree exactly in
those blind-spot regions; that disagreement damps the steering step toward zero instead of
trusting it. See [[project_ltxav_manipulation]] (the original collapse) and
[[research_2026_training_free_candidates]] (why plain gradient ascent on a small reward model
is the class of method this replaces, not a novel invention).
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineValueFunction(nn.Module):
    MIN_SAMPLES = 10
    BUFFER_SIZE = 100
    BATCH_SIZE = 16
    TRAIN_STEPS = 20
    ENSEMBLE_SIZE = 3  # tiny nets — 3x forward/backward is still near-zero vs a diffusion step

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        mid = min(256, hidden_dim // 4)

        def _make_net():
            return nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.SiLU(),
                nn.Linear(mid, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )

        self.nets = nn.ModuleList([_make_net() for _ in range(self.ENSEMBLE_SIZE)])
        self._optimizer = None
        self.buffer_c = []  # compressed conditioning tensors [hidden_dim]
        self.buffer_r = []  # reward floats
        self.n_trained = 0

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
        return self._optimizer

    @staticmethod
    def compress(conditioning):
        """[1, seq, D] or [seq, D] → [D] mean vector."""
        c = conditioning.float()
        if c.dim() == 3:
            c = c.squeeze(0)
        return c.mean(dim=0)

    def forward(self, c):
        """Ensemble MEAN prediction. Used by search() and anywhere only a score (not a
        gradient) is needed — those call sites are unaffected by the ensemble change."""
        return torch.stack([net(c) for net in self.nets], dim=0).mean(dim=0)

    def _ensemble_grad(self, c):
        """Per-member reward + gradient w.r.t. c (must already require grad), reduced to
        (mean_reward, mean_grad, agreement). agreement in [0,1] is the mean pairwise cosine
        similarity between members' gradient DIRECTIONS — high when the ensemble agrees which
        way improves the input, low when they don't (the blind-spot signal)."""
        compressed = self.compress(c)
        rewards, grads = [], []
        for net in self.nets:
            r = net(compressed.unsqueeze(0))
            g, = torch.autograd.grad(r, c, retain_graph=True)
            rewards.append(r.detach())
            grads.append(g)
        mean_reward = torch.stack(rewards).mean()
        mean_grad = torch.stack(grads).mean(dim=0)
        n = len(grads)
        if n > 1:
            unit = torch.stack([F.normalize(g.flatten().float(), dim=0) for g in grads])
            sim = unit @ unit.T
            agreement = float(((sim.sum() - n) / (n * (n - 1))).clamp(0.0, 1.0).item())
        else:
            agreement = 1.0
        return mean_reward, mean_grad, agreement

    def train_on(self, conditioning, reward):
        """Add a new (conditioning, reward) sample, then train each ensemble member via
        pairwise ranking on its OWN independently-sampled pairs from the buffer: Bradley-Terry
        loss -log(sigmoid(V(winner) - V(loser))), same objective DPO derives its update from.
        Only needs the ORDER between two rated generations to be right, not the absolute
        reward magnitude — more robust than MSE regression at 10-100 samples, and
        correct-by-construction for hand-tuned reward gaps that don't perfectly agree with
        intent (e.g. two "great" ratings a tier apart in reward but not in quality — ranking
        loss only needs both to beat a worse rating, never needs their gap to mean anything).
        Reward numbers/labels are untouched by this — see _v2_reward_admissible in
        conditioning.py for what's allowed to reach this buffer.

        Independent per-member pair sampling (bootstrap-style), on top of independent random
        init, is what gives the ensemble a meaningful disagreement signal later — members that
        saw identical data every step would converge to near-identical blind spots too."""
        with torch.inference_mode(False), torch.enable_grad():
            c = self.compress(conditioning.clone()).detach().cpu()
            self.buffer_c.append(c)
            self.buffer_r.append(float(reward))
            if len(self.buffer_c) > self.BUFFER_SIZE:
                self.buffer_c = self.buffer_c[-self.BUFFER_SIZE:]
                self.buffer_r = self.buffer_r[-self.BUFFER_SIZE:]
            if len(self.buffer_c) < 2:
                return
            # Train where the sample already is. These nets are tiny in FLOPs but not in
            # PARAMETERS — a 5120-wide input makes the first layer 94% of ~4M weights, and
            # twenty Adam steps over that on a CPU measured at ~1.1s locally and around 7.5s
            # on a rental's vCPU, per rated generation, three times over. On the GPU the
            # conditioning is already sitting on it is a few milliseconds.
            #
            # The buffer and the checkpoint stay on the CPU: the file has to stay portable,
            # and inference elsewhere still expects a CPU module, so the device is restored
            # before returning.
            home = next(self.parameters()).device
            device = self._training_device(conditioning, home)
            if device != home:
                self.to(device)
                # Adam's state tensors do not follow module.to(), so a moved optimizer steps
                # against buffers on the old device. Dropped rather than migrated — it is
                # rebuilt lazily and never persisted, so a run starts from fresh moments
                # either way.
                self._optimizer = None
            try:
                self._train_steps(device)
            finally:
                if device != home:
                    self.to(home)
                    self._optimizer = None

    @staticmethod
    def _training_device(conditioning, home):
        """Where to run the twenty training steps: the sample's GPU if it is on one."""
        try:
            if isinstance(conditioning, torch.Tensor) and conditioning.is_cuda:
                return conditioning.device
        except Exception:  # noqa: BLE001
            pass
        return home

    def _train_steps(self, device):
        with torch.inference_mode(False), torch.enable_grad():
            n = len(self.buffer_c)
            for _ in range(self.TRAIN_STEPS):
                self.optimizer.zero_grad()
                total_loss = None
                for net in self.nets:
                    idx_win, idx_lose = [], []
                    for _ in range(self.BATCH_SIZE):
                        i, j = random.sample(range(n), 2)
                        if self.buffer_r[i] == self.buffer_r[j]:
                            continue  # tie — no ranking signal, skip
                        if self.buffer_r[i] < self.buffer_r[j]:
                            i, j = j, i  # i = winner (higher reward), j = loser
                        idx_win.append(i)
                        idx_lose.append(j)
                    if not idx_win:
                        continue  # this member's draw was all ties; try again next step
                    c_win = torch.stack([self.buffer_c[i] for i in idx_win]).to(device)
                    c_lose = torch.stack([self.buffer_c[i] for i in idx_lose]).to(device)
                    margin = net(c_win) - net(c_lose)
                    member_loss = -F.logsigmoid(margin).mean()
                    total_loss = member_loss if total_loss is None else total_loss + member_loss
                if total_loss is None:
                    continue
                total_loss.backward()
                self.optimizer.step()
        self.n_trained += 1

    def gradient(self, conditioning):
        """∂reward/∂conditioning — same shape as input, scaled by ensemble agreement
        (near 0 when members disagree on direction, near 1 when they concur)."""
        mlp_device = next(self.parameters()).device
        orig_device = conditioning.device
        with torch.inference_mode(False), torch.enable_grad():
            # new_empty inside inference_mode(False) creates a normal (non-inference) tensor.
            # copy_ fills it from the inference tensor without inheriting the flag.
            c_fresh = torch.empty(conditioning.shape, dtype=torch.float32, device=mlp_device)
            c_fresh.copy_(conditioning)
            c_fresh.requires_grad_(True)
            _, mean_grad, agreement = self._ensemble_grad(c_fresh)
        return (mean_grad * agreement).to(orig_device, conditioning.dtype)

    def ascend(self, conditioning):
        """Gradient ascent on conditioning until reward plateaus. Self-terminating, no user
        params. Each step is scaled by ensemble agreement — low agreement (an OOD/blind-spot
        region none of the bootstrap-trained members actually learned) shrinks the step toward
        zero, which combined with the plateau-based early exit means ascent naturally stalls
        instead of confidently walking into a spurious high-score region — the failure mode
        that produced the documented "static character" collapse under a single value function."""
        if not self.is_ready():
            return conditioning
        mlp_device = next(self.parameters()).device
        orig_device = conditioning.device
        orig_dtype = conditioning.dtype
        step_size = conditioning.float().norm().item() * 0.005
        # Cap total displacement at 8% of original norm — prevents stripping action/content
        max_displacement = conditioning.float().norm().item() * 0.08
        prev_reward = -float("inf")
        with torch.inference_mode(False), torch.enable_grad():
            c_orig = torch.empty(conditioning.shape, dtype=torch.float32, device=mlp_device)
            c_orig.copy_(conditioning)
            c = c_orig.clone()
            for _ in range(50):
                c = c.detach().requires_grad_(True)
                reward, grad, agreement = self._ensemble_grad(c)
                reward_val = reward.item()
                step = F.normalize(grad.float(), dim=-1) * agreement
                c = c.detach() + step_size * step
                # Project back if displacement exceeds cap
                delta = c - c_orig
                disp = delta.norm()
                if disp > max_displacement:
                    c = c_orig + delta * (max_displacement / disp)
                if reward_val - prev_reward < 1e-3:
                    break
                prev_reward = reward_val
        return c.detach().to(orig_device, orig_dtype)

    def search(self, conditioning, n=16, eps_scale=1.0):
        """Monte Carlo conditioning search: score N random perturbations, return the best.

        eps_scale widens (>1) or tightens (<1) the perturbation radius. The path planner uses
        a wider radius to push a disliked config region toward a different conditioning, and the
        caller skips this search entirely to LOCK a liked region's conditioning in place."""
        if not self.is_ready():
            return conditioning
        mlp_device = next(self.parameters()).device
        orig_device = conditioning.device
        orig_dtype = conditioning.dtype
        eps = conditioning.float().norm().item() * 0.025 * float(eps_scale)
        with torch.inference_mode(False), torch.no_grad():
            c_base = torch.empty(conditioning.shape, dtype=torch.float32, device=mlp_device)
            c_base.copy_(conditioning)
            best = c_base
            best_score = float(self.forward(self.compress(c_base).unsqueeze(0)).item())
            for _ in range(n):
                noise = F.normalize(torch.randn_like(c_base).flatten(), dim=0).reshape(c_base.shape)
                candidate = c_base + eps * noise
                score = float(self.forward(self.compress(candidate).unsqueeze(0)).item())
                if score > best_score:
                    best_score = score
                    best = candidate
        return best.detach().to(orig_device, orig_dtype)

    def is_ready(self):
        return len(self.buffer_c) >= self.MIN_SAMPLES

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        buf = torch.stack(self.buffer_c) if self.buffer_c else torch.zeros(0, self.hidden_dim)
        torch.save({
            "hidden_dim": self.hidden_dim,
            "state_dict": self.state_dict(),
            "buffer_c": buf,
            "buffer_r": list(self.buffer_r),
            "n_trained": self.n_trained,
        }, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        vf = cls(hidden_dim=data["hidden_dim"])
        try:
            vf.load_state_dict(data["state_dict"])
        except Exception:
            # Pre-ensemble checkpoint (single "net.*" instead of "nets.<i>.*"). Warm-start
            # every ensemble member from the one saved net rather than discarding trained
            # weights outright — they'll diverge from there as independent bootstrap training
            # proceeds. If even this remapping fails, members stay at fresh random init; either
            # way the buffer below (the actually irreplaceable part — real accumulated ratings)
            # is restored unconditionally, regardless of whether the weights loaded.
            legacy_sd = data["state_dict"]
            if all(k.startswith("net.") for k in legacy_sd):
                remapped = {k[len("net."):]: v for k, v in legacy_sd.items()}
                for net in vf.nets:
                    try:
                        net.load_state_dict(remapped)
                    except Exception:
                        pass
        buf = data["buffer_c"]
        vf.buffer_c = [buf[i] for i in range(len(buf))]
        vf.buffer_r = list(data["buffer_r"])
        vf.n_trained = data.get("n_trained", 0)
        return vf

    @classmethod
    def load_or_create(cls, path, hidden_dim=None):
        if os.path.exists(path):
            try:
                return cls.load(path)
            except Exception:
                pass
        return cls(hidden_dim=hidden_dim) if hidden_dim is not None else None


def compress_packed_latent(x, hidden_dim):
    """Pool a packed AV latent [B,1,N] (LTXAV's flattened video/audio stream layout — see
    samplers._get_latent_shapes) down to a fixed-size [hidden_dim] vector, differentiably.

    N varies per generation (scene length/resolution), unlike text conditioning's fixed
    encoder width, so adaptive pooling (not a plain mean) is what makes a single value
    function usable across differently-shaped runs. Idempotent on already-compressed input
    (shape [hidden_dim]) so the same call works whether fed a live raw latent (inference,
    where autograd needs to flow through this op to produce a full-shape gradient) or a
    pre-compressed snapshot loaded back from disk (training, where the raw latent was never
    persisted — only this vector was, to keep the sidecar file small)."""
    if x.dim() == 1 and x.shape[0] == hidden_dim:
        return x
    c = x.float()
    if c.dim() == 1:
        c = c.view(1, 1, -1)
    elif c.dim() == 2:
        c = c.unsqueeze(1)
    elif c.dim() > 3:
        c = c.reshape(c.shape[0], 1, -1)
    pooled = F.adaptive_avg_pool1d(c, hidden_dim)  # [B,1,hidden_dim]
    return pooled.mean(dim=0).reshape(-1)  # average over batch (usually B=1, a no-op) -> [hidden_dim]


class LatentValueFunction(OnlineValueFunction):
    """OnlineValueFunction trained on the SAMPLER'S OWN predicted output (a packed-latent
    x0_hat estimate) instead of the input text conditioning — same tiny-MLP mechanism, just
    fed a different, more direct signal. See [[research_2026_training_free_candidates]] /
    NoiseTilt for the idea this borrows (reward gradient computed from the model's own
    prediction) and [[project_embed_guidance]] for the conditioning-space sibling this
    complements. Cost stays the same as embed_guidance: one backward pass through a few
    hundred params, no extra diffusion-model forward pass."""

    DEFAULT_HIDDEN_DIM = 512

    def compress(self, x):
        return compress_packed_latent(x, self.hidden_dim)


