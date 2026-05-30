"""Online value function for reward-guided sampling.

Trains a small MLP incrementally on user-rated generations.
At inference, provides ∂reward/∂conditioning as a per-step steering gradient.
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

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        mid = min(256, hidden_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
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
        return self.net(c)

    def train_on(self, conditioning, reward):
        """Add a new (conditioning, reward) sample and do a few training steps."""
        with torch.inference_mode(False), torch.enable_grad():
            c = self.compress(conditioning.clone()).detach().cpu()
            self.buffer_c.append(c)
            self.buffer_r.append(float(reward))
            if len(self.buffer_c) > self.BUFFER_SIZE:
                self.buffer_c = self.buffer_c[-self.BUFFER_SIZE:]
                self.buffer_r = self.buffer_r[-self.BUFFER_SIZE:]
            if len(self.buffer_c) < 2:
                return
            device = next(self.parameters()).device
            for _ in range(self.TRAIN_STEPS):
                idx = random.sample(range(len(self.buffer_c)), min(self.BATCH_SIZE, len(self.buffer_c)))
                c_b = torch.stack([self.buffer_c[i] for i in idx]).to(device)
                r_b = torch.tensor([self.buffer_r[i] for i in idx], device=device).unsqueeze(1)
                self.optimizer.zero_grad()
                F.mse_loss(self.forward(c_b), r_b).backward()
                self.optimizer.step()
        self.n_trained += 1

    def gradient(self, conditioning):
        """∂reward/∂conditioning — same shape as input."""
        mlp_device = next(self.parameters()).device
        orig_device = conditioning.device
        with torch.inference_mode(False), torch.enable_grad():
            # new_empty inside inference_mode(False) creates a normal (non-inference) tensor.
            # copy_ fills it from the inference tensor without inheriting the flag.
            c_fresh = torch.empty(conditioning.shape, dtype=torch.float32, device=mlp_device)
            c_fresh.copy_(conditioning)
            c_fresh.requires_grad_(True)
            reward = self.forward(self.compress(c_fresh).unsqueeze(0))
            reward.backward()
        return c_fresh.grad.to(orig_device, conditioning.dtype)

    def ascend(self, conditioning):
        """Gradient ascent on conditioning until reward plateaus. Self-terminating, no user params."""
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
                reward = self.forward(self.compress(c).unsqueeze(0))
                reward.backward()
                reward_val = reward.item()
                grad = F.normalize(c.grad.float(), dim=-1)
                c = c.detach() + step_size * grad
                # Project back if displacement exceeds cap
                delta = c - c_orig
                disp = delta.norm()
                if disp > max_displacement:
                    c = c_orig + delta * (max_displacement / disp)
                if reward_val - prev_reward < 1e-3:
                    break
                prev_reward = reward_val
        return c.detach().to(orig_device, orig_dtype)

    def search(self, conditioning, n=16):
        """Monte Carlo conditioning search: score N random perturbations, return the best."""
        if not self.is_ready():
            return conditioning
        mlp_device = next(self.parameters()).device
        orig_device = conditioning.device
        orig_dtype = conditioning.dtype
        eps = conditioning.float().norm().item() * 0.025
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
        vf.load_state_dict(data["state_dict"])
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


