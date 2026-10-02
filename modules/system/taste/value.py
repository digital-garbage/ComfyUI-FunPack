"""A tiny learned judge of rated clips, for features that steer by its gradient.

Three small MLPs, each trained on its own random pairs of rated descriptors to
rank the liked one above the disliked one (Bradley-Terry: only the ORDER must
be right, which is what a handful of ratings can support). Their gradients are
averaged and scaled by how much they agree on the direction: where they
disagree the judge is guessing, and the push shrinks toward zero. A single
judge in v4's first version walked confidently into a "static character"
collapse; the disagreement damping is the fix for that, kept here.

Rebuilt from the rated rows whenever they change, never updated in place: the
rows are the record, a judge is a pure function of them.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 512
MIN_SAMPLES = 10
ENSEMBLE = 3
BATCH = 16
# ponytail: one fixed training budget over the whole buffer instead of v4's
# replay (20 steps per rating as it arrived); same objective, bounded cost.
STEPS = 400
BUFFER = 100


def describe(video):
    """Any-shaped picture latent -> [DIM], differentiably (adaptive pooling, so
    clips of any length or size share one judge)."""
    c = video.float().reshape(1, 1, -1)
    return F.adaptive_avg_pool1d(c, DIM).reshape(-1)


class Judge(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)

        def net():
            layers = nn.Sequential(nn.Linear(DIM, 128), nn.SiLU(), nn.Linear(128, 64),
                                   nn.SiLU(), nn.Linear(64, 1))
            for p in layers.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=gen) * (1.0 / max(1, p.shape[-1])) ** 0.5)
            return layers

        self.nets = nn.ModuleList(net() for _ in range(ENSEMBLE))

    @classmethod
    def train_on(cls, descriptors, rewards, seed=0):
        """-> a trained Judge, or None under MIN_SAMPLES or with no two ratings
        that differ (nothing to rank)."""
        pairs = list(zip(descriptors, rewards))[-BUFFER:]
        if len(pairs) < MIN_SAMPLES or len({r for _d, r in pairs}) < 2:
            return None
        judge = cls(seed)
        pick = random.Random(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        xs = torch.stack([d.float().reshape(-1) for d, _r in pairs]).to(device)
        rs = [float(r) for _d, r in pairs]
        n = len(rs)
        with torch.inference_mode(False), torch.enable_grad():
            judge.to(device)
            opt = torch.optim.Adam(judge.parameters(), lr=2e-3)
            for _ in range(STEPS):
                opt.zero_grad()
                loss = None
                for net in judge.nets:
                    win, lose = [], []
                    for _ in range(BATCH):
                        i, j = pick.sample(range(n), 2)
                        if rs[i] == rs[j]:
                            continue
                        if rs[i] < rs[j]:
                            i, j = j, i
                        win.append(i)
                        lose.append(j)
                    if win:
                        term = -F.logsigmoid(net(xs[win]) - net(xs[lose])).mean()
                        loss = term if loss is None else loss + term
                if loss is not None:
                    loss.backward()
                    opt.step()
            judge.cpu()
        judge.requires_grad_(False)
        return judge

    def gradient(self, video):
        """d(score)/d(video), same shape, scaled by the members' agreement."""
        device = video.device
        if next(self.parameters()).device != device:
            self.to(device)
        with torch.inference_mode(False), torch.enable_grad():
            x = torch.empty(video.shape, dtype=torch.float32, device=device)
            x.copy_(video)
            x.requires_grad_(True)
            d = describe(x)
            grads = [torch.autograd.grad(net(d.unsqueeze(0)).sum(), x, retain_graph=True)[0]
                     for net in self.nets]
        unit = torch.stack([F.normalize(g.flatten(), dim=0) for g in grads])
        n = len(grads)
        agreement = ((unit @ unit.T).sum() - n) / (n * (n - 1)) if n > 1 else torch.ones(())
        return (torch.stack(grads).mean(0) * agreement.clamp(0.0, 1.0)).to(video.dtype)

    def nudge(self, video, amount):
        """video + this judge's gradient, sized to `amount` x the video's own
        norm: the raw gradient reaches each value through a pooling window, so
        used as-is it would be far too small to matter."""
        grad = self.gradient(video)
        gn = grad.float().norm()
        if not bool(gn > 0):
            return video
        return video + grad * (amount * video.float().norm() / gn).to(video.dtype)

    def score(self, video):
        """The members' mean score for a picture: higher = more like what was liked."""
        device = video.device
        if next(self.parameters()).device != device:
            self.to(device)
        with torch.no_grad():
            d = describe(video.float()).unsqueeze(0)
            return float(sum(net(d) for net in self.nets)) / len(self.nets)
