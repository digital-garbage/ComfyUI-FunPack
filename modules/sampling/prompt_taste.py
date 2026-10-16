"""What the taste modules that read a prompt's pooled embedding share. A plain file, not a module: the modules that use it
(score_slider, prompt_nudge, velocity_bias) must each load without the others."""
import torch
import torch.nn.functional as F

KIND = "prompt_taste"
NAME = "pooled"
NEAREST, MIN_SIM = 3, 0.5
MIN_LIKED = 3               # v4: direction_count >= 3


def direction(rows, current=None, similar=False):
    """-> (unit vector | None, how it was found)."""
    pooled = [(r["rows"][NAME].detach().float().cpu(), r["reward"]) for r in rows if NAME in r["rows"]]
    liked = [p for p, w in pooled if w > 0]
    if len(liked) < MIN_LIKED:
        return None, f"needs {MIN_LIKED} liked clips ({len(liked)} so far)"
    mean = torch.stack([p for p, _w in pooled]).mean(0)
    if similar and current is not None:
        current = current.detach().float().cpu()      # the stored rows live on the CPU; the live prompt is on the GPU
        near = sorted(((float(F.cosine_similarity(p, current, dim=0)), p) for p in liked),
                      key=lambda s: -s[0])[:NEAREST]
        near = [(s, p) for s, p in near if s >= MIN_SIM]
        if near:
            d = sum(s * F.normalize(p - mean, dim=0) for s, p in near)
            if d.norm() > 1e-8:
                return F.normalize(d, dim=0), f"from {len(near)} similar liked prompt(s)"
    d = torch.stack(liked).mean(0) - mean
    if d.norm() <= 1e-8:
        return None, "liked prompts don't differ from the rest yet"
    return F.normalize(d, dim=0), f"from {len(liked)} liked of {len(pooled)} rated"


def cond_row(c, to):
    """The batch row holding the positive prompt, or None when this call carries only the negative.
    A split CFG call (cond and uncond in separate calls) must not teach the negative prompt."""
    kinds = list((to or {}).get("cond_or_uncond") or [])
    if not kinds:
        return 0
    return next((i for i, k in enumerate(kinds) if k == 0 and i < c.shape[0]), None)
