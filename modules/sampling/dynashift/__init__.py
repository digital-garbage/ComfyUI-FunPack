"""DynaShift: steer each late step's predicted picture away from clips you disliked.

A negative prompt for a model that ignores negative prompts, made of your own
disliked clips instead of words. Every run banks its final picture latent on the
taste key; rating it decides which side of the bank it lands on. Then, over the
last half of the steps:

1. each frame of the current prediction is matched (cosine, on a pooled
   fingerprint) against every banked disliked frame -- by content, not time, so
   scene position and length don't matter;
2. a disliked clip made from a different prompt counts less;
3. a frame that matches above the threshold has the matched frame's direction
   taken out of it, harder the closer the match. Once the unwanted thing is gone
   the match drops and it stops by itself.

With 2+ liked and 2+ disliked clips at this resolution it also nudges every
frame along liked-minus-disliked: an average direction, never toward one clip
(there is one way to BE a past clip and many ways to not resemble one).

No extra model pass. Sound untouched by construction.

User-confirmed on H3 (2026-09-20) at threshold 0.6. Every earlier "does
nothing" verdict was measured at 0.95, which is the practically-off end.
"""

import torch
import torch.nn.functional as F
from comfy.patcher_extension import WrappersMP

from ..._core import dit_hooks, input_steer, log, registry, streams

ID = "dynashift"
TITLE = "DynaShift"
MOUNT = "generation.sampling"
STAGE = "guidance"
CATEGORY = "guidance"
STATUS = "proven"
REQUIRES = ["temporal_latent"]

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Avoid what I disliked",
        "hint": "Steers away from clips you rated down, and toward ones you liked.",
    },
    "strength": {
        "type": "float", "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
        "label": "Strength", "ui": "slider",
        "hint": "0 = only learn.",
        "when": {"enabled": True},
    },
    "threshold": {
        "type": "float", "default": 0.6, "min": 0.3, "max": 0.95, "step": 0.05,
        "label": "How alike before it acts", "ui": "slider",
        "hint": "Lower acts on looser resemblance. 0.95 almost never acts.",
        "when": {"enabled": True},
    },
}

KIND = "dynashift"
BANK = 8            # per side: v4's ring size
MIN_FOR_PULL = 2
_DESC = 512


def _say(message):
    log.once(f"{ID}:{message}", log.ALERT, "FunPack DynaShift", message)


def _pooled(cond):
    if not torch.is_tensor(cond):
        return None
    p = cond.detach().float()
    while p.dim() > 1:
        p = p.mean(0)
    return p


def _frames(latent):
    """[C, T, H, W] -> [T, C*H*W] float."""
    return latent.permute(1, 0, 2, 3).reshape(latent.shape[1], -1).float()


def _fingerprint(frames):
    return F.normalize(F.adaptive_avg_pool1d(frames.unsqueeze(1), _DESC).squeeze(1), dim=-1)


class Bank:
    """The key's banked clips, prepared once per (device, C, H, W)."""

    def __init__(self, rows):
        rated = [r for r in rows if torch.is_tensor(r["rows"].get("latent"))
                 and r["rows"]["latent"].dim() == 4]
        self.negatives = [r["rows"] for r in rated if r["reward"] < 0][-BANK:]
        self.positives = [r["rows"] for r in rated if r["reward"] > 0][-BANK:]
        self._ready = {}

    def at(self, device, c, h, w):
        key = (str(device), c, h, w)
        if key not in self._ready:
            self._ready[key] = self._prepare(device, c, h, w)
        return self._ready[key]

    def _prepare(self, device, c, h, w):
        def fits(entry):
            lat = entry["latent"]
            return lat.shape[0] == c and lat.shape[2] == h and lat.shape[3] == w

        negs = [e for e in self.negatives if fits(e)]
        poss = [e for e in self.positives if fits(e)]
        if len(negs) < len(self.negatives) or len(poss) < len(self.positives):
            _say(f"{len(self.negatives) - len(negs) + len(self.positives) - len(poss)} banked "
                 "clip(s) skipped: made at a different resolution than this run")
        if not negs:
            return None
        units, descs, owner, conds = [], [], [], []
        for i, e in enumerate(negs):
            f = _frames(e["latent"].to(device))
            descs.append(_fingerprint(f))
            units.append(F.normalize(f, dim=-1).half())
            owner += [i] * f.shape[0]
            conds.append(e.get("cond"))
        units = torch.cat(units)
        pull = None
        if len(negs) >= MIN_FOR_PULL and len(poss) >= MIN_FOR_PULL:
            liked = torch.cat([F.normalize(_frames(e["latent"].to(device)), dim=-1) for e in poss])
            diff = liked.mean(0) - units.float().mean(0)
            n = diff.norm()
            if torch.isfinite(n) and n > 1e-8:
                pull = diff / n
        return torch.cat(descs), units, torch.tensor(owner, device=device), conds, pull


def shift(video, bank, strength, threshold, cond=None):
    """One step's x0 video [B, C, T, H, W] -> the shifted one, or None when
    nothing changed."""
    b, c, t, h, w = video.shape
    ready = bank.at(video.device, c, h, w)
    if ready is None:
        return None
    desc, units, owner, conds, pull = ready
    current = _pooled(cond)
    weights = torch.ones(len(conds), device=video.device)
    if current is not None:
        for i, stored in enumerate(conds):
            if torch.is_tensor(stored) and stored.numel() == current.numel():
                weights[i] = F.cosine_similarity(current.to(video.device), stored.to(video.device).float(),
                                                 dim=0).clamp(0.0, 1.0)
    out = []
    for sample in video:
        f = _frames(sample)
        best, idx = ((_fingerprint(f) @ desc.T) * weights[owner]).max(dim=1)
        g = ((best - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0) * strength
        matched = units[idx].float()
        coef = (f * matched).sum(-1).clamp(min=0.0)          # only frames that point AT it
        f = f - (g * coef).unsqueeze(1) * matched
        if pull is not None:
            f = f + strength * f.norm(dim=-1, keepdim=True) * pull.unsqueeze(0)
        out.append(f.reshape(t, c, h, w).permute(1, 0, 2, 3))
    return torch.stack(out).to(video.dtype)


def install(patcher, values, key):
    if not values.get("enabled"):
        return None
    taste = registry.current().ask("taste_store", patcher)
    if taste is None:
        _say("off: set a Taste key first -- it is what your ratings teach")
        return None
    strength = float(values.get("strength", 0.3))
    threshold = float(values.get("threshold", 0.6))
    live = {"bank": Bank([])}

    def fresh():
        bank = live["bank"] = Bank(taste.rows(KIND, blind_to="composition"))
        if not bank.negatives:
            summary = "learning: nothing disliked banked yet"
        else:
            pull = ("pulling toward liked" if len(bank.positives) >= MIN_FOR_PULL
                    and len(bank.negatives) >= MIN_FOR_PULL
                    else "no pull yet (needs 2 liked + 2 disliked)")
            summary = f"{len(bank.negatives)} disliked, {len(bank.positives)} liked banked; {pull}"
        log.once(f"{ID}:state", log.INFO, "FunPack DynaShift", f"key {taste.key!r}: {summary}")

    captured = taste.collect(patcher, key, KIND, keep=4 * BANK, fresh=fresh)
    steer = input_steer.Steer("DynaShift")
    steer.attach(patcher, key)

    def apply_model(executor, x, t, *args, **kwargs):
        named = streams.model_args(args, kwargs)
        to = named.get("transformer_options")
        step = steer.begin(x, t, to)
        out = executor(step.x, t, *args, **kwargs)
        split = streams.video_of(out, named)
        if split is None:
            _say("off this run: could not find the picture in this model's latent")
            return out
        video, rebuild = split
        if dit_hooks.last_step(to):
            captured["latent"] = video[0].detach().half()
            cond = _pooled(named.get("c_crossattn"))
            if cond is not None:
                captured["cond"] = cond
        amount = strength * step.gate
        if amount <= 0.0:
            return out
        shifted = shift(video, live["bank"], amount, threshold, named.get("c_crossattn"))
        return out if shifted is None else step.keep(out, rebuild(shifted))

    patcher.add_wrapper_with_key(WrappersMP.APPLY_MODEL, key, apply_model)
    if strength <= 0.0:
        return "strength 0: learning only"
    return (f"strength {strength:g}, threshold {threshold:g}, last half of steps; "
            "banks every run, steers once something is disliked (read fresh every run)")


PROVIDES = {"modifier": install}
