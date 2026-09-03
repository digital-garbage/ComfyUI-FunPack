"""Steering that can reach the first half of a generation.

Every rating-driven mechanism in the Scene Chain sampler shares one gate,
`samplers._make_steer_ramp`, which opens only over the LAST HALF of the schedule. Motion
and layout are decided in the half that gate never reaches, so a rating about movement has
never had a window in which it could act.

`trajectory_probe` measured whether the early half carries anything to act on. Across two
independent sessions (different prompts, different label balances, 51 rated runs) the early
window measured 88% and 89% of the late window's effect — the late window being where
`output_guidance` already works. Nothing reached the strict significance bar on its own;
the direction was unanimous across all 8 bucket-by-session results, and the early/late
RATIO is what this is built on, because that comparison is paired within a session and far
better powered than either measurement against zero.

So: one value function per schedule bucket instead of one for the whole run, each trained
on that bucket's own descriptor, each steering inside its own window.

WHAT THIS IS NOT. It does not replace `output_guidance`, which stays exactly as it is: that
one is shipped, and the honest way to find out whether this helps is to run the same seed
with it off and on. If nothing changes, the answer is no, and a mechanism that cannot be
told apart from its own absence is the failure this project keeps finding.

The descriptors come from the probe's log, which means this starts with whatever has
already been rated rather than needing its own collection pass.
"""

from __future__ import annotations

import os

import torch


def _lvf():
    try:
        from .value_function import LatentValueFunction
    except ImportError:
        from value_function import LatentValueFunction
    return LatentValueFunction


def _probe():
    try:
        from . import trajectory_probe as tp
    except ImportError:
        import trajectory_probe as tp
    return tp


def state_path(refinement_key):
    try:
        from .conditioning import refinement_state_path
    except ImportError:
        from conditioning import refinement_state_path
    return refinement_state_path(refinement_key, "value_fn_buckets", prefix="refine_v2",
                                 extension="pt")


class BucketedValue:
    """One value function per schedule bucket.

    Deliberately a container around `LatentValueFunction` rather than a new model: the
    per-bucket claim is about WHERE a value is learned, not about what kind of thing learns
    it, and a second architecture would make a difference in results impossible to
    attribute.
    """

    def __init__(self, n_buckets, hidden_dim=None):
        self.n_buckets = int(n_buckets)
        self.hidden_dim = int(hidden_dim or _lvf().DEFAULT_HIDDEN_DIM)
        self.heads = {}

    def _head(self, bucket):
        bucket = int(bucket)
        if bucket not in self.heads:
            self.heads[bucket] = _lvf()(hidden_dim=self.hidden_dim)
        return self.heads[bucket]

    def train_on(self, bucket, descriptor, reward):
        if descriptor is None:
            return None
        head = self._head(bucket)
        head.train_on(descriptor, float(reward))
        return head.n_trained

    def ready(self, bucket):
        head = self.heads.get(int(bucket))
        return bool(head and head.is_ready())

    def gradient(self, bucket, x):
        head = self.heads.get(int(bucket))
        return None if head is None else head.gradient(x)

    def trained(self):
        """{bucket: samples}, so a run can say which windows can actually steer. A bucket
        with too few samples is not an error and not a failure -- it simply does not act,
        and saying which ones do is the difference between 'on' and 'doing something'."""
        return {b: h.n_trained for b, h in sorted(self.heads.items())}

    # --- persistence ------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "version": 1,
            "n_buckets": self.n_buckets,
            "hidden_dim": self.hidden_dim,
            "heads": {b: {"state": h.state_dict(),
                          "buffer_c": h.buffer_c, "buffer_r": h.buffer_r,
                          "n_trained": h.n_trained}
                      for b, h in self.heads.items()},
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, n_buckets=4):
        if not os.path.exists(path):
            return cls(n_buckets)
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return cls(n_buckets)
        if not isinstance(data, dict):
            return cls(n_buckets)
        out = cls(int(data.get("n_buckets") or n_buckets), data.get("hidden_dim"))
        for bucket, blob in (data.get("heads") or {}).items():
            try:
                head = out._head(int(bucket))
                head.load_state_dict(blob["state"])
                head.buffer_c = list(blob.get("buffer_c") or [])
                head.buffer_r = list(blob.get("buffer_r") or [])
                head.n_trained = int(blob.get("n_trained") or 0)
            except Exception:
                out.heads.pop(int(bucket), None)
        return out


def _train_all(model, rows):
    """Every rated descriptor into the head for its own bucket."""
    for row in rows:
        per_scene = row.get("scene_rewards") or None
        for cell in row["rows"]:
            # Pass 0 only: a second pass runs its own schedule from its own starting sigma,
            # so its bucket 2 is not the window bucket 2 steers in.
            if int(cell.get("pass", 0)) != 0:
                continue
            reward = (per_scene or {}).get(int(cell.get("scene", 0)))
            if reward is None:
                reward = row.get("reward")
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                continue
            model.train_on(cell.get("bucket", 0), cell["desc"].float(), reward)


def train_from_rows(refinement_key, rows, n_buckets=None):
    """Rebuild the per-bucket heads from probe rows. -> {bucket: samples}, or None.

    Rebuilt rather than updated in place: the probe's log is the record of what was rated,
    a head is a pure function of it, and rebuilding means a log carried from another box
    trains the same heads it would have trained there. It is a few hundred parameters over
    a few dozen rows -- cheap enough that being able to reason about it is worth more than
    the saved cycles.
    """
    if not refinement_key:
        return None
    tp = _probe()
    usable = [r for r in rows if isinstance(r, dict) and r.get("rows")]
    if not usable:
        return None
    n_buckets = int(n_buckets or max(
        (c.get("bucket", 0) for r in usable for c in r["rows"]), default=-1) + 1)
    if n_buckets < 1:
        return None

    model = BucketedValue(n_buckets)
    # ComfyUI executes a node under torch.inference_mode(), so every tensor made in here is
    # an inference tensor and autograd refuses it ("Inference tensors do not track version
    # counter"). Same guard the output value function's trainer carries. It has to wrap the
    # whole loop, not one call: `.float()` on a stored descriptor makes a new tensor too, and
    # that is the one training receives.
    with torch.inference_mode(False), torch.enable_grad():
        _train_all(model, usable)

    try:
        model.save(state_path(refinement_key))
    except OSError as e:
        tp._failed("trajectory guidance", "save", e, "this run's learning is not kept")
    return model.trained()


def load(refinement_key, n_buckets=4):
    """The heads for a key, or None when nothing has been rated into them yet."""
    if not refinement_key:
        return None
    try:
        model = BucketedValue.load(state_path(refinement_key), n_buckets)
    except Exception:
        return None
    return model if any(model.ready(b) for b in range(model.n_buckets)) else None
