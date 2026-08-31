"""Per-step trajectory probe: does a run's fate show up BEFORE the halfway point?

Every rating-driven mechanism in the sampler (embed guidance, score slider, DynaShift,
output guidance) shares one gate — `samplers._make_steer_ramp` — which opens only over the
last half of the schedule. Motion and layout are decided in the half that gate never
reaches, so a "Missing action" rating currently has no window in which it could act. The
proposed fix is to bank the model's own x0_hat at several schedule positions and learn a
value per position, so steering can reach the early half.

That fix is only worth building if the early half carries a usable signal at all. This
module answers exactly that question and nothing else:

    Do runs the user rated well and runs the user rated badly separate, in x0_hat space,
    at EARLY schedule positions — by more than label-shuffling alone would produce?

It is instrumentation, not a feature. It records, it never steers, and it is off unless
``FUNPACK_TRAJECTORY_PROBE=1`` is set in the environment — so it stays out of the sampler
UI and out of every user's run. What it records per step is a 512-float pooled descriptor
of the predicted output, the same `compress_packed_latent` compression the output value
function is already trained on, so a positive result feeds straight into the existing
`LatentValueFunction` machinery.

BUILT FOR H3, which sets three of the design choices here:

- **Few steps.** H3 runs at 4 (turbo), 7 (bong_tangent + euler_a) or 12. Buckets are
  clamped to the step count, so a 4-step run measures 4 buckets of one step each rather
  than reporting empty ones. On H3's default 12-step shift-12 schedule, buckets 0 and 1 of
  4 are precisely the window `_make_steer_ramp` leaves unreachable.
- **CFG is always 1.0**, so comfy passes one row through the model per step and a bucket
  mean is a mean over steps. At CFG>1 comfy stacks the conditional and unconditional rows
  into ONE call's batch dimension, and `compress_packed_latent` averages over batch — so
  the descriptor would be a blend of a prediction and its negation. Rather than guess which
  row is which, the observer refuses any prediction with a batch above 1 and counts it, so
  such a run reports an incomplete measurement instead of a contaminated one. The same
  refusal covers batched generation, where one descriptor cannot describe several videos.
- **The schedule is generated from a shift**, not hand-authored, so bucket 2 of a shift-3
  run and bucket 2 of a shift-12 run cover different sigma ranges. Each row records its
  step count and both shifts, and the analysis refuses to pool across different schedules
  without saying so.

WHY x0_hat AND NOT x_t: at mid sigma, x_t is dominated by that run's own seed noise, so
banking it and pulling a later run toward it partially copies the old seed — the
static-character attractor `value_function` was rewritten to avoid. x0_hat is the model's
guess at the finish: blurry early, but seed-independent in the part that matters.

WHERE THE FILES LIVE: `refinements/trajectory_probe/`, deliberately NOT the `<key>.*`
sidecar convention. Refinement keys are disposable and get cleared between rentals; this
is accumulated measurement data whose whole value is surviving that. Nothing steers off
it, so it cannot cause the orphaned-sidecar drift the sidecar convention exists to
prevent.
"""

import os
import random
import re
import time

import torch

#: Schedule positions to bucket into. 4 splits the run into quarters: the first two are the
#: window no rating-driven mechanism can currently reach.
DEFAULT_BUCKETS = 4

#: Descriptor width. Matches LatentValueFunction.DEFAULT_HIDDEN_DIM so a positive result can
#: train the existing value function without re-recording anything.
DESCRIPTOR_DIM = 512

#: Rows per log file. Large enough for many rentals' worth of ratings, bounded so an
#: always-on probe cannot grow without limit.
MAX_ROWS = 512


def probe_enabled():
    """True when the probe should record. Off by default: this is an experiment, and an
    experiment does not belong in the sampler's UI."""
    return os.environ.get("FUNPACK_TRAJECTORY_PROBE", "").strip().lower() in ("1", "true", "yes", "on")


def bucket_count():
    try:
        n = int(os.environ.get("FUNPACK_TRAJECTORY_BUCKETS", "").strip() or DEFAULT_BUCKETS)
    except ValueError:
        return DEFAULT_BUCKETS
    return n if 2 <= n <= 16 else DEFAULT_BUCKETS


def _probe_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "refinements", "trajectory_probe")


def _safe_key(refinement_key):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(refinement_key or "default").strip())
    return (safe or "default")[:80]


def log_path(refinement_key):
    return os.path.join(_probe_dir(), f"{_safe_key(refinement_key)}.trajectory.pt")


def pending_path(refinement_key):
    return os.path.join(_probe_dir(), f"{_safe_key(refinement_key)}.pending.pt")


def _atomic_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # a concurrent read never sees a partial write


# --- recording ------------------------------------------------------------------


def schedule_positions(sigmas):
    """The schedule's sigma values as a plain descending list, or None.

    Bucketing reads POSITION on this list, not the absolute sigma value, for the reason
    `_make_steer_ramp` was rewritten: on H3 a large shift keeps sigma high right up to the
    final step, so absolute sigma says nothing about how far through the run we are.
    """
    if sigmas is None:
        return None
    try:
        vals = [float(v) for v in sigmas.flatten().tolist()]
    except Exception:
        return None
    return vals if len(vals) >= 2 else None


class TrajectoryRecorder:
    """Accumulates pooled x0_hat descriptors per (scene, schedule bucket) for one run.

    One recorder spans the whole run; the sampler installs a fresh observer wrapper per
    scene and tells the recorder which scene it is, so a multi-scene chain does not average
    scene 1's opening across scene 3's ending.
    """

    def __init__(self, sigmas, n_buckets=None, dim=DESCRIPTOR_DIM):
        self.requested_buckets = int(n_buckets or bucket_count())
        self.dim = int(dim)
        self.scene = 0
        self.pass_index = 0
        # (scene, pass, bucket) -> {"sum": [dim], "steps": int, "sigma_hi", "sigma_lo"}
        self.cells = {}
        self._seen = set()  # (scene, pass, sigma) already recorded — see observe()
        self.pass_steps = {}  # pass index -> step count of the schedule it ran
        self._foreign = 0     # steps refused because they belong to no bound schedule
        self.begin_pass(sigmas, index=0)

    def begin_pass(self, sigmas, index=None):
        """Bind the schedule the next steps will be sampled on.

        A scene can run the sampler more than once — a second pass runs its own schedule IN
        FULL, from its own starting sigma — and the observer wrapper stays installed across
        both. Sigmas from a second schedule have no defined position on the first, so each
        pass is bound and bucketed separately rather than being nearest-matched into the
        previous pass's buckets, which would silently average two different denoisings into
        one measurement.
        """
        self.values = schedule_positions(sigmas)
        self.steps = max(0, len(self.values) - 1) if self.values else 0
        # H3 runs at 4-12 steps. More buckets than steps would report empty ones as if they
        # had been measured, so the request is a ceiling, not a promise.
        self.n_buckets = max(1, min(self.requested_buckets, self.steps or 1))
        self.pass_index = self.pass_index + 1 if index is None else int(index)
        self.pass_steps[self.pass_index] = self.steps

    def begin_scene(self, scene_index):
        self.scene = int(scene_index)

    #: A sigma has to BE one of the bound schedule's values, not merely the closest one to
    #: it. Without that, a step from an unbound schedule lands in whichever bucket it is
    #: nearest and is counted as if it had been measured there. The tolerance covers the
    #: float32 round-trip through the model call, nothing wider.
    SIGMA_TOLERANCE = 1e-4

    def bucket_for(self, sigma):
        """Which quarter (or n-th) of the BOUND schedule this sigma sits in, or None.

        None means "this step is not on the schedule I was told about" — a step from a pass
        nobody bound, which must be refused rather than nearest-matched. The final entry of
        a sigma list is the terminal 0 and is not a step, so positions are taken over the n
        steps, not the n+1 boundaries.
        """
        vals = self.values
        if not vals:
            return None
        n = len(vals) - 1
        if n < 1:
            return None
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            return None
        k = min(range(n), key=lambda i: abs(vals[i] - sigma))
        if abs(vals[k] - sigma) > self.SIGMA_TOLERANCE:
            return None
        return min(self.n_buckets - 1, (k * self.n_buckets) // n)

    def observe(self, sigma, descriptor):
        """Fold one step's descriptor into its bucket. Cheap by construction: the caller
        has already pooled the latent down to `dim` floats.

        Only the FIRST call at a given sigma is kept, so a sampler that evaluates a sigma
        more than once contributes it once. (This is not what guards against CFG — comfy
        stacks the unconditional row into the same call's batch instead of calling again;
        see note_unmeasurable and the module docstring.)
        """
        if descriptor is None:
            return False
        bucket = self.bucket_for(sigma)
        if bucket is None:
            # Either no schedule is bound, or this step belongs to one that was never bound.
            # Counted so the run can say the measurement is incomplete instead of quietly
            # logging a partial one.
            self._foreign += 1
            return False
        mark = (self.scene, self.pass_index, round(float(sigma), 6))
        if mark in self._seen:
            return False
        self._seen.add(mark)
        d = descriptor.detach().float().reshape(-1).cpu()
        if d.numel() != self.dim:
            return False
        cell = self.cells.get((self.scene, self.pass_index, bucket))
        if cell is None:
            self.cells[(self.scene, self.pass_index, bucket)] = {
                "sum": d, "steps": 1, "sigma_hi": float(sigma), "sigma_lo": float(sigma),
            }
            return True
        cell["sum"] = cell["sum"] + d
        cell["steps"] += 1
        cell["sigma_hi"] = max(cell["sigma_hi"], float(sigma))
        cell["sigma_lo"] = min(cell["sigma_lo"], float(sigma))
        return True

    def note_unmeasurable(self):
        """Count a step that was reached but could not be measured honestly. Kept in the
        same tally as steps from an unbound schedule: both mean the saved run was measured
        from fewer steps than it actually sampled, which the analysis has to be able to say.
        """
        self._foreign += 1
        return False

    def cell_rows(self):
        """Per-cell means, ordered by (scene, pass, bucket). fp16 — these are descriptors
        for a distance test, not values anything is reconstructed from."""
        rows = []
        for (scene, pass_index, bucket) in sorted(self.cells.keys()):
            cell = self.cells[(scene, pass_index, bucket)]
            mean = (cell["sum"] / max(1, cell["steps"])).to(torch.float16)
            rows.append({
                "scene": scene,
                "pass": pass_index,
                "bucket": bucket,
                "desc": mean,
                "steps": cell["steps"],
                "sigma_hi": cell["sigma_hi"],
                "sigma_lo": cell["sigma_lo"],
            })
        return rows

    def is_empty(self):
        return not self.cells


def save_pending(refinement_key, recorder, prompt_hash=None, seed=None, meta=None):
    """Persist this run's per-bucket descriptors as the candidate the NEXT rating judges.

    One pending per key, overwritten by EVERY run — the same pairing rule DynaShift's
    pending latent follows, so a rating always scores the run that produced it.

    A run that measured nothing still overwrites, with no rows. Skipping the write instead
    would leave the previous run's pending in place for the next rating to consume, which
    logs one run's descriptors under another run's rating: silent, and precisely the kind of
    mislabelled row that would make the whole measurement lie.
    """
    if not refinement_key or recorder is None:
        return False
    try:
        _atomic_save({
            "version": 1,
            "rows": recorder.cell_rows(),
            "n_buckets": recorder.n_buckets,
            # Pass 0's step count: schedule_id compares runs, and a run is comparable to
            # another by the schedule its measured pass ran on.
            "steps": recorder.pass_steps.get(0, recorder.steps),
            "pass_steps": dict(recorder.pass_steps),
            "unbound_steps": recorder._foreign,
            "prompt_hash": prompt_hash,
            "seed": None if seed is None else int(seed),
            "meta": dict(meta or {}),
            "stamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        }, pending_path(refinement_key))
        return True
    except Exception as e:
        _failed("trajectory probe", "pending save", e, "this run will not enter the measurement")
        return False


def _failed(tag, what, error, effect):
    try:
        try:
            from .funpack_log import failed
        except ImportError:
            from funpack_log import failed
        failed(tag, what, error, effect)
    except Exception:
        print(f"[FunPack {tag}] {what} failed ({error}) — {effect}")


def commit(refinement_key, reward, rating_key=None, axes=None, scene_rewards=None):
    """Pair the pending descriptors with the rating that scores their run and append the
    row to the log. Always removes the pending file, so a stale candidate can never be
    paired with a later, unrelated rating. A pending with no rows (a run that measured
    nothing) is consumed and logs nothing. Returns the log length, or None."""
    if not refinement_key:
        return None
    path = pending_path(refinement_key)
    if not os.path.exists(path):
        return None
    count = None
    try:
        pending = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(pending, dict) and pending.get("rows"):
            pending["reward"] = float(reward)
            pending["rating"] = None if rating_key is None else str(rating_key)
            pending["axes"] = dict(axes or {})
            # Per-scene rewards when the user rated each scene separately. Without them a
            # multi-scene run collapses to ONE reward covering every scene's descriptors, so
            # a scene rated Perfect inside a run whose worst scene was Awful is filed as a
            # bad sample — the two labels the measurement is trying to tell apart, mixed.
            if scene_rewards:
                pending["scene_rewards"] = {int(k): float(v) for k, v in scene_rewards.items()}
            rows = load_log(refinement_key)
            rows.append(pending)
            rows = rows[-MAX_ROWS:]
            _atomic_save({"version": 1, "rows": rows}, log_path(refinement_key))
            count = len(rows)
    except Exception as e:
        _failed("trajectory probe", "log append", e, "this rating is missing from the measurement")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return count


def load_log(refinement_key):
    """Recorded runs (oldest first), or [] when this key has none yet."""
    path = log_path(refinement_key)
    if not os.path.exists(path):
        return []
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        rows = data.get("rows") if isinstance(data, dict) else None
        return [r for r in (rows or []) if isinstance(r, dict) and r.get("rows")]
    except Exception as e:
        _failed("trajectory probe", "log load", e, "the measurement reads as empty")
        return []


def load_all_logs():
    """Every recorded run across every key, as (key, rows) pairs — the probe is meant to
    accumulate across disposable keys, so the analysis pools them by default."""
    out = []
    directory = _probe_dir()
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".trajectory.pt"):
            continue
        key = name[: -len(".trajectory.pt")]
        out.append((key, load_log(key)))
    return out


def prompt_hash(conditioning):
    """A stable id for "the same prompt", so the analysis can ask whether a separation is
    really about ratings rather than about which prompt was running."""
    if not isinstance(conditioning, torch.Tensor):
        return None
    try:
        from hashlib import md5
        c = conditioning.detach().float()
        while c.dim() > 1:
            c = c.mean(dim=0)
        return md5(c.to(torch.float16).cpu().numpy().tobytes()).hexdigest()[:16]
    except Exception:
        return None


# --- analysis -------------------------------------------------------------------
#
# The question is whether ratings separate descriptors at a bucket. With a handful of runs,
# any statistic will look impressive on noise, so the null distribution is built from the
# data itself: shuffle the rating labels and recompute. If the real labels do not beat their
# own shuffles, the bucket carries nothing and the early-window idea is dead there.


def _unit(vectors):
    m = torch.stack([v.detach().float().reshape(-1) for v in vectors])
    return torch.nn.functional.normalize(m, dim=-1)


def separation_statistic(descriptors, labels, groups=None):
    """Mean between-label cosine distance minus mean within-label cosine distance.

    Positive means runs rated differently look more different than runs rated the same —
    the thing that has to be true for a per-bucket value function to have anything to learn.
    `groups` (e.g. prompt hashes) restricts every pair to within one group, so "these are
    different prompts" cannot masquerade as "these are different ratings". Returns None when
    no admissible pair exists.
    """
    n = len(labels)
    if n < 3 or len(descriptors) != n:
        return None
    units = _unit(descriptors)
    dist = 1.0 - (units @ units.T)
    between, within = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if groups is not None and groups[i] != groups[j]:
                continue
            (between if labels[i] != labels[j] else within).append(float(dist[i, j]))
    if not between or not within:
        return None
    return sum(between) / len(between) - sum(within) / len(within)


def permutation_test(descriptors, labels, groups=None, trials=2000, seed=0):
    """Observed separation, its p-value under label shuffling, and the within-label mean
    distance (the run-to-run noise floor at this bucket).

    Labels are shuffled WITHIN each group when groups are given, so the stratification used
    for the statistic is preserved in the null — an unstratified shuffle against a stratified
    statistic would answer a different question and flatter the result.
    """
    observed = separation_statistic(descriptors, labels, groups)
    if observed is None:
        return None
    rng = random.Random(seed)
    if groups is None:
        pools = {None: list(range(len(labels)))}
    else:
        pools = {}
        for i, g in enumerate(groups):
            pools.setdefault(g, []).append(i)
    hits = 0
    for _ in range(trials):
        shuffled = list(labels)
        for idxs in pools.values():
            picked = [labels[i] for i in idxs]
            rng.shuffle(picked)
            for i, lab in zip(idxs, picked):
                shuffled[i] = lab
        stat = separation_statistic(descriptors, shuffled, groups)
        if stat is not None and stat >= observed:
            hits += 1
    units = _unit(descriptors)
    dist = 1.0 - (units @ units.T)
    same = [float(dist[i, j])
            for i in range(len(labels)) for j in range(i + 1, len(labels))
            if labels[i] == labels[j] and (groups is None or groups[i] == groups[j])]
    return {
        "separation": observed,
        "p_value": (1 + hits) / (1 + trials),
        "noise_floor": (sum(same) / len(same)) if same else None,
        "n": len(labels),
    }


def collect_bucket(rows, bucket, good_above=0.0, bad_below=0.0, scene=None, pass_index=0):
    """(descriptors, labels, prompt groups) for one bucket across recorded runs.

    Runs whose reward sits between the two thresholds are dropped rather than assigned to a
    side: "Forget it" carries reward 0.0 and is explicitly not a judgement, and a
    near-neutral rating is not evidence for either label.

    Only the first sampling pass is read by default. A second pass runs its own schedule
    from its own starting sigma, so its bucket 1 is not the same window as pass 1's — they
    are separate measurements, not more samples of one.
    """
    def _label(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value > good_above:
            return 1
        if value < bad_below:
            return 0
        return None

    descriptors, labels, groups = [], [], []
    for row in rows:
        picked = [c for c in row.get("rows", [])
                  if c.get("bucket") == bucket
                  and c.get("pass", 0) == pass_index
                  and (scene is None or c.get("scene") == scene)]
        if not picked:
            continue
        per_scene = row.get("scene_rewards") or None
        if per_scene:
            # Each rated scene is its own labelled sample: one aggregate reward spread over
            # every scene would file a Perfect scene as a bad one whenever another scene in
            # the same run was rated badly.
            for cell in picked:
                label = _label(per_scene.get(int(cell.get("scene", 0))))
                if label is None:
                    continue
                descriptors.append(cell["desc"].float())
                labels.append(label)
                groups.append(row.get("prompt_hash"))
            continue
        label = _label(row.get("reward"))
        if label is None:
            continue
        descriptors.append(torch.stack([c["desc"].float() for c in picked]).mean(dim=0))
        labels.append(label)
        groups.append(row.get("prompt_hash"))
    return descriptors, labels, groups


def schedule_id(row):
    """"Which schedule was this run sampled on" — step count plus H3's two flow shifts.

    Bucket 2 of a 4-step shift-12 run and bucket 2 of a 12-step shift-3 run cover different
    sigma ranges, so pooling them compares different windows and calls the result a rating
    effect. The analysis reports the mix rather than silently averaging it away.
    """
    meta = row.get("meta") if isinstance(row, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    return (row.get("steps") if isinstance(row, dict) else None,
            meta.get("shift_video"), meta.get("shift_audio"))


def schedule_mix(rows):
    """schedule_id -> run count, so a report can say what it is pooling."""
    mix = {}
    for row in rows:
        mix[schedule_id(row)] = mix.get(schedule_id(row), 0) + 1
    return mix


def analyse(rows, n_buckets=None, trials=2000, good_above=0.0, bad_below=0.0, pass_index=0):
    """Per-bucket verdicts across recorded runs, pooled and (when prompts repeat) stratified
    by prompt. Buckets are ordered earliest-first: bucket 0 is the window nothing currently
    steers in, and it is the one the whole proposal depends on."""
    if n_buckets is None:
        n_buckets = max((c.get("bucket", 0) for r in rows for c in r.get("rows", [])
                         if c.get("pass", 0) == pass_index), default=-1) + 1
    results = []
    for bucket in range(int(n_buckets)):
        descriptors, labels, groups = collect_bucket(
            rows, bucket, good_above, bad_below, pass_index=pass_index)
        entry = {"bucket": bucket, "n": len(labels),
                 "n_good": sum(labels), "n_bad": len(labels) - sum(labels)}
        entry["pooled"] = permutation_test(descriptors, labels, None, trials=trials)
        stratified = None
        if groups and all(g is not None for g in groups) and len(set(groups)) > 1:
            stratified = permutation_test(descriptors, labels, groups, trials=trials)
        entry["within_prompt"] = stratified
        results.append(entry)
    return results
