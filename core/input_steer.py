"""Carry a step's edit into the NEXT step's input, never onto its answer (H3).

On a few-step schedule the last answer IS the video: an edit to it lands in the
output with no later step to clean it, and on 4 steps that shows as grain
(simple@4's last step starts at sigma 0.80). So a wrapper that wants to edit
step i's predicted picture hands the edit here instead. The model's own answer
is returned untouched, and the edit is added to step i+1's input scaled to that
step's noise level -- a rectified-flow input is (1 - sigma) * picture + sigma *
noise -- so the model cleans the push like the rest of its input. The last step
computes nothing.

    step = steer.begin(x, t, transformer_options)
    out = executor(step.x, t, ...)            # step.x = x plus the previous push
    if step.gate > 0: out = step.keep(out, edited)

Packed models only (the H3 shape, a 3-D [B,1,N] latent). A plain 5-D latent
keeps editing the answer as before: `keep` returns the edit, `gate` is the
ordinary late-half gate. When an LTX module lands it decides for itself.

Same-step calls (a split batch, a context window) never receive a push made in
that step: pushes are filed by the step that made them. A schedule that restarts
(index goes backwards) drops everything held.
"""

import torch

from . import dit_hooks, log

# Below this share of the input it joins, a push is reported as barely acting (v4: 3%).
FLOOR = 0.03



class Call:
    """One model call: the input to run, the gate to use, and where to put the edit.
    `steering` is False for a call that will never take an edit (a probe, off the schedule,
    the sampler declared unsupported); `final` is the call whose answer is the output."""

    def __init__(self, x, gate, keep=None, final=False, steering=False):
        self.x, self.gate, self.final, self.steering = x, gate, final, steering
        self._keep = keep

    def keep(self, out, steered):
        """File `steered - out` for the next step and return the model's own answer."""
        return steered if self._keep is None else self._keep(out, steered)


class Steer:
    def __init__(self, what: str):
        self.what = what
        self.reset()

    def reset(self):
        """A run starts (or ends): nothing held, nothing counted."""
        self._pushes = {}       # key -> {step: delta}
        self._seen = {}         # key -> last step index fed
        self._runs = getattr(self, "_runs", 0) + 1     # a result line is said once per RUN
        self._multi = False     # the sampler calls the model more than once per step
        self._made, self._delivered = 0, 0
        self._reach = 0.0       # largest delivered push, as a share of the input it joined

    def felt(self):
        """Whether this run's pushes were big enough to be felt (and so worth learning from)."""
        return self._delivered > 0 and self._reach >= FLOOR

    def effect(self):
        """How hard this run's pushes landed (largest, as a share of the input they joined),
        or None when none arrived. What a rating is weighted by."""
        return self._reach if self._delivered else None

    def attach(self, patcher, key):
        """Reset around every sampling call. The node that installs the modifier is cached
        by ComfyUI, so this object outlives a run; without this a run that made no edit
        would report the previous run's."""
        from comfy.patcher_extension import WrappersMP

        def outer(executor, *args, **kwargs):
            self.reset()
            try:
                return executor(*args, **kwargs)
            finally:
                self._pushes = {}           # latent-sized tensors must not outlive the run

        patcher.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, key, outer)

    def _say(self, level, message, key):
        log.once(f"input steer {self.what}:{key}:{self._runs}", level, f"FunPack {self.what}", message)

    @staticmethod
    def _packed(x):
        return getattr(x, "dim", None) is not None and x.dim() == 3

    def _key(self, to):
        win = to.get("context_window")
        return (tuple(win.index_list) if win is not None and hasattr(win, "index_list") else None,
                tuple(int(v) for v in (to.get("cond_or_uncond") or ())))

    def begin(self, x, t, to) -> Call:
        to = to or {}
        if not self._packed(x):
            self._say(log.INFO, "this latent is not packed (not H3), so edits land on the model's "
                                "own answer, as before; the last step is edited too", "plain latent")
            return Call(x, dit_hooks.late_half(to), steering=True)
        if dit_hooks.probing(to):
            return Call(x, 0.0, keep=lambda out, _steered: out)   # a discarded candidate
        where = dit_hooks.current_step(to)
        if where is None:
            self._say(log.ALERT, "this call is not a step of the schedule (a midpoint call), so "
                                 "it is left unsteered; other calls still are", "off schedule")
            return Call(x, 0.0, keep=lambda out, _steered: out)
        i, n = where
        if self._multi:
            return Call(x, 0.0, keep=lambda out, _steered: out)
        sched = to["sample_sigmas"]
        if len(torch.unique(sched[:-1])) < n:
            # current_step finds a step by its sigma: a repeated one is two steps with one name.
            self._multi = True
            self._say(log.ALERT, "Inactive | this schedule repeats a sigma value, so steps cannot "
                                 "be told apart and edits are not carried; use a schedule with "
                                 "distinct sigmas", "repeated sigma")
            return Call(x, 0.0, keep=lambda out, _steered: out)
        if i == self._seen.get(self._key(to)) and n > 1:
            # The same step index twice in a row for one conditioning: a corrector call
            # (heun, dpm_2 ...) or a split batch. A push cannot be placed on the right call,
            # so it is dropped and nothing is steered for the rest of the run.
            self._multi = True
            self._pushes = {}
            self._say(log.ALERT, "Inactive | the model is called more than once per step "
                                 "(a second-order sampler, or a conditioning split across "
                                 "calls), so edits are not carried; use euler-style sampling", "multi call")
            return Call(x, 0.0, keep=lambda out, _steered: out)
        if n <= 1:
            self._say(log.ALERT, "Inactive | a 1-step schedule has no step before the output "
                                 "to carry an edit into", "one step")
        key = self._key(to)
        held = self._pushes.setdefault(key, {})
        if i < self._seen.get(key, 0):      # a new sampling pass: nothing carries over
            held.clear()
        self._seen[key] = i
        push = held.pop(i - 1, None)       # delivered once; a repeated sigma cannot re-deliver it
        if push is not None and to.get(dit_hooks.FRAME_CHANGE):
            push = None                    # filed in the frame before the latent was moved
        if push is not None:
            if tuple(push.shape) == tuple(x.shape):
                add = (1.0 - float(t.max())) * push.to(x.device, x.dtype)
                self._delivered += 1
                self._reach = max(self._reach, float(add.norm() / x.norm().clamp(min=1e-8)))
                x = x + add
            else:
                self._say(log.WARNING, "Inactive | the latent changed size between steps, "
                                       "so that step went unsteered", "resized")
        for old in [s for s in held if s < i - 1]:
            del held[old]
        if i >= n - 1:                      # the answer is the output: nothing after it
            self._report()
            return Call(x, 0.0, final=True, keep=lambda out, _steered: out)

        def keep(out, steered):
            self._made += 1
            held[i] = (steered - out).detach()
            return out

        return Call(x, dit_hooks.late_half(to, ahead=1), keep=keep, steering=True)

    def _report(self):
        """Said on the last step. Reach is what counts: an edit made is not an edit that
        arrived, and one that is a sliver of the input it joins changes nothing the
        model can feel."""
        if self._multi:
            return
        if not self._made:
            self._say(log.INFO, "Inactive | no step made an edit this run (nothing learned yet "
                                "to steer with, strength 0, or a schedule too short for the "
                                "late-step gate to open)", "result")
        elif not self._delivered:
            self._say(log.ALERT, f"Inactive | {self._made} edit(s) made but none reached a later "
                                 f"step's input", "result")
        elif self._reach < FLOOR:
            self._say(log.ALERT, f"Inactive | barely acts: the largest push was {self._reach:.1%} of "
                                 f"the input it joined (under {FLOOR:.0%}); more steps or a "
                                 f"higher strength would let it be felt", "result")
        else:
            self._say(log.INFO, f"Active | {self._delivered} push(es) carried into later steps, the "
                                f"largest {self._reach:.1%} of the input it joined", "result")
