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

from . import dit_hooks, log



class Call:
    """One model call: the input to run, the gate to use, and where to put the edit."""

    def __init__(self, x, gate, keep=None, final=False):
        self.x, self.gate, self.final = x, gate, final
        self._keep = keep

    def keep(self, out, steered):
        """File `steered - out` for the next step and return the model's own answer."""
        return steered if self._keep is None else self._keep(out, steered)


class Steer:
    def __init__(self, what: str):
        self.what = what
        self._pushes = {}       # key -> {step: delta}
        self._seen = {}         # key -> last step index fed
        self._edit, self._edits = 0.0, 0

    def effect(self):
        """Mean size of the edits made, relative to the answer, or None when none were."""
        return self._edit / self._edits if self._edits else None

    def _say(self, level, message, key):
        log.once(f"input steer {self.what}:{key}", level, f"FunPack {self.what}", message)

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
            return Call(x, dit_hooks.late_half(to))
        if dit_hooks.probing(to):
            return Call(x, 0.0, keep=lambda out, _steered: out)   # a discarded candidate
        where = dit_hooks.current_step(to)
        if where is None:
            self._say(log.ALERT, "Inactive | this call is not a step of the schedule "
                                 "(a midpoint sampler?), so it is not steered", "off schedule")
            return Call(x, 0.0, keep=lambda out, _steered: out)
        i, n = where
        if n <= 1:
            self._say(log.ALERT, "Inactive | a 1-step schedule has no step before the output "
                                 "to carry an edit into", "one step")
        key = self._key(to)
        held = self._pushes.setdefault(key, {})
        if i < self._seen.get(key, 0):      # a new sampling pass: nothing carries over
            held.clear()
        self._seen[key] = i
        push = held.get(i - 1)
        if push is not None:
            if tuple(push.shape) == tuple(x.shape):
                x = x + (1.0 - float(t.max())) * push.to(x.device, x.dtype)
            else:
                self._say(log.WARNING, "Inactive | the latent changed size between steps, "
                                       "so that step went unsteered", "resized")
        for old in [s for s in held if s < i - 1]:
            del held[old]
        if i >= n - 1:                      # the answer is the output: nothing after it
            self._report()
            return Call(x, 0.0, final=True, keep=lambda out, _steered: out)

        def keep(out, steered):
            try:
                self._edit += float((steered - out).norm() / out.norm().clamp(min=1e-8))
                self._edits += 1
            except Exception:               # noqa: BLE001 -- a measurement must never break sampling
                pass
            held[i] = (steered - out).detach()
            return out

        return Call(x, dit_hooks.late_half(to, ahead=1), keep=keep)

    def _report(self):
        size = self.effect()
        if size is None:
            self._say(log.ALERT, "Inactive | no step made an edit this run", "result")
        else:
            self._say(log.INFO, f"Active | edits averaged {size:.1%} of the picture's size, "
                                f"each carried into the next step's input", "result")
