"""FunPack's console voice.

Two rules that pull against each other until they share one mechanism:

1. **An operation that does not happen must say so.** Absorbing an exception and reporting it
   are separate decisions, and for most of this codebase only the first was ever made — a
   guide silently not applied, an anchor pin silently not restored, a taste direction silently
   not steered. The run survives, which is right, and nobody learns anything, which is not.

2. **The log has to stay readable.** A line that repeats identically on every run, or once per
   scene for eight scenes, is noise — and noise is what makes a log stop being read at all. A
   report nobody reads is worth exactly as much as no report.

So: `note` always speaks. `note_on_change` states a standing condition once and restates it
only when it stops being true — "this model is not LTX" is worth saying the first time and
worthless the twentieth. `failed` is the standard phrasing for rule 1, collapsed to once per
run so a per-step failure cannot bury everything else.
"""

import os
import re

_last_by_key: dict[str, str] = {}   # key -> the text that key last printed
_said_this_run: set[str] = set()    # keys already spoken since begin_run()

#: Compact by default. A log nobody reads reports nothing, and a paragraph per feature is
#: how a log stops being read. The reasoning is still written at the call sites — it is
#: worth having the first time someone hits a problem — so `FUNPACK_LOG=verbose` in the
#: environment (or `set_verbose(True)`) prints it in full.
_verbose = os.environ.get("FUNPACK_LOG", "").strip().lower() in ("verbose", "full", "1")

#: Where a message stops being the answer and starts being the explanation. SENTENCE ENDS
#: ONLY — an em-dash just as often introduces the reason as the essay, and a renderer that
#: guesses wrong eats the one part of the line worth reading. Anything that wants the
#: reason kept apart says so by calling `feature`.
_CUT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_MAX = 150


def set_verbose(on: bool) -> None:
    """Print the full text of every message, explanations included."""
    global _verbose
    _verbose = bool(on)


def verbose() -> bool:
    return _verbose


def compact(message: str) -> str:
    """The first clause of a message, which is the part that says WHAT happened.

    Everything FunPack logs is written as answer-then-reasoning, so cutting at the first
    sentence end keeps the answer and drops the essay. A message without one is hard-trimmed
    rather than left to run: the shape of the log matters more than any one line's
    completeness, and the full text is one environment variable away.
    """
    if _verbose:
        return message
    head = _CUT.split(str(message).strip(), 1)[0].strip()
    head = head.rstrip(" ,;:")
    if len(head) > _MAX:
        head = head[:_MAX].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
    return head


def begin_run() -> None:
    """Start a new generation. Clears the once-per-run set; leaves standing conditions alone,
    since those are about configuration and have not necessarily changed."""
    _said_this_run.clear()


def reset() -> None:
    """Forget everything, standing conditions included. For tests."""
    _last_by_key.clear()
    _said_this_run.clear()


def feature(tag: str, name: str, active: bool, reason: str = "") -> None:
    """The standard shape for a capability's state, and the only shape worth scanning:

        [FunPack] Region locks: Active
        [FunPack] Region locks: Inactive | This model is not MiniMax H3

    Say it once per run for anything the user switched on, whether or not it took. A feature
    that is on and silent is indistinguishable from one that is off.
    """
    state = "Active" if active else "Inactive"
    why = _reason(reason) if active else (_reason(reason) or "No reason given")
    note_on_change(f"{tag}:{name}", tag, f"{name}: {state}" + (f" | {why}" if why else ""))


def _reason(text: str) -> str:
    """One clause, starting with a capital. The reason is the half people read."""
    out = compact(text or "").strip().rstrip(".")
    return out[:1].upper() + out[1:] if out else ""


def note(tag: str, message: str) -> None:
    """Say it, every time. For events that genuinely differ run to run."""
    print(f"[{tag}] {compact(message)}")


def note_once(tag: str, message: str, key: str | None = None) -> bool:
    """Say it at most once per run. `key` defaults to the message itself, so a line that
    varies (a count, a scene number) speaks per distinct value unless you pin the key.

    Returns whether it printed, so a caller can branch without tracking state itself.
    """
    k = key if key is not None else f"{tag}:{message}"
    if k in _said_this_run:
        return False
    _said_this_run.add(k)
    print(f"[{tag}] {compact(message)}")
    return True


def note_on_change(key: str, tag: str, message: str) -> bool:
    """Say it only when what it would say has changed since this key last spoke.

    For standing conditions: model family, a missing backend, a feature inert because of how
    the graph is wired. Stated once, and again the moment the answer is different — which is
    the point at which it is news again.
    """
    if _last_by_key.get(key) == message:
        return False
    _last_by_key[key] = message
    print(f"[{tag}] {compact(message)}")
    return True


def failed(tag: str, what: str, error, effect: str, key: str | None = None) -> bool:
    """The standard shape for rule 1: an operation raised, the run continues without it.

    Three things a reader needs and a bare `except: pass` gives none of: what was attempted,
    why it stopped, and what the output looks like as a result. The third matters most — it
    is the difference between "ignore this" and "that explains what I am looking at".

    Collapsed to once per run per `key` (default: tag + what), because these fire inside
    per-step wrappers and twenty identical lines drown the report they belong to.
    """
    k = key if key is not None else f"{tag}:{what}"
    if k in _said_this_run:
        return False
    _said_this_run.add(k)
    # The EFFECT is the half that survives the trim, not the exception text: "that explains
    # what I am looking at" is what rule 1 exists for, and an exception message rarely says
    # it. The error itself is one FUNPACK_LOG=verbose away.
    detail = f" ({error})" if _verbose and error is not None and str(error) else ""
    print(f"[{tag}] {what}: Failed{detail}"
          + (f" | {_reason(effect)}" if effect else ""))
    return True


def skipped(tag: str, what: str, why: str, key: str | None = None) -> bool:
    """A deliberate no-op, as opposed to a failure: the feature was asked for and does not
    apply here. Once per run — the reason does not change mid-generation."""
    k = key if key is not None else f"{tag}:skip:{what}"
    if k in _said_this_run:
        return False
    _said_this_run.add(k)
    print(f"[{tag}] {what}: Inactive | {_reason(why) or 'No reason given'}")
    return True
