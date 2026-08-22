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

_last_by_key: dict[str, str] = {}   # key -> the text that key last printed
_said_this_run: set[str] = set()    # keys already spoken since begin_run()


def begin_run() -> None:
    """Start a new generation. Clears the once-per-run set; leaves standing conditions alone,
    since those are about configuration and have not necessarily changed."""
    _said_this_run.clear()


def reset() -> None:
    """Forget everything, standing conditions included. For tests."""
    _last_by_key.clear()
    _said_this_run.clear()


def note(tag: str, message: str) -> None:
    """Say it, every time. For events that genuinely differ run to run."""
    print(f"[{tag}] {message}")


def note_once(tag: str, message: str, key: str | None = None) -> bool:
    """Say it at most once per run. `key` defaults to the message itself, so a line that
    varies (a count, a scene number) speaks per distinct value unless you pin the key.

    Returns whether it printed, so a caller can branch without tracking state itself.
    """
    k = key if key is not None else f"{tag}:{message}"
    if k in _said_this_run:
        return False
    _said_this_run.add(k)
    print(f"[{tag}] {message}")
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
    print(f"[{tag}] {message}")
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
    detail = f" ({error})" if error is not None and str(error) else ""
    print(f"[{tag}] {what} failed{detail} — {effect}")
    return True


def skipped(tag: str, what: str, why: str, key: str | None = None) -> bool:
    """A deliberate no-op, as opposed to a failure: the feature was asked for and does not
    apply here. Once per run — the reason does not change mid-generation."""
    k = key if key is not None else f"{tag}:skip:{what}"
    if k in _said_this_run:
        return False
    _said_this_run.add(k)
    print(f"[{tag}] {what} — SKIPPED: {why}")
    return True
