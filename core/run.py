"""Which run this is.

Previews, progress, results and log lines all have to be attributable to one
generation, and ComfyUI gives a node no way to ask "which run am I in" -- there
is no hook for "a run started". So FunPack marks its own: the first FunPack node
to execute in a graph starts a run, and everything after it reports under that
id until the next one begins.

The id is short and readable rather than a uuid, because it is going to appear in
log lines a person is scanning: `run 7` is findable, a hex string is not.

Deliberately not a wall-clock timestamp. The name has to be stable to compare
two runs in a report, and identical prompts a second apart must not collide.
"""

import itertools
import threading
from typing import Optional

_counter = itertools.count(1)
_lock = threading.Lock()
_current: Optional[str] = None


def start() -> str:
    """Begin a run, and return its id.

    Also clears anything scoped to a run -- what has been said once, and which
    modifiers were dropped -- so the two cannot disagree about when a run began.
    """
    global _current
    with _lock:
        _current = f"run {next(_counter)}"
    from . import log
    log.new_run()
    return _current


def current() -> Optional[str]:
    """The run in progress, or None before anything has started one."""
    return _current


def _reset() -> None:
    """Tests only."""
    global _counter, _current
    _counter = itertools.count(1)
    _current = None
