"""Saying what happened, at a severity, from somewhere.

    [Log] Info: FunPack Diffusion Model Loader | flux.safetensors loaded, attention sage
    [Log] Alert: FunPack Sampler | no image is connected, starting from noise
    [Log] Warning: FunPack LoRA Loader | style.safetensors is empty and will not load
    [Log] Critical: torch | cannot allocate 14 GB (3 GB free); the run will not start

Four levels, and the difference between the middle two is the one that matters:

* **Info** -- it worked, and here is what it actually did. Not decoration: "loaded
  on GPU using sage attention" is how you find out the attention setting did
  nothing, months before the pictures look wrong enough to investigate.
* **Alert** -- something is unset or unconnected and the run continues without
  it. A choice you may not have made on purpose.
* **Warning** -- something will NOT do what its name says. This is the level v4
  needed and did not have: 560 handlers swallowed a failure and carried on, so a
  feature could be switched on, do nothing, and never say so.
* **Critical** -- the run cannot proceed.

Records are kept in a small ring as well as printed, so the app can show them.
The buffer is bounded because a long rental session must not turn a log into a
memory leak.
"""

import sys
import time
from collections import deque
from typing import Optional

LABEL = "[Log]"

INFO = "Info"
ALERT = "Alert"
WARNING = "Warning"
CRITICAL = "Critical"

LEVELS = (INFO, ALERT, WARNING, CRITICAL)

# Enough to cover a long generation and its aftermath, small enough to be free.
HISTORY = 500
_records: deque = deque(maxlen=HISTORY)

# Keys already said once. Cleared per run rather than per process: v4's dedup
# lasted the life of the interpreter, so a session reported the first generation
# that went inert and stayed quiet for every one after it.
_said: set = set()


def line(level: str, source: str, message: str) -> str:
    """The one place the format is decided."""
    where = f"{source} | " if source else ""
    return f"{LABEL} {level}: {where}{message}"


def record(level: str, source: str, message: str) -> dict:
    # Imported here rather than at module level: `run` starts a run by clearing
    # what this module holds, so a top-level import would be circular.
    from . import run as run_mod
    entry = {"at": time.time(), "level": level, "source": source,
             "message": message, "run": run_mod.current()}
    _records.append(entry)
    print(line(level, source, message), file=sys.stderr, flush=True)
    return entry


def info(source: str, message: str) -> dict:
    return record(INFO, source, message)


def alert(source: str, message: str) -> dict:
    return record(ALERT, source, message)


def warning(source: str, message: str) -> dict:
    return record(WARNING, source, message)


def critical(source: str, message: str) -> dict:
    return record(CRITICAL, source, message)


def once(key: str, level: str, source: str, message: str) -> Optional[dict]:
    """Say it the first time only, until the next run.

    For anything that would otherwise repeat every sampling step. Thirty copies
    of one line say nothing the first did not, and burying the console is its own
    way of hiding a failure.
    """
    if key in _said:
        return None
    _said.add(key)
    return record(level, source, message)


def failed(what: str, exc: BaseException) -> dict:
    """Something did not load. Always a Warning: it means a feature is absent."""
    return warning(what, f"did not load -- {type(exc).__name__}: {exc}")


def broke(what: str, exc: BaseException, doing: Optional[str] = None) -> dict:
    """Something that DID load, and then failed while working.

    Kept apart from `failed` because the two send a reader to different places.
    "did not load" points at an import, a missing dependency, a module that was
    never there -- and someone who reads that about a provider which loaded,
    recognised the model and then raised goes looking for a failure that never
    happened. Said three times in three files before it was noticed, so it is one
    function now rather than three hand-written strings.
    """
    return warning(what, f"failed while {doing or 'working'} "
                         f"-- {type(exc).__name__}: {exc}")


def new_run() -> None:
    """A generation is starting: whatever was said once may be said again."""
    _said.clear()


def history(level: Optional[str] = None, limit: int = HISTORY,
            run: Optional[str] = None) -> list:
    """Recent records, newest last. For the app, and for the modules page."""
    items = [r for r in _records
             if (level is None or r["level"] == level)
             and (run is None or r.get("run") == run)]
    return items[-limit:]


def _reset() -> None:
    """Tests only."""
    _records.clear()
    _said.clear()
