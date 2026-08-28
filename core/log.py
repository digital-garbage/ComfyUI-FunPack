"""One-line logging with a fixed prefix.

`failed()` exists so a module that silently no-ops is impossible: every swallowed
exception says what stopped working, once.
"""

import sys

PREFIX = "[FunPack]"


def note(message: str) -> None:
    print(f"{PREFIX} {message}", file=sys.stderr, flush=True)


def failed(what: str, exc: BaseException) -> None:
    note(f"{what} failed to load: {type(exc).__name__}: {exc}")
