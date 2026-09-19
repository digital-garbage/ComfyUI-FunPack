"""Make a silent death talk.

Two failures leave nothing in the log, and this project has hit both:

* a **crash** in native code (a CUDA kernel, a compiled extension, an ABI mismatch) kills the
  process with no Python traceback at all;
* a **hang** prints nothing by definition — the process is alive and busy or blocked, and
  there is no way to ask it where without attaching something.

`faulthandler` answers both. Enabling it installs handlers for SIGSEGV / SIGBUS / SIGFPE /
SIGILL / SIGABRT that dump every thread's Python stack to stderr before dying, so a segfault
names the line that reached it. Registering SIGUSR1 gives the same dump ON DEMAND: from
another terminal, `kill -USR1 <pid>` prints where a hung ComfyUI actually is, without
py-spy, without gdb, and without restarting it.

Costs nothing while nothing is wrong: the handlers sit unused, and the SIGUSR1 dump only runs
when a signal arrives. Off by default all the same, because installing signal handlers in
someone else's process should be a choice: set FUNPACK_FAULTHANDLER=1.
"""
import os
import sys

ENV_VAR = "FUNPACK_FAULTHANDLER"
_ON = {"1", "true", "yes", "on", "enable", "enabled"}


def wanted(value=None):
    text = str(os.environ.get(ENV_VAR) if value is None else value or "").strip().lower()
    return text in _ON


def enable(value=None, fh=None, sig=None):
    """Install the handlers. Returns a status line, or None when not asked for.

    Never raises: this runs at import, and a diagnostic that stops FunPack loading is worse
    than no diagnostic at all.
    """
    if not wanted(value):
        return None
    if fh is None:
        import faulthandler as fh
    try:
        fh.enable(file=sys.stderr, all_threads=True)
    except Exception as e:  # noqa: BLE001
        return f"faulthandler could not be enabled: {e}"
    note = "faulthandler on — a native crash will print a Python traceback instead of nothing"
    # SIGUSR1 is the useful half for a HANG. Unavailable on Windows, which is fine: the
    # crash handler above is the part that matters there.
    if sig is None:
        try:
            import signal as sig
        except Exception:  # noqa: BLE001
            sig = None
    usr1 = getattr(sig, "SIGUSR1", None) if sig is not None else None
    if usr1 is not None:
        try:
            fh.register(usr1, file=sys.stderr, all_threads=True, chain=False)
            note += (f"; `kill -USR1 {os.getpid()}` dumps every thread's stack, which is how "
                     f"you find out where a hung run is stuck")
        except Exception:  # noqa: BLE001
            pass
    return note


AIMDO_ENV_VAR = "FUNPACK_AIMDO_LOG"


def enable_aimdo_debug_log(value=None):
    """Raise comfy_aimdo's own native log level to DEBUG. Off by default: set
    FUNPACK_AIMDO_LOG=1.

    comfy_aimdo (the dynamic-VRAM block-weight-prefetch allocator ComfyUI core loads,
    see comfy/model_prefetch.py) is a compiled native library. ComfyUI core sets ITS
    log level once at startup from ComfyUI's own --verbose flag -- a custom node loads
    after that and can only raise it further, never lower it, and only when asked.
    A native allocator failure (e.g. "aimdo memory compile error" from
    comfy_aimdo/malloc_graph.py) otherwise raises with zero detail about which
    allocation it choked on; aimdo's own DEBUG logging (routed through Python's
    `logging` module via its native log callback) is the only way to see that detail.
    Useful specifically when the launch command is templated/fixed (a hosted rental)
    and can't have --verbose added to it -- this is an env var instead.

    Also raises the root logger's OWN level to DEBUG: aimdo's callback calls
    `logging.log(...)`, which Python silently drops below the logger's configured
    level regardless of how verbose aimdo itself is set to be.

    Never raises: same reasoning as faulthandler above -- a diagnostic that breaks
    FunPack loading is worse than no diagnostic.
    """
    if not wanted(os.environ.get(AIMDO_ENV_VAR) if value is None else value):
        return None
    try:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        import comfy_aimdo.control as _control
    except Exception as e:  # noqa: BLE001
        return f"could not raise aimdo's log level: {e}"
    if _control.lib is None:
        return "aimdo is not loaded on this machine (no compatible GPU/driver) -- nothing to raise"
    try:
        _control.set_log_debug()
    except Exception as e:  # noqa: BLE001
        return f"could not raise aimdo's log level: {e}"
    return "aimdo native logging set to DEBUG (FUNPACK_AIMDO_LOG=1) -- expect a lot more console output"
