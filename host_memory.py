"""Cap ComfyUI's pinned host-memory budget from inside FunPack.

ComfyUI page-locks host RAM so weights can stream to the GPU without a bounce buffer, and
its default budget is up to 90% of system RAM (`comfy/model_management.py`):

    MAX_PINNED_MEMORY = max(ram*0.40, min(ram*0.90, ram - 4GB, ram + swap - 16GB))

On a 125 GB box that is ~112 GB, leaving ~12 GB for the OS, the page cache, ffmpeg and the
VAE decode. Pinned pages are UNEVICTABLE: the kernel cannot swap or reclaim them, so once the
budget is committed a further allocation puts the machine into direct reclaim with nothing it
is allowed to free. The process is not killed — the host stops responding. That is a very
different failure from an OOM, and it leaves no traceback anywhere.

`--disable-pinned-memory` is the supported switch, but a rented instance often bakes its
launch command into an image you cannot edit. FunPack is imported by ComfyUI *after*
model_management has computed the budget and *before* any model is staged, which is exactly
the window where the number can still be changed. So the same control is available here.

Set FUNPACK_PINNED_MEMORY to:
  off | 0 | disable   — no pinning at all (equivalent to --disable-pinned-memory)
  <N>                 — cap the budget at N gigabytes
  <N>%                — cap at N percent of total RAM
Unset leaves ComfyUI's own number alone.
"""
import os

ENV_VAR = "FUNPACK_PINNED_MEMORY"
_OFF = {"off", "0", "no", "false", "disable", "disabled", "none"}


def parse_budget(value, total_ram):
    """Requested budget in bytes, or None to leave ComfyUI's number alone.

    Returns 0 for "off" — that is the value model_management already treats as "do not pin",
    so disabling needs no separate flag. Anything unparseable returns None: a typo must not
    silently reconfigure memory.
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _OFF:
        return 0
    try:
        if text.endswith("%"):
            pct = float(text[:-1])
            if not 0 < pct <= 100 or not total_ram:
                return None
            return int(total_ram * pct / 100.0)
        gb = float(text.rstrip("gb").strip() or "nan")
        return int(gb * 1024 ** 3) if gb > 0 else None
    except (TypeError, ValueError):
        return None


def apply(value=None, total_ram=None, mm=None):
    """Apply the budget to comfy.model_management. Returns a status line, or None.

    Only ever LOWERS it. Raising the budget past what ComfyUI computed would hand the same
    unevictable-memory footgun back with a bigger foot, and nothing about FunPack knows
    better than model_management how much this box can pin.
    """
    if mm is None:
        try:
            import comfy.model_management as mm  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
    if total_ram is None:
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except Exception:  # noqa: BLE001
            total_ram = 0
    want = parse_budget(os.environ.get(ENV_VAR) if value is None else value, total_ram)
    if want is None:
        return None
    current = float(getattr(mm, "MAX_PINNED_MEMORY", -1) or -1)
    if current <= 0:
        return f"pinned memory is already off; {ENV_VAR} left it alone"
    if want >= current:
        return (f"{ENV_VAR} asks for {want / 1024 ** 3:.0f} GB but ComfyUI already allows "
                f"{current / 1024 ** 3:.0f} GB — left alone (this only lowers the budget)")
    mm.MAX_PINNED_MEMORY = want
    if want <= 0:
        return ("pinned host memory DISABLED (was "
                f"{current / 1024 ** 3:.0f} GB). Weight transfers use a bounce buffer and are "
                "somewhat slower; the host keeps memory it can actually reclaim.")
    return (f"pinned host memory capped at {want / 1024 ** 3:.0f} GB "
            f"(ComfyUI wanted {current / 1024 ** 3:.0f} GB"
            + (f" of {total_ram / 1024 ** 3:.0f} GB RAM" if total_ram else "") + ")")
