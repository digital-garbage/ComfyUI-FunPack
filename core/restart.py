"""Relaunching ComfyUI, and nothing else.

Its own file because it is the one operation here that ends the process: every
route that calls it has already finished answering, and anything that imports it
by accident should be obvious in the import list.

Carried across from v4. The three launch shapes it handles -- comfy-cli's reboot
file, `python -m comfy`, and a plain script -- are three ways somebody's install
would otherwise fail to come back, and each was found by an install failing to
come back.
"""

from __future__ import annotations

def restart() -> None:
    """Relaunch the ComfyUI process in place. Mirrors ComfyUI-Manager's reboot so it
    works the same whether launched directly, as a module, or via comfy-cli."""
    import os
    import sys
    try:
        sys.stdout.close_log()  # type: ignore[attr-defined]  # Manager's tee logger, if present
    except Exception:
        pass
    # comfy-cli watches for a .reboot file and relaunches us itself.
    if "__COMFY_CLI_SESSION__" in os.environ:
        try:
            open(os.environ["__COMFY_CLI_SESSION__"] + ".reboot", "w").close()
        except Exception:
            pass
        print("\n[FunPack] Restarting ComfyUI...\n", flush=True)
        os._exit(0)
    sys_argv = sys.argv.copy()
    if "--windows-standalone-build" in sys_argv:
        sys_argv.remove("--windows-standalone-build")
    if sys_argv and sys_argv[0].endswith("__main__.py"):  # python -m comfy
        module_name = os.path.basename(os.path.dirname(sys_argv[0]))
        cmds = [sys.executable, "-m", module_name] + sys_argv[1:]
    elif sys.platform.startswith("win32"):
        cmds = ['"' + sys.executable + '"', '"' + sys_argv[0] + '"'] + sys_argv[1:]
    else:
        cmds = [sys.executable] + sys_argv
    print(f"\n[FunPack] Restarting ComfyUI... {cmds}\n", flush=True)
    os.execv(sys.executable, cmds)
