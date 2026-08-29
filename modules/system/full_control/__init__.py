"""Full control: the foundation stops protecting you, and says so.

The foundation's job is that nothing it does can produce a bug on its own --
values that cannot work are refused before a run starts, and a module that fails
mid-run is dropped rather than taking the generation with it.

Both of those are in the way when the point IS to see what happens. A one-step
schedule, a width the model cannot use, a modifier that throws -- someone
exploring the edges wants the real behaviour and the real traceback, not a
careful refusal. So this turns the guards off wholesale.

It is one switch, not a switch per guard, because a half-disabled safety net is
harder to reason about than either extreme: you would never know which refusal
you were still getting. And it defaults to off, because the surprising direction
has to be the one you asked for.

Nothing here silently changes behaviour: every guard that steps aside says that
it did, and every run made with this on carries it in the status.
"""

ID = "full_control"
TITLE = "Full control"
MOUNT = "settings.general"
STAGE = "load"
STATUS = "proven"

SETTINGS = {
    "enabled": {
        "type": "bool", "default": False,
        "label": "Full control",
        "hint": "Stop refusing settings that look wrong, and let failures surface. "
                "Anything that breaks from here is yours to keep.",
    },
}

# The one place that decides what "on" means, so no caller can get the key or
# the default subtly wrong -- and offered as a CAPABILITY rather than imported,
# so nothing else has to name this module to ask. A build without it installed
# simply has nobody answering, which reads as guarded: the safe direction.
def is_on(settings) -> bool:
    return bool(((settings or {}).get(ID) or {}).get("enabled", False))


WARNING = ("full control is ON: nothing is being refused or guarded, and any "
           "consequences of these settings are yours")

PROVIDES = {"guards_off": is_on}
