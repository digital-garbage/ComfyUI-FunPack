"""The text actually sent as a scene's prompt at generation time.

`Scene.text` (core/projects.py) stays the user's literal typed words -- this
module never writes back to it. It only ever runs at the moment a prompt
value is about to leave the app for a real run, so editing a scene and
looking at what it produced always shows what was actually typed, not an
expanded copy nothing can trace back.

Order matches v4's: anchor and postfix are joined to the scene text FIRST,
then shortcuts expand across the whole combined string (a shortcut trigger
sitting right at the anchor/scene boundary still matches), then `$variables`
resolve last. Resolving last means a variable's OWN text is never re-scanned
for shortcut triggers -- only the anchor/scene/postfix text is -- but a
variable may still reference another variable, since resolve_variables()
recurses through the variable map itself (see its own docstring for the
cycle-safety that recursion needs).
"""

from __future__ import annotations

from . import shortcuts as shortcuts_mod


def build(scene_text: str, *, anchor: str = "", postfix: str = "",
          postfix_enabled: bool = True, variables=None, shortcuts=None, seed: int = 0) -> str:
    anchor = (anchor or "").strip()
    postfix = (postfix or "").strip() if postfix_enabled else ""
    combined = " ".join(p for p in (anchor, str(scene_text or "").strip(), postfix) if p)
    expanded = shortcuts_mod.expand(combined, shortcuts=shortcuts, seed=seed)
    return shortcuts_mod.resolve_variables(expanded, variables)
