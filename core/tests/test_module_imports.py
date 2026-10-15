"""No module may reach for `core` by absolute name.

ComfyUI loads a pack by file location and never puts its directory on sys.path,
so `from core import ...` inside a module resolves under pytest and finds nothing
under ComfyUI. This has now cost the same bug twice: once at module level, where
the node quietly failed to register, and once INSIDE A FUNCTION BODY, where it
imported fine, registered fine, passed every test, and raised at execute time --
during a real generation.

That second one is why this is a static check and not a runtime one. A test that
imports every module cannot see an import that only happens when a node runs.
"""

import ast
from pathlib import Path

import pytest

MODULES = Path(__file__).resolve().parents[2] / "modules"
SHIM = "_core.py"


def _module_files():
    return [p for p in MODULES.rglob("*.py")
            if p.name != SHIM and "/tests/" not in str(p) and p.parent.name != "tests"]


def _absolute_core_imports(path: Path):
    """Every `import core...` / `from core... import` anywhere in the file.

    Walks the AST rather than the text, so an import nested in a function, a
    method or a try block is found exactly like a top-level one.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # level > 0 is relative, which is what the shim exists to provide.
            if node.level == 0 and (node.module or "").split(".")[0] == "core":
                found.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "core":
                    found.append((node.lineno, f"import {alias.name}"))
    return found


def test_every_module_file_exists():
    assert _module_files(), "found no module files to check"


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.relative_to(MODULES)))
def test_a_module_reaches_core_only_through_the_shim(path):
    offences = _absolute_core_imports(path)
    assert not offences, (
        f"{path.relative_to(MODULES)} imports core by absolute name at "
        + ", ".join(f"line {line} ({what})" for line, what in offences)
        + ". Under ComfyUI there is no top-level `core`. Use `from ..._core import ...`."
    )


def _module_dirs():
    """group/name of every directory that announces a module (has `ID = ` in its __init__)."""
    return {(p.parent.parent.name, p.parent.name) for p in MODULES.glob("*/*/__init__.py") if "\nID = " in "\n" + p.read_text()}


def _other_module_imports(path: Path):
    """Imports at MODULE level that reach into a different module: `from ...group.other import` (three dots), or a
    sibling module of the same group (`from ..other import`, two dots). Plain shared files are fine."""
    parts = path.relative_to(MODULES).parts
    if len(parts) < 3:
        return []
    own = parts[:2]
    mods = _module_dirs()
    found = []
    for node in ast.parse(path.read_text()).body:             # module level only: a guarded import inside a function is how a module may ask another for help
        if isinstance(node, ast.ImportFrom) and node.module:
            target = node.module.split(".")
            if node.level >= 3 and len(target) >= 2:
                hit = tuple(target[:2]) != own and tuple(target[:2]) in mods
            elif node.level == 2:
                hit = (own[0], target[0]) in mods and target[0] != own[1]
            else:
                hit = False
            if hit:
                found.append((node.lineno, f"from {'.' * node.level}{node.module} import"))
    return found


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.relative_to(MODULES)))
def test_a_module_does_not_depend_on_another_module_at_import(path):
    offences = _other_module_imports(path)
    assert not offences, (
        f"{path.relative_to(MODULES)} imports another module at line {offences[0][0]} ({offences[0][1]}). "
        "If that module fails to import, this one would be lost with it. Import inside the function that needs it, "
        "guarded, or ask through PROVIDES / USES.")
