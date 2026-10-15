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


def _import_nodes(body):
    """Import statements that run when the file is imported: top level, and inside module-level try / if / with
    blocks. Function and class bodies run later, on demand, and are where a guarded import may live."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.ClassDef):                # a class body runs at import; its methods do not
            yield from _import_nodes(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        else:
            for part in ("body", "orelse", "finalbody"):
                yield from _import_nodes(getattr(node, part, None) or [])
            for handler in getattr(node, "handlers", []):
                yield from _import_nodes(handler.body)
            for case in getattr(node, "cases", []):
                yield from _import_nodes(case.body)


def _other_module_imports(path: Path, mods=None, root: Path = MODULES):
    """Imports that run at import time and reach into a module other than the file's own: `from ...group.other`,
    `from ..other` (a sibling module), `from ...group import other`, `from .. import other`, or an absolute
    `modules.group.other`. Plain shared files are fine to import, but a shared file may not import a module either."""
    parts = path.relative_to(root).parts
    mods = _module_dirs() if mods is None else mods
    group = parts[0]
    own = parts[:2] if len(parts) >= 3 else None            # a file at group level belongs to no module
    found = []
    for node in _import_nodes(ast.parse(path.read_text()).body):
        names = [a.name for a in node.names]
        if isinstance(node, ast.Import):
            targets = [tuple(n.split(".")[1:3]) for n in names if n.split(".")[0] == "modules"]
        else:
            mod = (node.module or "").split(".") if node.module else []
            if node.level == 0:
                targets = ([tuple(mod[1:3])] if len(mod) >= 3 else [(mod[1], n) for n in names]) if mod[:1] == ["modules"] and len(mod) >= 2 else []
            elif node.level >= 3:
                targets = [tuple(mod[:2])] if len(mod) >= 2 else [(mod[0], n) for n in names] if mod else []
            elif node.level == 2:
                targets = [(group, mod[0])] if mod else [(group, n) for n in names]
            elif node.level == 1 and own is None:           # a group-level helper: its `.x` is a sibling of the modules
                targets = [(group, mod[0])] if mod else [(group, n) for n in names]
            else:
                targets = []
        for t in targets:
            if len(t) == 2 and t in mods and t != own:
                found.append((node.lineno, f"imports module {t[0]}/{t[1]}"))
    return found


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: str(p.relative_to(MODULES)))
def test_a_module_does_not_depend_on_another_module_at_import(path):
    offences = _other_module_imports(path)
    assert not offences, (
        f"{path.relative_to(MODULES)} imports another module at line {offences[0][0]} ({offences[0][1]}). "
        "If that module fails to import, this one would be lost with it. Import inside the function that needs it, "
        "guarded, or ask through PROVIDES / USES.")


def test_the_guard_sees_every_way_to_import_another_module_at_load(tmp_path):
    """Positive cases: the guard has to be shown to fail, or it only proves it passes."""
    for g, m in (("sampling", "a"), ("sampling", "b"), ("loaders", "c")):
        d = tmp_path / g / m
        d.mkdir(parents=True)
        (d / "__init__.py").write_text('ID = "x"\n')
    mods = {("sampling", "a"), ("sampling", "b"), ("loaders", "c")}
    cases = {
        "from ..b import x": True, "from ...loaders.c import x": True, "from .. import b": True,
        "from ...loaders import c": True, "import modules.sampling.b": True, "from modules.sampling.b import x": True,
        "try:\n    from ..b import x\nexcept ImportError:\n    x = None": True, "if True:\n    from ..b import x": True,
        "from modules.sampling import b": True, "class K:\n    from ..b import x": True,
        "for i in []:\n    from ..b import x": True, "match 1:\n    case 1:\n        from ..b import x": True,
        "from . import helper": False, "from ..._core import log": False, "from ..prompt_taste import x": False,
        "def f():\n    from ..b import x": False,
    }
    for src, bad in cases.items():
        f = tmp_path / "sampling" / "a" / "mod.py"
        f.write_text(src + "\n")
        assert bool(_other_module_imports(f, mods, tmp_path)) is bad, src
    helper = tmp_path / "sampling" / "helper.py"
    helper.write_text("from .b import x\n")
    assert _other_module_imports(helper, mods, tmp_path), "a shared helper may not import a module"
