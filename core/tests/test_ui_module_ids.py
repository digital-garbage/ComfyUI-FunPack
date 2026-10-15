"""The UI names backend modules by id (`modules: [...]`, TAB_MODULE, hasModule("...")). A rename on the Python side would
hide a feature silently, so every id the UI names must be a module that exists."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ids():
    return {m.group(1) for p in (ROOT / "modules").glob("*/*/__init__.py")
            for m in [re.search(r'^ID\s*=\s*"([^"]+)"', p.read_text(), re.M)] if m}


def test_every_module_id_the_ui_names_exists():
    named = {}
    for p in (ROOT / "app" / "ui").rglob("*.js"):
        if p.name.endswith(".test.js"):
            continue
        t = p.read_text()
        for m in re.finditer(r'\bmodules:\s*\[([^\]]*)\]', t):
            for i in re.findall(r'"([a-z0-9_]+)"', m.group(1)):
                named.setdefault(i, p)
        for m in re.finditer(r'hasModule\("([a-z0-9_]+)"\)', t):
            named.setdefault(m.group(1), p)
        for m in re.finditer(r'TAB_MODULE\s*=\s*\{([^}]*)\}', t):
            for i in re.findall(r':\s*"([a-z0-9_]+)"', m.group(1)):
                named.setdefault(i, p)
    assert named, "found no module ids in the UI: the pattern broke"
    missing = {i: str(p.relative_to(ROOT)) for i, p in named.items() if i not in _ids()}
    assert not missing, f"the UI names modules that do not exist: {missing}"
