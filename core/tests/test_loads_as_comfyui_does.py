"""Loading the pack exactly the way ComfyUI loads it.

Every other test puts the repo root on sys.path, which makes `import core` work
and hides the one import rule that actually matters: ComfyUI loads a pack by FILE
LOCATION and never adds its directory to sys.path, so an absolute `from core
import ...` inside a module finds nothing.

That gap has now cost the same bug twice -- a module imported cleanly under
pytest, failed silently under ComfyUI, and its nodes simply did not appear. The
suite could not see it, because the suite is the friendlier of the two
environments. This test is the unfriendly one.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# Run in a subprocess with ONLY ComfyUI on the path -- not the pack root. In this
# process the root is already imported, which is exactly the leniency being
# tested for.
PROBE = """
import asyncio, json, sys
sys.path.insert(0, {comfy!r})
import nodes
ok = asyncio.run(nodes.load_custom_node({pack!r}, module_parent="custom_nodes"))
print("FUNPACK_RESULT " + json.dumps({{
    "ok": bool(ok),
    "nodes": sorted(k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith("FunPack")),
}}))
"""


@pytest.fixture(scope="module")
def loaded(comfyui):
    # cwd MUST NOT be the pack root. Python puts the working directory on
    # sys.path for `-c`, so running from the repo makes `import core` succeed and
    # the whole test passes while the bug it exists for is present -- verified by
    # reintroducing the bug and watching this go green.
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE.format(comfy=comfyui, pack=str(ROOT))],
        capture_output=True, text=True, timeout=600, cwd=comfyui,
    )
    line = next((l for l in result.stdout.splitlines() if l.startswith("FUNPACK_RESULT ")), None)
    if line is None:
        pytest.fail(f"the probe did not report.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    import json
    return json.loads(line[len("FUNPACK_RESULT "):]), result.stderr


def test_the_pack_loads(loaded):
    report, _ = loaded
    assert report["ok"], "ComfyUI refused the pack entirely"


def test_every_module_that_announces_a_node_actually_registers_it(loaded, comfyui):
    """The real assertion: what ComfyUI ends up with must equal what the registry
    says exists. A module that fails to import under ComfyUI is absent there and
    present here, and only comparing the two can tell."""
    report, stderr = loaded

    from core import nodes as nodes_mod
    from core import registry as registry_mod
    expected = {n.GET_SCHEMA().node_id
                for n in nodes_mod.collect(registry_mod.scan())[0]}

    missing = expected - set(report["nodes"])
    assert not missing, (
        f"these nodes exist under pytest but not under ComfyUI: {sorted(missing)}.\n"
        f"Almost always an absolute `import core` in a module.\n"
        f"loader stderr:\n{stderr}"
    )


def test_no_module_failed_to_import_under_comfyui(loaded):
    _, stderr = loaded
    # core.routes is expected to fail here: PromptServer has no instance outside
    # a running server, and that failure is guarded on purpose.
    bad = [line for line in stderr.splitlines()
           if "failed to load" in line and "core.routes" not in line]
    assert not bad, "a module failed to import under ComfyUI:\n" + "\n".join(bad)
