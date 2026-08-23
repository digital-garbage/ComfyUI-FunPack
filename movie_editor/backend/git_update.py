"""Git pull / branch switch for the FunPack repo (ComfyUI custom node root)."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_BRANCH_PRIORITY = ("dev", "cutting_edge", "main")


class GitUpdateError(RuntimeError):
    pass


def _run_git(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    if not shutil.which("git"):
        raise GitUpdateError("git not found on PATH.")
    if not (REPO_ROOT / ".git").exists():
        raise GitUpdateError("FunPack is not a git checkout (no .git directory).")
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


def funpack_version() -> str:
    """Custom-node version from pyproject.toml (shown in Settings ▸ About)."""
    try:
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return m.group(1) if m else ""


# Adjective + vegetable sharing an initial, advancing alphabetically. Keyed by MAJOR, so
# every 3.x release ships under one name.
CODENAMES = {
    "3": "Auspicious Asparagus",
    "4": "Blinding Blackout",
}


def funpack_codename(version: str = "") -> str:
    """Codename for `version` (default: the installed one); "" when its major has none."""
    major = (version or funpack_version()).split(".", 1)[0].strip()
    return CODENAMES.get(major, "")


def _current_branch() -> str:
    proc = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    if proc.returncode != 0:
        raise GitUpdateError((proc.stderr or proc.stdout or "git rev-parse failed").strip())
    return (proc.stdout or "").strip()


def _current_commit() -> str:
    proc = _run_git("rev-parse", "--short", "HEAD")
    if proc.returncode != 0:
        raise GitUpdateError((proc.stderr or proc.stdout or "git rev-parse failed").strip())
    return (proc.stdout or "").strip()


def _is_dirty() -> bool:
    proc = _run_git("status", "--porcelain")
    if proc.returncode != 0:
        raise GitUpdateError((proc.stderr or "git status failed").strip())
    return bool((proc.stdout or "").strip())


def _ahead_behind(branch: str) -> tuple[int, int]:
    proc = _run_git("rev-list", "--left-right", "--count", f"HEAD...origin/{branch}")
    if proc.returncode != 0:
        return 0, 0
    parts = (proc.stdout or "").strip().split()
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def _unique_local_commits(branch: str) -> list[str]:
    """Local commits with no patch-equivalent upstream — i.e. real work that a reset
    would destroy. `git cherry` marks those "+" and marks "-" the ones whose CONTENT is
    already upstream under a different hash, which is exactly what a rewritten history
    leaves behind."""
    proc = _run_git("cherry", f"origin/{branch}", branch)
    if proc.returncode != 0:
        return ["<unknown>"]  # can't prove it's safe → treat as if there were local work
    return [ln[2:].strip() for ln in (proc.stdout or "").splitlines() if ln.startswith("+")]


def _remote_names() -> set[str]:
    proc = _run_git("remote")
    if proc.returncode != 0:
        return set()
    return {ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()}


def _list_branches() -> list[str]:
    proc = _run_git("branch", "-a", "--format=%(refname:short)")
    if proc.returncode != 0:
        raise GitUpdateError((proc.stderr or "git branch failed").strip())
    # refs/remotes/origin/HEAD shortens to a bare "origin", which is not a branch and
    # fails on checkout — drop anything that is just a remote's name.
    remotes = _remote_names()
    names: set[str] = set()
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line or line == "HEAD" or line.endswith("/HEAD") or line in remotes:
            continue
        if line.startswith("origin/"):
            names.add(line[7:])
        elif not line.startswith("remotes/"):
            names.add(line)
    ordered = [b for b in _BRANCH_PRIORITY if b in names]
    rest = sorted(b for b in names if b not in _BRANCH_PRIORITY)
    return ordered + rest


def status() -> dict:
    """Current branch, switchable branches, and update availability."""
    try:
        branch = _current_branch()
        commit = _current_commit()
        dirty = _is_dirty()
        branches = _list_branches()
        fetch = _run_git("fetch", "--prune", "origin")
        fetch_ok = fetch.returncode == 0
        ahead, behind = _ahead_behind(branch) if fetch_ok else (0, 0)
        return {
            "ok": True,
            "version": funpack_version(),
            "codename": funpack_codename(),
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
            "branches": branches,
            "ahead": ahead,
            "behind": behind,
            "fetch_ok": fetch_ok,
            "repo": str(REPO_ROOT),
        }
    except GitUpdateError as e:
        return {"ok": False, "version": funpack_version(),
                "codename": funpack_codename(), "detail": str(e)}


REQUIREMENTS = "requirements.txt"


def requirements_changed(before: str, after: str) -> bool:
    """Did this update touch requirements.txt?

    The only honest trigger for installing: running pip on every pull would be slow and
    surprising, and running it on none leaves an update that added a dependency looking
    like a broken build instead of an unfinished install.
    """
    if not before or not after or before == after:
        return False
    proc = _run_git("diff", "--name-only", f"{before}..{after}", "--", REQUIREMENTS)
    if proc.returncode != 0:
        # Cannot tell — install rather than skip. A redundant pip run costs seconds; a
        # skipped one costs a broken node pack and a confusing error.
        return True
    return bool((proc.stdout or "").strip())


def install_requirements(timeout: int = 900) -> dict:
    """`pip install -r requirements.txt` into the interpreter ComfyUI is running.

    `sys.executable`, never a bare `pip`: ComfyUI is usually in a venv, and the pip on PATH
    belongs to whatever else is on it. Installing into the wrong environment succeeds
    loudly and changes nothing.

    Never raises. A failed install must not turn a completed update into an error — the
    code IS updated by this point, and the remedy is a command the user can run.
    """
    req = REPO_ROOT / REQUIREMENTS
    if not req.is_file():
        return {"ran": False, "ok": True, "detail": "no requirements.txt in this checkout"}
    # What is installed BEFORE, so the update can say what it changed. An update that
    # silently moves a shared dependency is the worst kind: ComfyUI is full of compiled
    # extensions (torch, comfy-kitchen, comfy-aimdo, onnxruntime, opencv) and a numpy or
    # transformers bump under them does not raise — it segfaults, or corrupts memory, hours
    # later, with nothing in the log connecting it to the update that caused it.
    before = _pip_freeze()
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "-r", str(req)]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ran": True, "ok": False,
                "detail": f"pip did not finish within {timeout}s — run it yourself: "
                          f"pip install -r {req}"}
    except OSError as e:
        return {"ran": True, "ok": False, "detail": f"could not run pip: {e}"}
    if proc.returncode != 0:
        tail = ((proc.stderr or proc.stdout or "").strip() or "pip failed")[-800:]
        return {"ran": True, "ok": False,
                "detail": f"pip install failed — run it yourself:\n"
                          f"  {sys.executable} -m pip install -r {req}\n\n{tail}"}
    changed = _pip_diff(before, _pip_freeze())
    if changed:
        print("[FunPack update] pip changed these packages: " + ", ".join(changed))
    return {"ran": True, "ok": True, "changed": changed,
            "detail": (("Changed: " + ", ".join(changed) + "\n\n") if changed else "")
                      + (proc.stdout or "").strip()[-800:]}


def _pip_freeze() -> dict:
    """{name: version} for the environment ComfyUI is running in. {} if pip cannot be read —
    an unreadable freeze must never block the install it was only meant to describe."""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze",
                              "--disable-pip-version-check"],
                             capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            return {}
        found = {}
        for line in (out.stdout or "").splitlines():
            if "==" in line:
                name, _, ver = line.partition("==")
                found[name.strip().lower()] = ver.strip()
        return found
    except Exception:  # noqa: BLE001
        return {}


def _pip_diff(before: dict, after: dict) -> list[str]:
    """Human-readable list of what moved. Empty when nothing did, or when either side is
    unknown — reporting every package as "new" because the freeze failed would be worse
    than saying nothing."""
    if not before or not after:
        return []
    out = []
    for name, ver in sorted(after.items()):
        was = before.get(name)
        if was is None:
            out.append(f"{name} {ver} (new)")
        elif was != ver:
            out.append(f"{name} {was} -> {ver}")
    for name in sorted(set(before) - set(after)):
        out.append(f"{name} {before[name]} (removed)")
    return out


def pull(branch: str | None = None, *, install_deps: bool = False) -> dict:
    """Fast-forward pull from origin on the given branch (current if omitted).

    `install_deps` is OFF by default and turned on by the update route. Running pip is a
    side effect no caller should acquire by accident just for asking git to move a branch —
    it has to be asked for at the point that means "the user pressed Update".
    """
    branch = (branch or _current_branch()).strip()
    if not branch:
        raise GitUpdateError("Could not determine current branch.")
    if _is_dirty():
        raise GitUpdateError("Working tree has local changes. Commit or stash them before updating.")
    before = _current_commit()
    fetch = _run_git("fetch", "--prune", "origin")
    if fetch.returncode != 0:
        raise GitUpdateError((fetch.stderr or fetch.stdout or "git fetch failed").strip())
    if branch != _current_branch():
        co = _run_git("checkout", branch)
        if co.returncode != 0:
            raise GitUpdateError((co.stderr or co.stdout or "git checkout failed").strip())
    pull_proc = _run_git("pull", "--ff-only", "origin", branch)
    realigned = False
    if pull_proc.returncode != 0:
        msg = (pull_proc.stderr or pull_proc.stdout or "git pull failed").strip()
        # A fast-forward is impossible once upstream history has been rewritten: every
        # local commit is a different hash, so git reports the branches as diverged even
        # though the content matches. Only realign when nothing would actually be lost —
        # a clean tree (checked above) and no local commit whose content is missing
        # upstream. Anything else keeps the original error and the user's work.
        unique = _unique_local_commits(branch)
        if unique:
            raise GitUpdateError(
                f"{msg}\n\nThis checkout has {len(unique)} local commit(s) that are not on "
                f"origin/{branch}. Push or move them first — updating would discard them.")
        reset = _run_git("reset", "--hard", f"origin/{branch}")
        if reset.returncode != 0:
            raise GitUpdateError(msg)
        realigned = True
    after = _current_commit()
    # Dependencies are part of the update. Done BEFORE the response, so the restart the
    # caller schedules cannot race an install that is still running.
    deps = None
    if install_deps and before != after and requirements_changed(before, after):
        deps = install_requirements()
    return {
        "branch": branch,
        "before": before,
        "after": after,
        "updated": before != after,
        "requirements": deps,
        # Surfaced so the update reads as what it was, not a silent jump to another commit.
        "realigned": realigned,
        "output": ((pull_proc.stdout or "").strip() if not realigned
                   else f"Upstream history was rewritten — realigned to origin/{branch}. "
                        "No local commits were lost."),
    }


def checkout(branch: str, *, pull_after: bool = True, install_deps: bool = False) -> dict:
    """Switch branch, optionally pull, return combined result."""
    branch = (branch or "").strip()
    if not branch:
        raise GitUpdateError("Branch name is required.")
    branches = _list_branches()
    if branch not in branches:
        raise GitUpdateError(f'Branch "{branch}" is not available locally or on origin.')
    if _is_dirty():
        raise GitUpdateError("Working tree has local changes. Commit or stash them before switching branches.")
    before_branch = _current_branch()
    before_commit = _current_commit()
    if branch != before_branch:
        co = _run_git("checkout", branch)
        if co.returncode != 0:
            raise GitUpdateError((co.stderr or co.stdout or "git checkout failed").strip())
    result = {"branch": branch, "before_branch": before_branch, "before": before_commit}
    if pull_after:
        pulled = pull(branch, install_deps=install_deps)
        result.update(pulled)
    else:
        result["after"] = _current_commit()
        result["updated"] = result["before"] != result["after"]
        # A branch switch alone can cross a requirements change just as a pull can — the
        # checkout above already moved the working tree onto the other branch's files.
        if install_deps and result["updated"] and requirements_changed(before_commit,
                                                                      result["after"]):
            result["requirements"] = install_requirements()
    return result
