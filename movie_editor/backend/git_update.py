"""Git pull / branch switch for the FunPack repo (ComfyUI custom node root)."""
from __future__ import annotations

import shutil
import subprocess
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


def _list_branches() -> list[str]:
    proc = _run_git("branch", "-a", "--format=%(refname:short)")
    if proc.returncode != 0:
        raise GitUpdateError((proc.stderr or "git branch failed").strip())
    names: set[str] = set()
    for raw in (proc.stdout or "").splitlines():
        line = raw.strip()
        if not line or line == "HEAD" or line.endswith("/HEAD"):
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
        return {"ok": False, "detail": str(e)}


def pull(branch: str | None = None) -> dict:
    """Fast-forward pull from origin on the given branch (current if omitted)."""
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
    if pull_proc.returncode != 0:
        msg = (pull_proc.stderr or pull_proc.stdout or "git pull failed").strip()
        raise GitUpdateError(msg)
    after = _current_commit()
    return {
        "branch": branch,
        "before": before,
        "after": after,
        "updated": before != after,
        "output": (pull_proc.stdout or "").strip(),
    }


def checkout(branch: str, *, pull_after: bool = True) -> dict:
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
        pulled = pull(branch)
        result.update(pulled)
    else:
        result["after"] = _current_commit()
        result["updated"] = result["before"] != result["after"]
    return result
