"""Install, update and remove ComfyUI custom node packs.

A small stand-in for ComfyUI-Manager's three operations, so FunPack users are not sent to
another UI to add the one pack a workflow needs: clone from a git URL, pull, delete.

There is no registry and no curation here — the user supplies the URL. That keeps this
honest about what it is (git, in a directory) rather than implying a vetted catalogue.

**The delete is the whole risk in this file.** Everything that resolves a pack name goes
through `node_dir`, which refuses anything that is not a direct child directory of
`custom_nodes` — resolved, so a symlink cannot point the deletion somewhere else — and
refuses FunPack itself. Nothing here interpolates into a shell.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

FUNPACK_ROOT = Path(__file__).resolve().parents[2]


class CustomNodeError(RuntimeError):
    pass


def custom_nodes_root() -> Path:
    """The custom_nodes directory. FunPack lives inside it, which is the reliable anchor —
    folder_paths is asked first only so an unusual install layout still works."""
    try:
        import folder_paths
        base = Path(folder_paths.base_path) / "custom_nodes"
        if base.is_dir():
            return base.resolve()
    except Exception:  # noqa: BLE001
        pass
    return FUNPACK_ROOT.parent.resolve()


def node_dir(name: str) -> Path:
    """Resolve a pack name to its directory, or refuse.

    Every rule here exists to keep a delete inside custom_nodes:
      * one path segment, so "../.." cannot climb out;
      * resolved, so a symlink cannot redirect it;
      * the resolved parent must BE custom_nodes, not merely start with it;
      * never FunPack itself, which would delete the code doing the deleting.
    """
    raw = str(name or "").strip()
    if not raw or raw in (".", ".."):
        raise CustomNodeError("A node pack name is required.")
    if "/" in raw or "\\" in raw or raw.startswith("."):
        raise CustomNodeError(f"{raw!r} is not a node pack name.")
    root = custom_nodes_root()
    path = (root / raw)
    if not path.exists():
        raise CustomNodeError(f"No custom node named {raw!r} is installed.")
    resolved = path.resolve()
    if resolved.parent != root:
        raise CustomNodeError(
            f"{raw!r} does not resolve to a directory inside custom_nodes — refusing to touch it.")
    if not resolved.is_dir():
        raise CustomNodeError(f"{raw!r} is not a directory.")
    if resolved == FUNPACK_ROOT:
        raise CustomNodeError(
            "That is FunPack itself. Use Settings ▸ Update FunPack, or remove it by hand.")
    return resolved


# ── git helpers ───────────────────────────────────────────────────────────────

def _git(cwd: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    if not shutil.which("git"):
        raise CustomNodeError("git not found on PATH.")
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout)


def _git_info(path: Path) -> dict:
    if not (path / ".git").exists():
        return {"git": False, "branch": "", "commit": "", "remote": ""}
    def one(*args):
        p = _git(path, *args, timeout=30)
        return (p.stdout or "").strip() if p.returncode == 0 else ""
    return {
        "git": True,
        "branch": one("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": one("rev-parse", "--short", "HEAD"),
        "remote": one("config", "--get", "remote.origin.url"),
    }


# ── listing ───────────────────────────────────────────────────────────────────

def _is_pack(p: Path) -> bool:
    if not p.is_dir() or p.name.startswith(".") or p.name == "__pycache__":
        return False
    # ComfyUI's own disabled-pack convention, and its example directory.
    return not p.name.endswith(".disabled")


def list_nodes() -> dict:
    root = custom_nodes_root()
    out = []
    if root.is_dir():
        for entry in sorted(root.iterdir(), key=lambda e: e.name.lower()):
            if not _is_pack(entry):
                continue
            info = {"name": entry.name, "is_funpack": entry.resolve() == FUNPACK_ROOT}
            info.update(_git_info(entry))
            out.append(info)
    return {"root": str(root), "nodes": out}


# ── install ───────────────────────────────────────────────────────────────────

# Deliberately narrow: an https/ssh git URL, nothing that could name a local path. Cloning
# is not a shell call, so this is not injection defence — it is refusing to fetch code from
# somewhere the user did not mean.
_URL_RE = re.compile(r"^(https://|git@)[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$")


def repo_name(url: str) -> str:
    """Directory a clone of `url` would land in — git's own rule: last segment, no .git."""
    tail = str(url or "").rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[:-4]
    return tail.strip()


def install(url: str) -> dict:
    """Clone a pack into custom_nodes, then install its requirements if it has any."""
    url = str(url or "").strip()
    if not _URL_RE.match(url):
        raise CustomNodeError(
            "Enter a git URL, e.g. https://github.com/owner/repo (https:// or git@ only).")
    name = repo_name(url)
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise CustomNodeError(f"Could not work out a directory name from {url!r}.")
    root = custom_nodes_root()
    target = root / name
    if target.exists():
        raise CustomNodeError(
            f"{name!r} is already in custom_nodes. Update it instead, or remove it first.")
    proc = _git(root, "clone", "--depth", "1", url, name, timeout=900)
    if proc.returncode != 0:
        raise CustomNodeError((proc.stderr or proc.stdout or "git clone failed").strip()[-800:])
    reqs = install_requirements(target)
    return {"name": name, "requirements": reqs,
            "detail": (proc.stderr or proc.stdout or "").strip()[-400:]}


def install_requirements(path: Path, timeout: int = 900) -> dict | None:
    """`pip install -r requirements.txt` for one pack, into ComfyUI's interpreter.

    None when the pack declares none. Never raises: the pack is on disk either way, and a
    failure the user can act on beats an exception that hides what already happened.
    """
    req = path / "requirements.txt"
    if not req.is_file():
        return None
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
           "-r", str(req)]
    try:
        proc = subprocess.run(cmd, cwd=str(path), capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ran": True, "ok": False,
                "detail": f"pip did not finish within {timeout}s — run it yourself: "
                          f"{sys.executable} -m pip install -r {req}"}
    except OSError as e:
        return {"ran": True, "ok": False, "detail": f"could not run pip: {e}"}
    if proc.returncode != 0:
        tail = ((proc.stderr or proc.stdout or "").strip() or "pip failed")[-800:]
        return {"ran": True, "ok": False,
                "detail": f"pip install failed — run it yourself:\n"
                          f"  {sys.executable} -m pip install -r {req}\n\n{tail}"}
    return {"ran": True, "ok": True, "detail": (proc.stdout or "").strip()[-400:]}


# ── update / remove ───────────────────────────────────────────────────────────

def update(name: str) -> dict:
    path = node_dir(name)
    if not (path / ".git").exists():
        raise CustomNodeError(f"{name!r} is not a git checkout — nothing to pull.")
    dirty = _git(path, "status", "--porcelain", timeout=60)
    if (dirty.stdout or "").strip():
        raise CustomNodeError(
            f"{name!r} has local changes. Commit or discard them before updating.")
    before = _git_info(path).get("commit", "")
    proc = _git(path, "pull", "--ff-only", timeout=600)
    if proc.returncode != 0:
        raise CustomNodeError((proc.stderr or proc.stdout or "git pull failed").strip()[-800:])
    after = _git_info(path).get("commit", "")
    reqs = install_requirements(path) if before != after else None
    return {"name": name, "before": before, "after": after, "updated": before != after,
            "requirements": reqs, "detail": (proc.stdout or "").strip()[-400:]}


def remove(name: str) -> dict:
    """Delete a pack's directory. Irreversible, so `node_dir` is the whole safety story."""
    path = node_dir(name)
    shutil.rmtree(path)
    return {"name": name, "removed": True, "path": str(path)}


# ── checking for updates ──────────────────────────────────────────────────────

def _behind_ahead(path: Path) -> dict:
    """How far one pack is from its upstream, after fetching it.

    Costs a network round trip per pack, which is why it is a button rather than part of
    the listing: on a tunnelled instance with a dozen packs, doing this on open would make
    the panel feel broken.
    """
    if not (path / ".git").exists():
        return {"checked": False, "reason": "not a git checkout"}
    br = _git(path, "rev-parse", "--abbrev-ref", "HEAD", timeout=30)
    branch = (br.stdout or "").strip()
    if br.returncode != 0 or not branch or branch == "HEAD":
        return {"checked": False, "reason": "not on a branch (detached HEAD)"}
    remote = _git(path, "config", "--get", "remote.origin.url", timeout=30)
    if remote.returncode != 0 or not (remote.stdout or "").strip():
        return {"checked": False, "reason": "no origin remote"}
    fetched = _git(path, "fetch", "--quiet", "origin", branch, timeout=180)
    if fetched.returncode != 0:
        return {"checked": False,
                "reason": (fetched.stderr or "could not reach origin").strip()[-200:]}
    counts = _git(path, "rev-list", "--left-right", "--count",
                  f"HEAD...origin/{branch}", timeout=60)
    if counts.returncode != 0:
        return {"checked": False, "reason": "no upstream branch to compare against"}
    parts = (counts.stdout or "").split()
    if len(parts) != 2:
        return {"checked": False, "reason": "could not compare with origin"}
    ahead, behind = int(parts[0]), int(parts[1])
    return {"checked": True, "branch": branch, "ahead": ahead, "behind": behind}


def check_updates() -> dict:
    """Fetch every git pack and report how far behind each one is.

    Fetches run concurrently and bounded: they are network-bound, so serial would be as slow
    as the sum of them, and unbounded would open a connection per pack at once.
    """
    from concurrent.futures import ThreadPoolExecutor

    packs = [n for n in list_nodes()["nodes"]]
    targets = [p for p in packs if p.get("git")]
    results: dict[str, dict] = {}
    if targets:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for p in targets:
                try:
                    futures[pool.submit(_behind_ahead, node_dir(p["name"]))] = p["name"]
                except CustomNodeError as e:
                    results[p["name"]] = {"checked": False, "reason": str(e)}
            for fut, name in futures.items():
                try:
                    results[name] = fut.result()
                except Exception as e:  # noqa: BLE001 — one bad pack must not sink the sweep
                    results[name] = {"checked": False, "reason": str(e)[-200:]}
    for p in packs:
        if p["name"] not in results:
            results[p["name"]] = {"checked": False, "reason": "not a git checkout"}
    return {"checked": results}
