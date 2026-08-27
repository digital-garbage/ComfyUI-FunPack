"""Install / update / remove for custom node packs.

Most of this file is about `node_dir`. It is the only thing standing between a name typed
into a web UI and `shutil.rmtree`, so every way of escaping custom_nodes gets a test.
"""
import sys
import types
from pathlib import Path

import pytest

from movie_editor.backend import custom_nodes as cn


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fake custom_nodes with FunPack and one other pack in it."""
    cnodes = tmp_path / "custom_nodes"
    (cnodes / "ComfyUI-FunPack").mkdir(parents=True)
    (cnodes / "ComfyUI-GGUF").mkdir()
    (cnodes / "SomePack").mkdir()
    monkeypatch.setattr(cn, "custom_nodes_root", lambda: cnodes.resolve())
    monkeypatch.setattr(cn, "FUNPACK_ROOT", (cnodes / "ComfyUI-FunPack").resolve())
    return cnodes


# ── node_dir: the guard in front of rmtree ────────────────────────────────────

def test_resolves_a_real_pack(root):
    assert cn.node_dir("ComfyUI-GGUF") == (root / "ComfyUI-GGUF").resolve()


@pytest.mark.parametrize("name", [
    "", "   ", ".", "..", "../..", "../../etc", "sub/dir", "sub\\dir",
    ".hidden", "/etc", "\\windows",
])
def test_path_escapes_are_refused(root, name):
    with pytest.raises(cn.CustomNodeError):
        cn.node_dir(name)


def test_a_symlink_pointing_outside_is_refused(root, tmp_path):
    """The name is one segment and the entry really is in custom_nodes — but following it
    lands elsewhere, which is exactly what resolving before checking is for."""
    outside = tmp_path / "not_custom_nodes"
    outside.mkdir()
    try:
        (root / "sneaky").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable here")
    with pytest.raises(cn.CustomNodeError, match="inside custom_nodes"):
        cn.node_dir("sneaky")


def test_funpack_cannot_delete_itself(root):
    with pytest.raises(cn.CustomNodeError, match="FunPack itself"):
        cn.node_dir("ComfyUI-FunPack")


def test_a_missing_pack_is_refused_not_created(root):
    with pytest.raises(cn.CustomNodeError, match="No custom node"):
        cn.node_dir("NeverInstalled")


def test_a_file_is_not_a_pack(root):
    (root / "notes.txt").write_text("x")
    with pytest.raises(cn.CustomNodeError, match="not a directory"):
        cn.node_dir("notes.txt")


def test_remove_deletes_only_that_directory(root):
    cn.remove("SomePack")
    assert not (root / "SomePack").exists()
    assert (root / "ComfyUI-GGUF").is_dir()
    assert (root / "ComfyUI-FunPack").is_dir()


def test_remove_refuses_what_node_dir_refuses(root):
    """rmtree is never reached on a name node_dir rejects — the same guard, one path in."""
    with pytest.raises(cn.CustomNodeError):
        cn.remove("../..")
    assert root.exists()


# ── install: URLs ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/owner/repo", "repo"),
    ("https://github.com/owner/repo.git", "repo"),
    ("https://github.com/owner/repo/", "repo"),
    ("git@github.com:owner/repo.git", "repo"),
])
def test_directory_name_follows_gits_own_rule(url, expected):
    assert cn.repo_name(url) == expected


@pytest.mark.parametrize("url", [
    "", "not a url", "ftp://example.com/repo", "file:///etc",
    "/local/path", "../repo", "javascript:alert(1)",
])
def test_non_git_urls_are_refused(root, url):
    with pytest.raises(cn.CustomNodeError, match="git URL"):
        cn.install(url)


def test_installing_over_an_existing_pack_is_refused(root):
    with pytest.raises(cn.CustomNodeError, match="already in custom_nodes"):
        cn.install("https://github.com/owner/ComfyUI-GGUF")


def test_install_clones_then_installs_requirements(root, monkeypatch):
    calls = []

    def fake_git(cwd, *args, **kw):
        calls.append(args)
        (Path(cwd) / args[-1]).mkdir()
        (Path(cwd) / args[-1] / "requirements.txt").write_text("numpy\n")
        return types.SimpleNamespace(returncode=0, stdout="cloned", stderr="")

    monkeypatch.setattr(cn, "_git", fake_git)
    monkeypatch.setattr(cn.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout="Successfully installed numpy", stderr=""))
    out = cn.install("https://github.com/owner/NewPack")
    assert out["name"] == "NewPack"
    assert calls[0][0] == "clone"
    assert out["requirements"]["ok"] is True


def test_a_pack_without_requirements_installs_nothing(root, monkeypatch):
    monkeypatch.setattr(cn.subprocess, "run",
                        lambda *a, **k: pytest.fail("pip must not run"))
    assert cn.install_requirements(root / "SomePack") is None


def test_a_failed_pip_reports_instead_of_raising(root, monkeypatch):
    """The pack is on disk either way; an exception here would hide that it was cloned."""
    (root / "SomePack" / "requirements.txt").write_text("nope\n")
    monkeypatch.setattr(cn.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=1, stdout="", stderr="No matching distribution found for nope"))
    out = cn.install_requirements(root / "SomePack")
    assert out["ok"] is False
    assert "pip install -r" in out["detail"]
    assert sys.executable in out["detail"]


# ── update ────────────────────────────────────────────────────────────────────

def test_update_refuses_a_non_git_pack(root):
    with pytest.raises(cn.CustomNodeError, match="not a git checkout"):
        cn.update("SomePack")


def test_update_refuses_to_discard_local_changes(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()
    monkeypatch.setattr(cn, "_git", lambda cwd, *a, **k: types.SimpleNamespace(
        returncode=0, stdout=" M nodes.py\n", stderr=""))
    with pytest.raises(cn.CustomNodeError, match="local changes"):
        cn.update("SomePack")


def test_update_skips_pip_when_nothing_moved(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()
    monkeypatch.setattr(cn, "_git", lambda cwd, *a, **k: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(cn, "_git_info", lambda p: {"commit": "same"})
    monkeypatch.setattr(cn, "install_requirements",
                        lambda *a, **k: pytest.fail("pip must not run"))
    out = cn.update("SomePack")
    assert out["updated"] is False and out["requirements"] is None


# ── listing ───────────────────────────────────────────────────────────────────

def test_listing_names_funpack_so_the_ui_can_protect_it(root, monkeypatch):
    monkeypatch.setattr(cn, "_git_info", lambda p: {"git": False, "branch": "",
                                                    "commit": "", "remote": ""})
    names = {n["name"]: n for n in cn.list_nodes()["nodes"]}
    assert names["ComfyUI-FunPack"]["is_funpack"] is True
    assert names["SomePack"]["is_funpack"] is False


def test_listing_skips_dotfiles_and_disabled_packs(root, monkeypatch):
    (root / "__pycache__").mkdir()
    (root / ".git").mkdir()
    (root / "Old.disabled").mkdir()
    monkeypatch.setattr(cn, "_git_info", lambda p: {"git": False, "branch": "",
                                                    "commit": "", "remote": ""})
    names = [n["name"] for n in cn.list_nodes()["nodes"]]
    assert "__pycache__" not in names and ".git" not in names
    assert "Old.disabled" not in names


# ── check for updates ─────────────────────────────────────────────────────────
# A network round trip per pack, which is why it is a button. Every failure mode has to
# produce a reason rather than an absent answer, or a pack silently reads as "fine".

def test_behind_ahead_parses_the_counts(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()

    def fake_git(cwd, *args, **kw):
        if args[0] == "rev-parse":
            return types.SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if args[0] == "config":
            return types.SimpleNamespace(returncode=0, stdout="https://x/y\n", stderr="")
        if args[0] == "fetch":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="2\t7\n", stderr="")

    monkeypatch.setattr(cn, "_git", fake_git)
    out = cn._behind_ahead(root / "SomePack")
    assert out == {"checked": True, "branch": "main", "ahead": 2, "behind": 7}


def test_a_detached_head_says_so(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()
    monkeypatch.setattr(cn, "_git", lambda cwd, *a, **k: types.SimpleNamespace(
        returncode=0, stdout="HEAD\n", stderr=""))
    out = cn._behind_ahead(root / "SomePack")
    assert out["checked"] is False and "detached" in out["reason"]


def test_an_unreachable_origin_says_so(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()

    def fake_git(cwd, *args, **kw):
        if args[0] == "rev-parse":
            return types.SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        if args[0] == "config":
            return types.SimpleNamespace(returncode=0, stdout="https://x/y\n", stderr="")
        return types.SimpleNamespace(returncode=128, stdout="",
                                     stderr="fatal: could not read from remote")

    monkeypatch.setattr(cn, "_git", fake_git)
    out = cn._behind_ahead(root / "SomePack")
    assert out["checked"] is False and "remote" in out["reason"]


def test_a_pack_with_no_remote_is_not_compared(root, monkeypatch):
    (root / "SomePack" / ".git").mkdir()

    def fake_git(cwd, *args, **kw):
        if args[0] == "rev-parse":
            return types.SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(cn, "_git", fake_git)
    assert cn._behind_ahead(root / "SomePack")["reason"] == "no origin remote"


def test_a_non_git_pack_is_reported_not_omitted(root):
    out = cn._behind_ahead(root / "SomePack")
    assert out == {"checked": False, "reason": "not a git checkout"}


def test_every_pack_appears_in_the_result(root, monkeypatch):
    """Including the ones that could not be checked — an absent entry would read as
    'nothing to say', which is not the same as 'could not tell'."""
    monkeypatch.setattr(cn, "_git_info", lambda p: {
        "git": p.name == "ComfyUI-GGUF", "branch": "main", "commit": "abc", "remote": "x"})
    monkeypatch.setattr(cn, "_behind_ahead",
                        lambda p: {"checked": True, "branch": "main", "ahead": 0, "behind": 3})
    got = cn.check_updates()["checked"]
    assert set(got) == {"ComfyUI-FunPack", "ComfyUI-GGUF", "SomePack"}
    assert got["ComfyUI-GGUF"]["behind"] == 3
    assert got["SomePack"]["checked"] is False


def test_one_failing_pack_does_not_sink_the_sweep(root, monkeypatch):
    monkeypatch.setattr(cn, "_git_info", lambda p: {
        "git": True, "branch": "main", "commit": "abc", "remote": "x"})

    def flaky(path):
        if path.name == "SomePack":
            raise RuntimeError("git exploded")
        return {"checked": True, "branch": "main", "ahead": 0, "behind": 1}

    monkeypatch.setattr(cn, "_behind_ahead", flaky)
    got = cn.check_updates()["checked"]
    assert got["SomePack"]["checked"] is False
    assert "exploded" in got["SomePack"]["reason"]
    assert got["ComfyUI-GGUF"]["behind"] == 1
