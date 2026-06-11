"""Git update helpers for the Movie Editor."""

from types import SimpleNamespace

import pytest

from movie_editor.backend import git_update


def _proc(code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def test_status_reports_branch(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _proc(stdout="dev\n")
        if args == ("rev-parse", "--short", "HEAD"):
            return _proc(stdout="abc1234\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout="")
        if args == ("branch", "-a", "--format=%(refname:short)"):
            return _proc(stdout="dev\norigin/dev\norigin/main\n")
        if args == ("fetch", "--prune", "origin"):
            return _proc()
        if args == ("rev-list", "--left-right", "--count", "HEAD...origin/dev"):
            return _proc(stdout="0\t2\n")
        return _proc(code=1, stderr="unexpected")

    monkeypatch.setattr(git_update, "_run_git", fake_run)

    st = git_update.status()
    assert st["ok"] is True
    assert st["branch"] == "dev"
    assert st["commit"] == "abc1234"
    assert st["behind"] == 2
    assert st["branches"] == ["dev", "main"]


def test_pull_ff_only(monkeypatch):
    state = {"branch": "dev", "commit": "aaa"}

    def fake_run(*args, **kwargs):
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _proc(stdout=f"{state['branch']}\n")
        if args == ("rev-parse", "--short", "HEAD"):
            return _proc(stdout=f"{state['commit']}\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout="")
        if args == ("fetch", "--prune", "origin"):
            return _proc()
        if args == ("pull", "--ff-only", "origin", "dev"):
            state["commit"] = "bbb"
            return _proc(stdout="Already up to date.\n")
        return _proc(code=1, stderr="unexpected")

    monkeypatch.setattr(git_update, "_run_git", fake_run)

    result = git_update.pull("dev")
    assert result["branch"] == "dev"
    assert result["before"] == "aaa"
    assert result["after"] == "bbb"
    assert result["updated"] is True


def test_pull_rejects_dirty_tree(monkeypatch):
    def fake_run(*args, **kwargs):
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _proc(stdout="dev\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout=" M movie_editor/server.py\n")
        return _proc(code=1, stderr="unexpected")

    monkeypatch.setattr(git_update, "_run_git", fake_run)

    with pytest.raises(git_update.GitUpdateError, match="local changes"):
        git_update.pull("dev")


def test_checkout_unknown_branch(monkeypatch):
    def fake_run(*args, **kwargs):
        if args == ("branch", "-a", "--format=%(refname:short)"):
            return _proc(stdout="dev\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout="")
        return _proc(code=1, stderr="unexpected")

    monkeypatch.setattr(git_update, "_run_git", fake_run)

    with pytest.raises(git_update.GitUpdateError, match="not available"):
        git_update.checkout("missing-branch", pull_after=False)


def test_checkout_switches_branch(monkeypatch):
    state = {"branch": "dev", "commit": "aaa"}

    def fake_run(*args, **kwargs):
        if args == ("branch", "-a", "--format=%(refname:short)"):
            return _proc(stdout="dev\ncutting_edge\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout="")
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _proc(stdout=f"{state['branch']}\n")
        if args == ("rev-parse", "--short", "HEAD"):
            return _proc(stdout=f"{state['commit']}\n")
        if args == ("checkout", "cutting_edge"):
            state["branch"] = "cutting_edge"
            return _proc()
        return _proc(code=1, stderr="unexpected")

    monkeypatch.setattr(git_update, "_run_git", fake_run)

    result = git_update.checkout("cutting_edge", pull_after=False)
    assert result["branch"] == "cutting_edge"
    assert result["before_branch"] == "dev"
