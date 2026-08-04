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


def _rewritten_upstream(cherry_stdout, *, reset_ok=True):
    """A checkout whose branch diverged because upstream history was rewritten."""
    def fake_run(*args, **kwargs):
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _proc(stdout="dev\n")
        if args == ("rev-parse", "--short", "HEAD"):
            return _proc(stdout="old1234\n")
        if args == ("status", "--porcelain"):
            return _proc(stdout="")
        if args == ("fetch", "--prune", "origin"):
            return _proc()
        if args == ("pull", "--ff-only", "origin", "dev"):
            return _proc(code=1, stderr="fatal: Not possible to fast-forward, aborting.")
        if args == ("cherry", "origin/dev", "dev"):
            return _proc(stdout=cherry_stdout)
        if args == ("reset", "--hard", "origin/dev"):
            return _proc() if reset_ok else _proc(code=1, stderr="reset failed")
        return _proc(code=1, stderr="unexpected")
    return fake_run


def test_pull_realigns_after_upstream_history_rewrite(monkeypatch):
    """History was rewritten upstream, so every local commit has a new hash and no
    fast-forward exists. `git cherry` marks them "-" (content already upstream), which
    means a reset loses nothing — update instead of dead-ending the user in git."""
    monkeypatch.setattr(git_update, "_run_git", _rewritten_upstream("- aaa111\n- bbb222\n"))
    res = git_update.pull("dev")
    assert res["realigned"] is True
    assert res["updated"] is False        # same short hash from the stub; the reset ran
    assert "rewritten" in res["output"] and "No local commits were lost" in res["output"]


def test_pull_refuses_to_realign_over_real_local_work(monkeypatch):
    """A "+" from `git cherry` is a local commit whose content is NOT upstream. Resetting
    would destroy it, so the update has to stop and say so."""
    monkeypatch.setattr(git_update, "_run_git", _rewritten_upstream("- aaa111\n+ ccc333\n"))
    with pytest.raises(git_update.GitUpdateError) as e:
        git_update.pull("dev")
    assert "1 local commit" in str(e.value)
    assert "discard" in str(e.value)


def test_pull_keeps_the_original_error_when_it_cannot_prove_safety(monkeypatch):
    """`git cherry` itself failing is not evidence of safety — keep the plain failure."""
    def fake_run(*args, **kwargs):
        if args == ("cherry", "origin/dev", "dev"):
            return _proc(code=1, stderr="unknown revision")
        return _rewritten_upstream("")(*args, **kwargs)
    monkeypatch.setattr(git_update, "_run_git", fake_run)
    with pytest.raises(git_update.GitUpdateError):
        git_update.pull("dev")
