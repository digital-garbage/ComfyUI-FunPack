"""Git update helpers for the Movie Editor."""

import sys
import types
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


# ── requirements installed as part of an update ───────────────────────────────
# The gap this closes: an in-app update pulled new code but never new dependencies, so a
# release that added one left the node pack unable to import, with nothing said.

def test_requirements_change_is_detected(monkeypatch):
    from movie_editor.backend import git_update as gu
    calls = []

    def fake_git(*args, **kw):
        calls.append(args)
        return types.SimpleNamespace(returncode=0, stdout="requirements.txt\n", stderr="")

    monkeypatch.setattr(gu, "_run_git", fake_git)
    assert gu.requirements_changed("aaa", "bbb") is True
    assert any("diff" in a for a in calls[0])


def test_untouched_requirements_are_not_reinstalled(monkeypatch):
    from movie_editor.backend import git_update as gu
    monkeypatch.setattr(gu, "_run_git", lambda *a, **k: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    assert gu.requirements_changed("aaa", "bbb") is False


def test_no_update_means_no_install(monkeypatch):
    """Same commit before and after: nothing changed, so nothing to install."""
    from movie_editor.backend import git_update as gu
    monkeypatch.setattr(gu, "_run_git", lambda *a, **k: pytest.fail("git should not run"))
    assert gu.requirements_changed("aaa", "aaa") is False
    assert gu.requirements_changed("", "bbb") is False


def test_an_undecidable_diff_installs_rather_than_skips(monkeypatch):
    """A redundant pip run costs seconds; a skipped one costs a node pack that will not
    import and an error that does not name its cause."""
    from movie_editor.backend import git_update as gu
    monkeypatch.setattr(gu, "_run_git", lambda *a, **k: types.SimpleNamespace(
        returncode=128, stdout="", stderr="bad object"))
    assert gu.requirements_changed("aaa", "bbb") is True


MISSING = {"missing": ["gguf"], "below_floor": [], "present": ["numpy"]}


def _fake_pip(monkeypatch, gu, status=None, freezes=None, rc=0, stderr=""):
    """Drive install_requirements without touching the real environment."""
    monkeypatch.setattr(gu, "requirement_status", lambda: dict(status or MISSING))
    calls = []
    seq = iter(freezes) if freezes else None

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[3] == "freeze":
            return types.SimpleNamespace(
                returncode=0, stdout=(next(seq) if seq else ""), stderr="")
        return types.SimpleNamespace(returncode=rc, stdout="ok", stderr=stderr)

    monkeypatch.setattr(gu.subprocess, "run", fake_run)
    return calls


# ── only what is missing ──────────────────────────────────────────────────────
# Upgrading a package the rest of ComfyUI is built against is how a working install stops
# working: torch, numpy, transformers and the compiled extensions on top of them do not
# survive being moved underneath, and the failure is a segfault hours later, not an error.


def test_only_the_absent_packages_are_installed(monkeypatch):
    from movie_editor.backend import git_update as gu
    calls = _fake_pip(monkeypatch, gu)
    out = gu.install_requirements()
    install = [c for c in calls if c[3] == "install"][0]
    assert install[-1:] == ["gguf"]                 # only the missing one, by name
    assert "-r" not in install                       # never the whole requirements file
    assert out["ok"] and out["ran"]


def test_nothing_missing_means_pip_is_never_run(monkeypatch):
    from movie_editor.backend import git_update as gu
    calls = _fake_pip(monkeypatch, gu, {"missing": [], "below_floor": [], "present": ["numpy"]})
    out = gu.install_requirements()
    assert not calls and out["ran"] is False and out["ok"]
    assert "already present" in out["detail"]


def test_an_outdated_package_is_reported_and_left_alone(monkeypatch):
    """Present but below the floor: say so, hand over the command, change nothing."""
    from movie_editor.backend import git_update as gu
    calls = _fake_pip(monkeypatch, gu, {
        "missing": [], "below_floor": [("transformers", "4.44.0", ">=5.0.0")],
        "present": ["transformers"]})
    out = gu.install_requirements()
    assert not calls
    assert "transformers: have 4.44.0, wants >=5.0.0" in out["detail"]
    assert "LEFT ALONE" in out["detail"]


def test_install_uses_the_running_interpreter(monkeypatch):
    """Never a bare `pip`: ComfyUI is usually in a venv, and the pip on PATH belongs to
    whatever else is on it — installing into the wrong environment succeeds and changes
    nothing."""
    from movie_editor.backend import git_update as gu
    calls = _fake_pip(monkeypatch, gu)
    gu.install_requirements()
    assert all(c[0] == sys.executable and c[1:3] == ["-m", "pip"] for c in calls)


def test_the_install_reports_which_packages_moved(monkeypatch):
    """Installing a missing package still resolves ITS dependencies, so the freeze diff is
    what says whether anything else moved."""
    from movie_editor.backend import git_update as gu
    _fake_pip(monkeypatch, gu, freezes=["numpy==1.26.4\n",
                                        "numpy==2.1.0\ngguf==0.10.0\n"])
    changed = gu.install_requirements()["changed"]
    assert "numpy 1.26.4 -> 2.1.0" in changed
    assert "gguf 0.10.0 (new)" in changed


def test_an_unreadable_freeze_reports_nothing_rather_than_everything(monkeypatch):
    """Reporting every package as new because pip freeze failed is worse than silence."""
    from movie_editor.backend import git_update as gu
    monkeypatch.setattr(gu, "requirement_status", lambda: dict(MISSING))
    monkeypatch.setattr(gu, "_pip_freeze", lambda: {})
    monkeypatch.setattr(gu.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout="ok", stderr=""))
    out = gu.install_requirements()
    assert out["ok"] and out["changed"] == []


def test_a_failed_install_reports_the_command_and_never_raises(monkeypatch):
    """The code IS updated by this point — turning that into an exception would leave the
    user with a half-finished update and no instructions."""
    from movie_editor.backend import git_update as gu
    _fake_pip(monkeypatch, gu, rc=1, stderr="No matching distribution found for gguf")
    out = gu.install_requirements()
    assert out["ran"] is True and out["ok"] is False
    assert "pip install gguf" in out["detail"]
    assert "No matching distribution" in out["detail"]


# ── parsing ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("line,expected", [
    ("numpy", ("numpy", "")),
    ("transformers>=5.0.0", ("transformers", ">=5.0.0")),
    ("opencv-python-headless", ("opencv-python-headless", "")),
    ("pillow == 12.0 ; python_version > '3.8'", ("pillow", "== 12.0")),
    ("gguf  # a comment", ("gguf", "")),
    ("", None),
    ("   ", None),
    ("# just a comment", None),
    ("--extra-index-url https://x", None),
])
def test_requirement_lines(line, expected):
    from movie_editor.backend import git_update as gu
    assert gu.parse_requirement(line) == expected


def test_a_pip_timeout_is_reported_not_raised(monkeypatch):
    from movie_editor.backend import git_update as gu
    monkeypatch.setattr(gu, "requirement_status", lambda: dict(MISSING))

    def boom(cmd, **kw):
        if cmd[3] == "freeze":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        raise gu.subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(gu.subprocess, "run", boom)
    out = gu.install_requirements()
    assert out["ok"] is False and "did not finish" in out["detail"]
