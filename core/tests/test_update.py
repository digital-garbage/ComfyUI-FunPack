"""rollback() and checkout() against a real git repo -- specifically the
auto-stash behaviour: a dirty tree used to refuse both outright ("commit or
stash them by hand"); it now stashes for you and says so, because whatever
was left uncommitted right before "undo the last update" or "switch branch"
is essentially never the point of doing either.

Real git, not mocks: `_run_git` shells out, and the only thing worth trusting
here is what git itself reports after the fact.
"""

import subprocess

import pytest

from core import update


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "Test")
    # This machine's global git config signs commits by default, which needs
    # a gpg binary the test environment does not have -- irrelevant to what
    # is under test here, and a real commit would fail with a gpg error
    # before ever reaching the code being exercised.
    _git(r, "config", "commit.gpgsign", "false")
    (r / "a.txt").write_text("one\n")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "first")
    monkeypatch.setattr(update, "REPO_ROOT", r)
    return r


# --- _stash_if_dirty, directly ---------------------------------------------

def test_a_clean_tree_is_not_stashed(repo):
    assert update._stash_if_dirty("testing") is None
    assert _git(repo, "stash", "list").stdout.strip() == ""


def test_a_dirty_tree_is_stashed_and_the_message_says_so(repo):
    (repo / "a.txt").write_text("changed\n")
    message = update._stash_if_dirty("testing")
    assert message and "testing" in message
    assert not update._is_dirty()
    assert message in _git(repo, "stash", "list").stdout


def test_an_untracked_file_is_stashed_too(repo):
    """`_is_dirty()` (git status --porcelain) counts an untracked file as
    dirty, so the stash that resolves it has to cover the same ground -- one
    that stashed only tracked changes would leave the tree just as blocked."""
    (repo / "new.txt").write_text("scratch\n")
    assert update._is_dirty()
    update._stash_if_dirty("testing")
    assert not update._is_dirty()
    assert not (repo / "new.txt").exists()


def test_the_stash_is_real_and_recoverable(repo):
    (repo / "a.txt").write_text("changed\n")
    update._stash_if_dirty("testing")
    popped = _git(repo, "stash", "pop")
    assert popped.returncode == 0, popped.stderr
    assert (repo / "a.txt").read_text() == "changed\n"


# --- rollback() -------------------------------------------------------------

def test_rollback_auto_stashes_instead_of_refusing(repo):
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "second")  # HEAD@{1} now exists to roll back to
    (repo / "a.txt").write_text("uncommitted edit\n")

    result = update.rollback()
    assert "stashed" in result, "a dirty tree still refused instead of stashing"
    assert not update._is_dirty()
    assert (repo / "a.txt").read_text() == "one\n"  # actually rolled back


def test_rollback_on_a_clean_tree_says_nothing_was_stashed(repo):
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "second")

    result = update.rollback()
    assert "stashed" not in result


def test_rollback_still_refuses_when_there_is_nothing_to_roll_back_to(repo):
    """Auto-stash is not a reason to skip the OTHER refusal: a fresh repo with
    no prior HEAD position has nothing for a rollback to mean. The target is
    read before anything is stashed, so this refuses before touching the tree
    at all -- stashing first would manufacture a rollback target out of the
    stash's own reflog entry (the real bug this ordering fixes)."""
    (repo / "a.txt").write_text("uncommitted\n")
    with pytest.raises(update.GitUpdateError, match="Nothing to roll back to"):
        update.rollback()
    assert update._is_dirty(), "the tree was stashed despite the rollback being refused"


# --- checkout() --------------------------------------------------------------

def test_checkout_auto_stashes_instead_of_refusing(repo):
    _git(repo, "checkout", "-q", "-b", "other")
    _git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("uncommitted on main\n")

    result = update.checkout("other", pull_after=False)
    assert "stashed" in result
    assert update._current_branch() == "other"
    assert not update._is_dirty()


def test_checkout_on_a_clean_tree_says_nothing_was_stashed(repo):
    _git(repo, "checkout", "-q", "-b", "other")
    _git(repo, "checkout", "-q", "main")

    result = update.checkout("other", pull_after=False)
    assert "stashed" not in result


def test_checkout_still_refuses_a_branch_that_does_not_exist(repo):
    (repo / "a.txt").write_text("uncommitted\n")
    with pytest.raises(update.GitUpdateError, match="not available"):
        update.checkout("nope", pull_after=False)
    # Refused before anything was touched -- including the stash, which would
    # otherwise fire on every malformed request.
    assert update._is_dirty()


def test_a_stash_is_disclosed_even_when_the_checkout_step_itself_then_fails(repo, tmp_path):
    """Found by adversarial review: the stash already happened by the time
    `git checkout` can fail for a reason that has nothing to do with the
    stash (here, the target branch is checked out in another worktree) --
    the raised error used to say nothing about it, leaving local changes
    stashed with no on-screen sign it ever happened."""
    _git(repo, "branch", "other")
    other_wt = tmp_path / "other-worktree"
    added = _git(repo, "worktree", "add", str(other_wt), "other")
    assert added.returncode == 0, added.stderr

    (repo / "a.txt").write_text("uncommitted\n")
    with pytest.raises(update.GitUpdateError, match="stashed first"):
        update.checkout("other", pull_after=False)

    # The stash is real, not just claimed in the message.
    assert "FunPack: auto-stashed" in _git(repo, "stash", "list").stdout


def test_a_stash_is_disclosed_when_pull_fails_after_a_successful_switch(repo):
    """checkout(pull_after=True) -- the actual default the real "Switch to"
    control uses -- checks out the branch (succeeds) and then calls pull()
    (which can still fail on its own, here because no "origin" remote is
    configured). Round one of this fix only wrapped the checkout call
    itself; found by a second review round that the far more common
    default-path failure, inside pull(), was still silently losing the
    stash disclosure the exact same way."""
    _git(repo, "checkout", "-q", "-b", "other")
    _git(repo, "checkout", "-q", "main")
    (repo / "a.txt").write_text("uncommitted\n")

    with pytest.raises(update.GitUpdateError, match="stashed first"):
        update.checkout("other", pull_after=True)

    # The checkout itself succeeded before pull() failed.
    assert update._current_branch() == "other"
    assert "FunPack: auto-stashed" in _git(repo, "stash", "list").stdout


def test_a_stash_is_disclosed_when_the_reset_itself_then_fails(repo, monkeypatch):
    """rollback()'s own `git reset --hard` can fail too, after a successful
    stash -- simulated here (a healthy `git reset --hard` to a real commit
    is not something that fails on demand), but the code path and the
    stash-loss it would otherwise cause are real."""
    _git(repo, "commit", "--allow-empty", "-q", "-m", "second")  # a real HEAD@{1}
    (repo / "a.txt").write_text("uncommitted\n")

    real_run_git = update._run_git

    def fake_run_git(*args, **kwargs):
        if args and args[0] == "reset":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="simulated reset failure")
        return real_run_git(*args, **kwargs)

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    with pytest.raises(update.GitUpdateError, match="stashed first"):
        update.rollback()

    assert "FunPack: auto-stashed" in _git(repo, "stash", "list").stdout


def test_a_stash_is_disclosed_even_when_the_failure_is_not_a_gitupdateerror(repo, monkeypatch):
    """`_run_git` can raise something other than returning a failing
    CompletedProcess -- a hung `git fetch` timing out
    (subprocess.TimeoutExpired) on a slow or flaky network is the real case,
    exactly the kind of connection a rented GPU box has. Round three: the
    first two fixes only caught `GitUpdateError`, so a timeout (or anything
    else `subprocess.run` can raise) sailed past the catch with the stash
    disclosure lost the same way, one exception type later."""
    _git(repo, "commit", "--allow-empty", "-q", "-m", "second")
    (repo / "a.txt").write_text("uncommitted\n")

    real_run_git = update._run_git

    def fake_run_git(*args, **kwargs):
        if args and args[0] == "reset":
            raise subprocess.TimeoutExpired(cmd=["git", *args], timeout=120)
        return real_run_git(*args, **kwargs)

    monkeypatch.setattr(update, "_run_git", fake_run_git)

    with pytest.raises(update.GitUpdateError, match="stashed first"):
        update.rollback()

    assert "FunPack: auto-stashed" in _git(repo, "stash", "list").stdout


def test_a_missing_requirement_is_installed_by_its_whole_line_so_a_pin_or_url_survives(repo, monkeypatch):
    (repo / "requirements.txt").write_text(
        "# comment\nnumpy\nzz_absent_pkg>=3.8,<3.9  # why\nzz_model @ https://example.com/zz_model.whl\n")
    monkeypatch.setattr(update, "_installed_version", lambda n: "1.0" if n == "numpy" else None)
    ran = []
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **k: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(update, "_pip_freeze", lambda: {"numpy": "1.26.4"})
    assert update.install_requirements()["ok"]
    assert [c[-1] for c in ran] == ["zz_absent_pkg>=3.8,<3.9", "zz_model @ https://example.com/zz_model.whl"]


def test_one_package_that_cannot_install_does_not_stop_the_others(repo, monkeypatch):
    (repo / "requirements.txt").write_text("zz_bad_url @ https://example.com/missing.whl\nzz_good_pkg\n")
    monkeypatch.setattr(update, "_installed_version", lambda n: None)
    ran = []

    def fake_run(cmd, **k):
        ran.append(cmd[-1])
        return subprocess.CompletedProcess(cmd, 1 if "zz_bad_url" in cmd[-1] else 0, "", "404 not found")

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    monkeypatch.setattr(update, "_pip_freeze", lambda: {"numpy": "1.26.4"})
    out = update.install_requirements()
    assert ran == ["zz_bad_url @ https://example.com/missing.whl", "zz_good_pkg"], "the second still installs"
    assert out["ok"] is False and "zz_bad_url" in out["detail"] and "404 not found" in out["detail"]
    assert "zz_good_pkg" not in out["detail"].split("Could not install:")[1].split("\n")[0]


def test_a_branch_switch_installs_what_is_missing_even_when_the_requirements_file_did_not_change(repo, monkeypatch):
    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "b.txt").write_text("x\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "other")
    _git(repo, "checkout", "-q", "main")
    called = []
    monkeypatch.setattr(update, "install_requirements", lambda: called.append(1) or {"ok": True})
    update.checkout("other", pull_after=False, install_deps=True)
    assert called, "the switch itself must check for missing packages"


def test_a_spacy_model_that_does_not_load_is_downloaded_on_the_switch(repo, monkeypatch):
    (repo / "requirements.txt").write_text("spacy>=3.8,<3.9\nen_core_web_sm @ https://example.com/m.whl\n")
    monkeypatch.setattr(update, "_installed_version", lambda n: "3.8.0")
    ran = []
    monkeypatch.setattr(update, "_spacy_model_loads", lambda: bool(ran) and ran[-1] == ["spacy", "download", "en_core_web_sm"])
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **k: ran.append(cmd[-3:]) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(update, "_pip_freeze", lambda: {"numpy": "1.26.4"})
    assert update.install_requirements()["ok"]
    assert ran[-1] == ["spacy", "download", "en_core_web_sm"]


def test_an_install_pins_every_installed_package_so_nothing_already_here_moves(repo, monkeypatch):
    (repo / "requirements.txt").write_text("zz_new_pkg\n")
    monkeypatch.setattr(update, "_installed_version", lambda n: None)
    seen = {}

    def fake_run(cmd, **k):
        if "-c" in cmd:
            seen["constraints"] = open(cmd[cmd.index("-c") + 1]).read()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    monkeypatch.setattr(update, "_pip_freeze", lambda: {"numpy": "1.26.4", "torch": "2.9.0"})
    assert update.install_requirements()["ok"]
    assert "numpy==1.26.4" in seen["constraints"] and "torch==2.9.0" in seen["constraints"]


def test_an_unreadable_installed_list_installs_nothing(repo, monkeypatch):
    (repo / "requirements.txt").write_text("zz_new_pkg\n")
    monkeypatch.setattr(update, "_installed_version", lambda n: None)
    ran = []
    monkeypatch.setattr(update.subprocess, "run", lambda cmd, **k: ran.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(update, "_pip_freeze", lambda: {})
    out = update.install_requirements()
    assert out["ok"] is False and ran == []
