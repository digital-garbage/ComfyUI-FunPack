// Updating FunPack, switching branch, and getting back afterwards.
//
// The most-used feature in a pack whose value is experimental work shipped
// continuously -- so it is a window in the app rather than a terminal command,
// and it is honest about the one thing that makes it frightening: every one of
// these restarts ComfyUI. The window says so before it does it, and then stays
// up, polling, until the server answers again. A restart with nothing on screen
// is indistinguishable from a crash.

import { composer } from "../composer/composer.js";

const BASE = "/funpack/api/git";

async function ask(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
  return payload;
}

export const status = ({ remote = true } = {}) =>
  ask("GET", `/status${remote ? "" : "?remote=0"}`);

/** How far behind, in words. The number alone does not say whether to care. */
function summarise(s) {
  if (!s.ok) return s.detail || "FunPack cannot read its own git checkout.";
  if (!s.checked_remote) return "Checking for updates…";
  if (!s.fetch_ok) return "Could not reach the remote, so this may be out of date.";
  if (s.behind) return `${s.behind} update${s.behind === 1 ? "" : "s"} available.`;
  if (s.ahead) return `Up to date, with ${s.ahead} local commit${s.ahead === 1 ? "" : "s"} of your own.`;
  return "Up to date.";
}

/**
 * The overlay that stays up across a restart.
 *
 * Polls /api/health rather than guessing a duration: an update that installs
 * requirements takes as long as it takes, and a fixed wait either lies or gives
 * up early. Reloads once the server answers, because the code that just changed
 * is the code the page is running.
 */
export function waitForRestart({ poll = 1500, give_up = 180000 } = {}) {
  const overlay = composer.overlay.blocking({
    message: "Restarting ComfyUI. This page comes back on its own.",
  });
  const started = Date.now();

  const tick = async () => {
    try {
      const res = await fetch("/funpack/api/health", { cache: "no-store" });
      if (res.ok) { window.location.reload(); return; }
    } catch { /* still down, which is the expected half of this */ }
    if (Date.now() - started > give_up) {
      overlay.setMessage("ComfyUI has not come back. Check the terminal it was started from.");
      return;
    }
    setTimeout(tick, poll);
  };
  setTimeout(tick, poll);
  return overlay;
}

/**
 * open() -> the window. One at a time.
 *
 * Built from a live status every time it opens: what branch you are on and how
 * far behind are exactly the facts that go stale while a window is closed.
 */
export function open({ onRestart = waitForRestart, running = () => false } = {}) {
  let window_ = null;
  const body = composer.region.stack({ gap: "sm", label: "Updates" });

  const act = async (run) => {
    body.set([composer.hint.default({ text: "Working. This can take a few minutes if "
                                         + "requirements changed." })]);
    try {
      const result = await run();
      // The button is already disabled while a run is in flight, but that check
      // goes stale if the dialog was left open across one starting -- the server
      // re-checks and refuses the restart itself; this is what tells the user why
      // nothing happened instead of it looking like the click did nothing.
      if (result && result.blocked) { load(); return; }
      // The server only restarts when the checkout actually moved. Pressing
      // Update while already up to date is a normal thing to do, and waiting for
      // a restart that is not coming would hang on an overlay for three minutes.
      if (!result || result.restarting === false) { load(); return; }
      if (window_) window_.close("restarting");
      onRestart();
    } catch (err) {
      // Said in the window, where the button was pressed. Every refusal this can
      // give names what to do: commit or stash, install git, pick another branch.
      draw(null, err.message);
    }
  };

  function draw(s, problem) {
    if (problem) {
      body.set([composer.banner.danger({ text: problem }),
                composer.button.md({ label: "Try again", onClick: () => load() })]);
      return;
    }
    if (!s) { body.set([composer.hint.default({ text: "Reading the checkout…" })]); return; }

    const rows = [
      composer.settingsRow.default({ label: "Version", hint: s.codename || undefined,
                                     control: composer.text.sm({ text: s.version || "unknown" }) }),
      composer.settingsRow.default({ label: "Branch", hint: summarise(s),
                                     control: composer.text.sm({ text: s.branch || "?" }) }),
    ];

    // A run in flight is killed by the restart, without warning and with the GPU
    // time already spent. The window will not do that behind someone's back.
    const busy = running();
    if (busy) {
      rows.push(composer.banner.warn({
        text: "A generation is running. Updating or switching branch restarts ComfyUI "
            + "and stops it. Wait for it to finish, or cancel it first.",
      }));
    }

    // The change already landed last time this was tried; only the restart is
    // still owed. Nothing else in this window works until it happens -- the
    // server refuses any further git action while one is pending.
    if (s.restart_pending) {
      rows.push(composer.banner.warn({
        text: "An update already landed and is waiting to restart ComfyUI.",
      }));
      rows.push(composer.settingsRow.default({
        label: "Restart",
        hint: busy ? "Waiting for the generation to finish." : "Ready.",
        control: composer.button.md({ label: "Restart now", tone: "primary", disabled: busy,
                                      onClick: () => act(() => ask("POST", "/restart")) }),
      }));
      body.set([composer.group.default({ rows })]);
      if (window_) window_.setFooter({ actions: [
        composer.button.md({ label: "Close", tone: "ghost", onClick: () => window_.close("done") }),
      ] });
      return;
    }

    if (s.dirty) {
      // The one state that blocks both actions, said before either is pressed
      // rather than as a refusal afterwards.
      rows.push(composer.banner.warn({
        text: "This checkout has local changes. Commit or stash them before updating "
            + "or switching branch.",
      }));
    }

    rows.push(composer.settingsRow.default({
      label: "Switch to",
      hint: "Switching pulls that branch and restarts ComfyUI.",
      control: composer.select.md({
        label: "Branch",
        value: s.branch,
        options: (s.branches || []).map((b) => ({ value: b, label: b })),
        onChange: (branch) => { if (branch !== s.branch) act(() => ask("POST", "/checkout", { branch })); },
        disabled: Boolean(s.dirty) || busy,
      }),
    }));

    if (s.rollback_target) {
      rows.push(composer.settingsRow.default({
        label: "Roll back",
        hint: `Back to ${String(s.rollback_target.commit || "").slice(0, 8)} — the version before the last update.`,
        control: composer.button.md({ label: "Roll back", tone: "danger", disabled: busy,
                                      onClick: () => act(() => ask("POST", "/rollback")) }),
      }));
    }

    body.set([composer.group.default({ rows })]);
    if (window_) {
      window_.setFooter({
        note: s.ok ? "" : "FunPack is not a git checkout, so it cannot update itself.",
        actions: [
          composer.button.md({ label: "Close", tone: "ghost",
                               onClick: () => window_.close("done") }),
          composer.button.md({
            label: s.behind ? `Update (${s.behind})` : "Update", tone: "primary",
            disabled: !s.ok || Boolean(s.dirty) || busy,
            onClick: () => act(() => ask("POST", "/update", {})),
          }),
        ],
      });
    }
  }

  async function load() {
    draw(null);
    // Twice: the checkout alone first, so there is something to read instantly,
    // then again with the remote for ahead and behind. A `git fetch` takes
    // seconds on a good day and blocks until it times out on a machine that
    // cannot reach the network -- and neither is a reason to show nothing.
    let local = null;
    try {
      local = await status({ remote: false });
      draw(local);
    } catch (err) {
      draw(null, err.message);
      return;
    }
    try { draw(await status()); }
    catch { /* the local answer stands, and says it has not checked */ }
  }

  window_ = composer.modal.generic({
    title: "Updates",
    subtitle: "FunPack updates itself from git, then restarts ComfyUI.",
    size: "md",
    body,
    onClose: () => { window_ = null; },
  });
  load();
  return window_;
}
