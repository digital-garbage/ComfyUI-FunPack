// Keeping the install alive: update, switch branch, roll back, restart ComfyUI. Every one ends in a restart and a page
// reload, so edits are saved first and a card holds the screen until the server answers again.
import { composer as c } from "../composer/composer.js";

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export function createMaintenance({ api, flush }) {
  const tell = (text) => c.toast.warn({ text });

  // -> git status when it is usable, else says why and returns null
  async function status(needClean) {
    let gs;
    try { gs = await api.gitFull(); } catch (err) { return tell(err.message), null; }
    if (!gs || !gs.ok) return tell((gs && gs.detail) || "Git is not available for this install."), null;
    if (needClean && gs.dirty) return tell("Local changes in the FunPack folder: commit or stash them first."), null;
    return gs;
  }
  const ask = (title, message) => c.modal.dialogue({ title, message, confirmLabel: "Go ahead" }).result;

  async function reloadWhenBack(card) {
    const start = Date.now();
    await wait(3500);
    for (;;) {
      try { if ((await api.health()).ok) return location.reload(); } catch { /* still down */ }
      if (Date.now() - start > 90000) card.setText("Still waiting on ComfyUI… it may need a manual restart.");
      await wait(2000);
    }
  }
  async function run(message, action, said) {
    try { await flush(); } catch (err) { return tell(err.message); }
    const text = c.text.md({ text: message });
    const card = c.modal.generic({ title: "Working", size: "sm", closeOnOutside: false, closeOnEsc: false, body: c.region.stack({ gap: "md", children: [c.progress.indeterminate({ label: "Working" }), text] }) });
    card.setText = (t) => text.setText ? text.setText(t) : (text.node.textContent = t);
    try {
      const res = await action();
      if (res && res.restarting === false) {       // nothing restarted: say what is true, and stay on the page
        card.close();
        return c.modal.dialogue({ title: "Not restarted", message: `${res.blocked || (said ? said(res) : "Nothing changed.")} ${res.blocked ? "Restart ComfyUI yourself when the generation has finished." : ""}`.trim(), confirmLabel: "OK", cancelLabel: "Close" }).result;
      }
      const bad = res && res.requirements && res.requirements.ran && !res.requirements.ok;
      if (bad) { await c.modal.dialogue({ title: "Dependencies failed", message: `${res.requirements.detail || ""}\n\nFunPack was updated, but installing its dependencies failed: it may not load until they are installed by hand. ComfyUI will restart now.`, confirmLabel: "Restart", cancelLabel: "Restart" }).result; }
      card.setText(`${said ? said(res) : message}\nRestarting ComfyUI…`);
    } catch (err) {
      if (action.dropsConnection) { /* the server went down as it restarted: expected */ } else { card.close(); return tell(err.message); }
    }
    return reloadWhenBack(card);
  }
  const deps = (res) => (res && res.requirements && res.requirements.ran ? (res.requirements.ok ? " Dependencies installed." : ` Dependency install FAILED: ${String(res.requirements.detail || "").split("\n").slice(0, 2).join(" ")}`) : "");

  return {
    /** What each action would do right now: {update, branch, rollback, ok} for the cards' hints. */
    async hints() {
      let gs; try { gs = await api.gitFull(); } catch { gs = null; }
      if (!gs || !gs.ok) return { ok: false, update: "Git unavailable for this install", branch: "Git unavailable for this install", rollback: "Git unavailable for this install" };
      return { ok: true,
        update: gs.fetch_ok === false ? "Could not reach origin: update status unknown" : gs.dirty ? "Local changes — commit first" : gs.behind > 0 ? `${gs.behind} commit(s) behind origin/${gs.branch}` : `origin/${gs.branch} up to date`,
        branch: `On ${gs.branch}${gs.dirty ? " · local changes — commit first" : " · pick another"}`,
        rollbackOk: Boolean(gs.rollback_target),
        rollback: gs.rollback_target ? `Undo last update — back to ${String(gs.rollback_target.commit).slice(0, 8)}` : "Nothing to undo yet" };
    },
    async update() {
      const gs = await status(true); if (!gs) return;
      const branch = gs.branch || "dev";
      if (!(await ask("Update", `Pull the latest "${branch}" from origin and restart ComfyUI? Any running generation will be lost.${gs.behind > 0 ? ` ${gs.behind} commit(s) available.` : ""}`))) return;
      return run(`Pulling origin/${branch}…`, () => api.git("update", { branch }), (r) => `${r.updated ? `Updated ${r.before} → ${r.after}.` : "Already up to date."}${deps(r)}`);       // missing packages are installed even when the code is current
    },
    async switchBranch() {
      const gs = await status(true); if (!gs) return;
      if (!(gs.branches || []).length) return tell("No git branches found.");
      const branch = await c.modal.choice({ title: "Switch FunPack branch", items: gs.branches.map((b) => ({ id: b, label: b, hint: b === gs.branch ? "current" : "" })) }).result;
      if (!branch || branch === gs.branch) return;
      if (!(await ask("Switch branch", `Switch to "${branch}", pull from origin, and restart ComfyUI? Any running generation will be lost.`))) return;
      return run(`Switching to ${branch}…`, () => api.git("checkout", { branch }), (r) => `On ${branch}${r.updated ? ` (${r.before} → ${r.after})` : ", already up to date"}.${deps(r)}`);
    },
    async rollback() {
      const gs = await status(true); if (!gs) return;
      const target = gs.rollback_target;
      if (!target) return tell("Nothing to roll back to — no update or branch switch on record.");
      if (!(await ask("Roll back", `Roll back to ${String(target.commit).slice(0, 8)}${target.subject ? ` ("${target.subject}")` : ""}? If that update changed requirements.txt, dependencies are not reinstalled: run pip by hand afterwards if things do not load.`))) return;
      return run(`Rolling back to ${String(target.commit).slice(0, 8)}…`, () => api.git("rollback"), (r) => `Rolled back ${r.before} → ${r.after}.`);
    },
    async restart() {
      if (!(await ask("Restart ComfyUI", "Restart now? The server is down for 10–40 s and any running generation is lost. This page reloads when it is back."))) return;
      const action = () => api.git("restart"); action.dropsConnection = true;
      return run("Restarting ComfyUI…", action);
    },
  };
}
