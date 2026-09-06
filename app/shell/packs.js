// Node packs: install, update, remove.
//
// A stand-in for ComfyUI-Manager's three operations, so somebody adding the one
// pack a workflow needs is not sent to another UI to do it. There is no
// catalogue and no curation -- the URL is theirs -- which keeps this honest
// about what it is: git, in a directory.
//
// Everything here can take minutes. Nothing pretends otherwise: the row that is
// working says so, and the rest of the list stays usable.

import { composer } from "../composer/composer.js";

const BASE = "/funpack/api/packs";

async function ask(method, path = "", body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.detail || `HTTP ${res.status}`);
  return payload;
}

/** What a pack's git state says, in words rather than two numbers. */
function saySince(pack, checked) {
  if (!pack.git) return "not a git checkout — update and remove are not offered";
  const news = checked && checked[pack.name];
  if (!news) return `${pack.branch || "?"} · ${pack.commit || ""}`.trim();
  if (!news.checked) return `${pack.branch || "?"} · could not check: ${news.reason}`;
  if (news.behind) return `${news.behind} update${news.behind === 1 ? "" : "s"} available`;
  return `${pack.branch || "?"} · up to date`;
}

export function open() {
  let window_ = null;
  let listing = null;
  let checked = null;
  let busy = null;              // the pack currently being worked on, by name
  // What went wrong, held rather than passed: every redraw would otherwise wipe
  // it, and the one after a failure arrives immediately -- the message appeared
  // and vanished in the same frame.
  let problem = null;

  const body = composer.region.stack({ gap: "sm", label: "Node packs" });

  const run = async (name, work) => {
    busy = name;
    problem = null;
    draw();
    try {
      await work();
      await load();
    } catch (err) {
      problem = err.message;
    } finally {
      busy = null;
      draw();
    }
  };

  function rows() {
    const out = [];
    if (problem) out.push(composer.banner.danger({ text: problem }));

    const url = composer.input.md({ label: "Git URL",
                                    placeholder: "https://github.com/owner/pack" });
    out.push(composer.settingsRow.default({
      label: "Install a pack",
      hint: "Anything git can clone. ComfyUI has to be restarted before a new pack "
          + "loads — Settings ▸ Updates can do that.",
      control: composer.button.md({
        label: busy === "*" ? "Installing…" : "Install",
        disabled: Boolean(busy),
        onClick: () => {
          const value = String(url.value || "").trim();
          // Said here rather than sent: an empty URL is not a server's problem.
          if (!value) { problem = "Paste a git URL first."; draw(); return; }
          run("*", () => ask("POST", "/install", { url: value }));
        },
      }),
    }));
    out.push(composer.field.default({ label: "Git URL", control: url }));

    if (!listing) {
      out.push(composer.hint.default({ text: "Reading custom_nodes…" }));
      return out;
    }

    out.push(composer.label.section({ text: `${listing.nodes.length} installed` }));
    for (const pack of listing.nodes) {
      const working = busy === pack.name;
      out.push(composer.settingsRow.default({
        label: pack.name + (pack.is_funpack ? " — this is FunPack" : ""),
        hint: working ? "Working…" : saySince(pack, checked),
        // Buttons, not a button GROUP: these are two actions, not a choice
        // between two states. A radio group also leaves one of them looking
        // chosen afterwards, which for "Remove" is an alarming thing to look at.
        control: composer.toolbar.default({
          label: pack.name,
          items: [
            ...(pack.git && !pack.is_funpack ? [composer.button.sm({
              label: "Update", disabled: Boolean(busy),
              onClick: () => run(pack.name, () => ask("POST", "/update", { name: pack.name })),
            })] : []),
            ...(pack.is_funpack ? [] : [composer.button.sm({
              label: "Remove", tone: "danger", disabled: Boolean(busy),
              onClick: () => run(pack.name, () => ask("POST", "/remove", { name: pack.name })),
            })]),
          ],
        }),
      }));
    }
    return out;
  }

  function draw() {
    body.set(rows());
    if (window_) {
      window_.setFooter({
        note: listing ? listing.root : "",
        actions: [
          composer.button.md({ label: "Check for updates", disabled: Boolean(busy),
                               onClick: () => check() }),
          composer.button.md({ label: "Close", tone: "primary",
                               onClick: () => window_.close("done") }),
        ],
      });
    }
  }

  async function load() {
    try { listing = await ask("GET"); }
    catch (err) { listing = null; problem = err.message; }
    draw();
  }

  async function check() {
    // Fetching every pack is network-bound and slow; the button says what it is
    // doing rather than looking broken for twenty seconds.
    busy = "*check*";
    draw();
    problem = null;
    // `checked`, which is the key the server sends. It said `nodes` here for a
    // while: the request succeeded, the JSON was valid, and every pack reported
    // nothing to update no matter how far behind it was.
    try { checked = (await ask("POST", "/check")).checked || null; }
    catch (err) { problem = err.message; }
    finally { busy = null; draw(); }
  }

  window_ = composer.modal.generic({
    title: "Node packs",
    subtitle: "Install, update and remove ComfyUI custom nodes.",
    size: "lg",
    body,
    onClose: () => { window_ = null; },
  });
  load();
  return window_;
}
