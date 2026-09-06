// ComfyUI's log, in the app.
//
// Users do not open devtools. "It did not work" with nothing to read is where a
// bug report dies, and the one place the reason is written down is a terminal
// they may not even be able to see -- a rented box behind a tunnel has no
// terminal on this desk at all.
//
// It polls only while open, and it stops polling while you are selecting text:
// a panel that redraws under a drag makes copying a stack trace impossible,
// which is the single thing anybody opens a log to do.

import { composer } from "../composer/composer.js";

const EVERY = 2000;

/** mount({ poll, setFooter, close }) -> { node, destroy }. See tempfiles.js's
 *  mount() for why setFooter/close are the only things asked for -- `destroy`
 *  additionally stops the poll, since this one runs a timer nothing else does. */
export function mount({ poll = EVERY, setFooter = () => {}, close = () => {} } = {}) {
  let timer = null;
  let follow = true;                 // stick to the end until the user scrolls off it
  let last = "";

  const view = composer.code.block({ text: "Reading…" });
  const body = composer.region.stack({ gap: "sm", fill: true, children: [view] });

  const selecting = () => {
    const sel = document.getSelection();
    return sel && !sel.isCollapsed && view.node.contains(sel.anchorNode);
  };

  // Following the end is the default and stops being it the moment someone
  // scrolls up -- otherwise reading anything older than two seconds is a fight.
  view.node.addEventListener("scroll", () => {
    const room = view.node.scrollHeight - view.node.scrollTop - view.node.clientHeight;
    follow = room < 40;
  });

  async function refresh() {
    if (selecting()) return;         // never redraw under a drag
    try {
      const res = await fetch("/funpack/api/log?limit=600", { cache: "no-store" });
      const payload = await res.json().catch(() => ({}));
      const text = (payload.lines || []).join("\n")
        || payload.detail
        || "Nothing in the log yet.";
      if (text !== last) {
        last = text;
        view.setText(text);
        if (follow) view.node.scrollTop = view.node.scrollHeight;
      }
      setFooter({ note: payload.path || "", actions: footer() });
    } catch (err) {
      // The server being gone is the most interesting thing a log can say, so it
      // is said -- and the lines already on screen are kept, because they are
      // the ones from just before it went.
      setFooter({ note: `Cannot reach ComfyUI: ${err.message}`, actions: footer() });
    }
  }

  const footer = () => [
    composer.button.md({
      label: "Copy", onClick: () => {
        // Selecting a thousand lines by hand is the other reason this is hard to
        // get out of a log panel.
        if (navigator.clipboard) navigator.clipboard.writeText(last).catch(() => {});
      },
    }),
    composer.button.md({ label: "Close", tone: "primary", onClick: () => close() }),
  ];

  setFooter({ actions: footer() });
  refresh();
  timer = setInterval(refresh, poll);
  return { node: body.node, destroy: () => { clearInterval(timer); timer = null; body.destroy(); } };
}

export function open({ poll = EVERY } = {}) {
  let window_ = null;
  const content = mount({
    poll,
    setFooter: (f) => window_ && window_.setFooter(f),
    close: () => window_ && window_.close("done"),
  });
  window_ = composer.modal.generic({
    title: "ComfyUI log",
    subtitle: "What the server printed. Newest at the bottom.",
    size: "xl",
    body: content,
    onClose: () => { content.destroy(); window_ = null; },
  });
  return window_;
}
