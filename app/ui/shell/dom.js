// The page, as a service. A feature never names `document` or `window`: it asks `app.dom`.
// Only this file and the kit reach the page, so what a feature may do there is this short list.
export function createDom(doc = document) {
  const win = doc.defaultView;
  const field = /^(input|textarea)$/i;
  const on = (target, type, fn, opts) => { target.addEventListener(type, fn, opts); return () => target.removeEventListener(type, fn, opts); };
  const link = (href, download) => Object.assign(doc.createElement("a"), { href, download }).click();
  return {
    /** The caret is in a text field inside `node`: a redraw there would eat keystrokes. `only` narrows it to one tag. */
    typingIn(node, only) { const a = doc.activeElement; return !!a && node.contains(a) && (only ? a.tagName === only : field.test(a.tagName)); },
    /** `node` itself has the focus. */
    focusOn: (node) => doc.activeElement === node,
    /** The focus is `node` or something inside it. */
    holdsFocus: (node) => !!doc.activeElement && node.contains(doc.activeElement),
    /** Let go of whatever has the focus, so a field being typed in commits first. */
    blur() { const a = doc.activeElement; if (a && a !== doc.body && typeof a.blur === "function") a.blur(); },
    /** Listen on the page / the window. Each returns the function that stops listening. */
    onPage: (type, fn, opts) => on(doc, type, fn, opts),
    onWindow: (type, fn, opts) => on(win, type, fn, opts),
    /** A dialog is open, so nothing behind it should react. */
    dialogOpen: () => !!doc.querySelector('[role="dialog"]'),
    /** Text is being selected inside `node` (a redraw would cancel the drag). */
    selectingIn(node) { const s = doc.getSelection(); return !!s && !s.isCollapsed && node.contains(s.anchorNode); },
    /** Save a file from a URL the browser may fetch. */
    download: (href, name) => link(href, name),
    /** Save something built in the page (a Blob). */
    saveBlob(blob, name) { const href = URL.createObjectURL(blob); link(href, name); URL.revokeObjectURL(href); },
    /** Open a URL in a new tab. */
    openTab: (url) => win.open(url, "_blank", "noopener"),
    /** Ask for a file with the browser's own chooser -> the File, or null when it was cancelled. */
    pickFile: (accept) => new Promise((resolve) => {
      const input = Object.assign(doc.createElement("input"), { type: "file", accept });
      input.onchange = () => resolve(input.files[0] || null);
      input.oncancel = () => resolve(null);
      input.click();
    }),
    /** How long a media file is, from the browser's own reading of it (null when it cannot say). */
    length: (kind, src) => new Promise((resolve) => {
      const m = doc.createElement(kind === "audio" ? "audio" : "video");
      const done = (v) => { m.removeAttribute("src"); m.load(); resolve(v); };
      m.preload = "metadata";
      m.onloadedmetadata = () => done(Number.isFinite(m.duration) ? m.duration : null);
      m.onerror = () => done(null);
      setTimeout(() => done(null), 8000);
      m.src = src;
    }),
    /** A part of the screen by name: "menubar", "main", "zone:0" (Assets, Preview, Properties, Timeline) or "button:<label start>". */
    landmark(name) {
      if (name === "menubar") return doc.querySelector(".cx-frame-head");
      if (name === "main") return doc.querySelector(".cx-frame-main");
      const [kind, rest] = name.split(/:(.*)/s);
      if (kind === "zone") return doc.querySelectorAll("section.cx-zone")[Number(rest)] || null;
      if (kind === "button") return [...doc.querySelectorAll("button")].find((b) => b.textContent.trim().startsWith(rest)) || null;
      return null;
    },
    /** The page's body, as an anchor for things that float. */
    body: doc.body,
  };
}
