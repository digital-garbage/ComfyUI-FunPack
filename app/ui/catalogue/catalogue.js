// The token sheet.
//
// It reads the custom properties out of the loaded stylesheet rather than
// listing them, so it cannot drift from tokens.css: a token added there shows up
// here on reload, and one deleted there disappears. Same inversion the kit uses.

const GROUPS = [
  { id: "font",    title: "Font stacks",   match: (n) => n.startsWith("--font-"),  render: renderFont },
  { id: "type",    title: "Type scale",    match: (n) => /^--fs-/.test(n),         render: renderType,
    // each step shows its own leading, so --lh-N is spoken for here
    also: (names) => names.map((n) => n.replace("--fs-", "--lh-")) },
  { id: "lh",      title: "Line heights",  match: (n) => /^--lh-(tight|snug|loose)$/.test(n), render: renderPlain },
  { id: "weight",  title: "Weights",       match: (n) => n.startsWith("--fw-"),    render: renderWeight },
  { id: "space",   title: "Spacing",       match: (n) => /^--sp-\d+$/.test(n),     render: renderBar },
  { id: "radius",  title: "Radius",        match: (n) => /^--r-/.test(n),          render: renderRadius },
  { id: "control", title: "Controls",      match: (n) => n.startsWith("--ctl-"),   render: renderControl },
  { id: "elev",    title: "Elevation",     match: (n) => n.startsWith("--elev-"),  render: renderElev },
  { id: "shadow",  title: "Shadows",       match: (n) => /^--shadow-\d|^--focus-ring$/.test(n), render: renderShadow },
  { id: "motion",  title: "Motion",        match: (n) => /^--dur-|^--ease-/.test(n), render: renderMotion },
  { id: "state",   title: "States",        match: (n) => /^--(bw|disabled-opacity)$/.test(n), render: renderState },
  { id: "density", title: "Density",       match: (n) => n.startsWith("--cell-"),  render: renderBar },
  { id: "layout",  title: "Layout",        match: (n) => /^--menubar-h$/.test(n),  render: renderBar },
  // Colour is last and greedy: anything left that a browser accepts as a colour.
  { id: "colour",  title: "Colour",        match: isColourName,                    render: renderSwatch },
];

const COLOUR_HINT = /^--(elev|hover|active|line|text|muted|faint|accent|on-accent|teal|violet|danger|warn|good|pink|void|shadow-ink|backdrop)/;

function isColourName(name) {
  return COLOUR_HINT.test(name);
}

/** Every custom property declared in the kit's stylesheet, in declaration order. */
function declaredTokens() {
  const names = [];
  const seen = new Set();

  const walk = (sheet) => {
    let rules;
    try { rules = sheet.cssRules; } catch { return; }   // cross-origin: skip
    if (!rules) return;
    for (const rule of rules) {
      if (rule.styleSheet) { walk(rule.styleSheet); continue; }   // @import

      // Collect BEFORE recursing. Since CSS nesting shipped, every CSSStyleRule
      // carries a (usually empty) .cssRules list — treating that as "this is a
      // container, skip its properties" silently finds nothing at all.
      if (rule.style && rule.selectorText && /:root|\[data-theme/.test(rule.selectorText)) {
        for (const prop of rule.style) {
          if (prop.startsWith("--") && !seen.has(prop)) { seen.add(prop); names.push(prop); }
        }
      }

      if (rule.cssRules && rule.cssRules.length) walk(rule);      // @media, nesting
    }
  };

  for (const sheet of document.styleSheets) walk(sheet);
  return names;
}

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function row(name, value, visual) {
  const r = el("div", "cat-row");
  r.append(el("code", "cat-name", name), el("code", "cat-val", value));
  r.append(visual || el("div"));
  return r;
}

// ── renderers ──────────────────────────────────────────────────────────────

function renderPlain(name, value) { return row(name, value, el("div")); }

function renderFont(name, value) {
  const s = el("div", "cat-type", "Grid tuning — 0123456789");
  s.style.fontFamily = value;
  s.style.fontSize = "15px";
  return row(name, value.split(",")[0].replace(/"/g, ""), s);
}

function renderType(name, value, read) {
  const step = name.replace("--fs-", "");
  const lh = read(`--lh-${step}`);
  const s = el("div", "cat-type", "The quick brown fox");
  s.style.fontSize = value;
  s.style.lineHeight = lh;
  if (step.startsWith("d")) s.style.fontFamily = "var(--font-display)";
  return row(name, `${value} / ${lh}`, s);
}

function renderWeight(name, value) {
  const s = el("div", "cat-type", "The quick brown fox");
  s.style.fontWeight = value;
  s.style.fontSize = "15px";
  return row(name, value, s);
}

function renderBar(name, value) {
  const b = el("div", "cat-bar-fill");
  b.style.width = value;
  return row(name, value, b);
}

function renderRadius(name, value) {
  const b = el("div", "cat-box");
  b.style.borderRadius = value;
  return row(name, value, b);
}

function renderControl(name, value) {
  if (/^--ctl-h/.test(name)) {
    const c = el("div", "cat-ctl", "Button");
    c.style.height = value;
    return row(name, value, c);
  }
  return renderBar(name, value);
}

function renderElev(name, value) {
  const b = el("div", "cat-elev", value);
  b.style.background = value;
  return row(name, value, b);
}

function renderShadow(name, value) {
  const b = el("div", "cat-elev", "");
  b.style.background = "var(--elev-2)";
  b.style.boxShadow = value;
  b.style.width = "150px";
  return row(name, value.length > 34 ? value.slice(0, 32) + "…" : value, b);
}

function renderMotion(name, value) {
  if (name.startsWith("--ease-")) return renderPlain(name, value);
  const track = el("div", "cat-motion-track");
  const dot = el("div", "cat-motion");
  dot.style.transitionDuration = value;
  track.append(dot);
  return row(name, `${value} · hover`, track);
}

function renderState(name, value) {
  const b = el("div", "cat-ctl", name === "--bw" ? "border width" : "disabled");
  b.style.height = "var(--ctl-h-lg)";
  if (name === "--bw") b.style.borderWidth = value;
  else b.style.opacity = value;
  return row(name, value, b);
}

function renderSwatch(name, value) {
  const s = el("div", "cat-swatch");
  s.style.background = value;
  return row(name, value, s);
}

// ── page ───────────────────────────────────────────────────────────────────

function buildPane(theme, names, group) {
  const pane = el("section", "cat-pane");
  pane.setAttribute("data-theme", theme);
  pane.append(el("h3", null, theme));

  // Appended before reading, so getComputedStyle resolves against this pane's
  // own data-theme rather than the document's.
  document.body.append(pane);
  const read = (n) => getComputedStyle(pane).getPropertyValue(n).trim();

  const rows = el("div", "cat-rows");
  for (const name of names) rows.append(group.render(name, read(name), read));
  pane.append(rows);
  return pane;
}

function render(sideBySide) {
  const page = document.querySelector("#catalogue");
  page.textContent = "";

  const all = declaredTokens();
  const taken = new Set();

  for (const group of GROUPS) {
    const names = all.filter((n) => !taken.has(n) && group.match(n));
    names.forEach((n) => taken.add(n));
    if (group.also) group.also(names).forEach((n) => taken.add(n));
    if (!names.length) continue;

    const section = el("section", "cat-section");
    section.append(el("h2", null, group.title));
    section.append(el("p", null, `${names.length} token${names.length === 1 ? "" : "s"}`));

    const panes = el("div", "cat-panes" + (sideBySide ? "" : " single"));
    section.append(panes);
    page.append(section);

    const themes = sideBySide ? ["dark", "light"] : [document.documentElement.getAttribute("data-theme") || "dark"];
    for (const t of themes) panes.append(buildPane(t, names, group));
  }

  const leftover = all.filter((n) => !taken.has(n));
  if (leftover.length) {
    const s = el("section", "cat-section");
    s.append(el("h2", null, "Ungrouped"));
    s.append(el("p", null, "Declared in tokens.css but not claimed by any group above — either the token is new or the catalogue needs a renderer for it."));
    const panes = el("div", "cat-panes single");
    panes.append(buildPane(document.documentElement.getAttribute("data-theme") || "dark", leftover, { render: renderPlain }));
    s.append(panes);
    page.append(s);
  }
}

let sideBySide = true;

document.querySelector("#toggle-panes").addEventListener("click", (e) => {
  sideBySide = !sideBySide;
  e.currentTarget.setAttribute("aria-pressed", String(sideBySide));
  render(sideBySide);
});

for (const btn of document.querySelectorAll("[data-theme-set]")) {
  btn.addEventListener("click", () => {
    window.ComposerTheme.apply(btn.getAttribute("data-theme-set"));
    if (!sideBySide) render(false);
  });
}

// composer.css reaches the tokens through @import, and an imported sheet's
// `.styleSheet` is still null while it is in flight — a module script runs well
// before that. Wait for load, when every sheet in the chain is parsed.
function stylesheetsReady() {
  if (document.readyState === "complete") return Promise.resolve();
  return new Promise((r) => window.addEventListener("load", r, { once: true }));
}

stylesheetsReady().then(() => {
  if (!declaredTokens().length) {
    // A blank catalogue is the one thing this page must never be: it looks like
    // "no tokens" when it means "could not read them".
    const page = document.querySelector("#catalogue");
    page.textContent = "";
    const s = el("section", "cat-section");
    s.append(el("h2", null, "No tokens found"));
    s.append(el("p", null,
      "composer.css loaded but no custom properties could be read from it. " +
      "Either tokens.css failed to load, or its :root rules were renamed."));
    page.append(s);
    return;
  }
  render(sideBySide);
});
