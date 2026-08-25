// Appearance: the colour-scheme picker, shared by the Cutting Room and Easy Gen.
// Registered as its own Settings section rather than folded into Editor settings —
// the theme is a property of this browser, not of the open project, and putting it
// beside settings that travel with a project would say otherwise.
(function () {
  const { el } = window.dom;
  const T = window.FunPackTheme;

  const CARDS = [
    { key: "light", title: "Light", hint: "White, grey and daylight blue." },
    { key: "dark", title: "Dark", hint: "The cutting-room palette. Warm near-black and amber." },
    { key: "auto", title: "Auto", hint: "Follow the system setting and switch with it." },
  ];

  // The swatch is drawn from the tokens of the theme it advertises, not from the
  // active one — a light card has to look light while you are still in the dark theme.
  function swatch(key) {
    const box = el("div", "theme-swatch theme-swatch-" + key);
    ["bar", "panel", "accent"].forEach((part) => box.append(el("span", "theme-swatch-" + part)));
    return box;
  }

  function buildPicker(onPick) {
    const grid = el("div", "theme-cards");
    const cards = {};
    const paint = (active) => {
      Object.entries(cards).forEach(([k, c]) => c.classList.toggle("active", k === active));
    };
    CARDS.forEach(({ key, title, hint }) => {
      const c = el("button", "theme-card");
      c.type = "button";
      c.append(swatch(key));
      const meta = el("div", "theme-card-meta");
      meta.append(el("div", "theme-card-title", title));
      meta.append(el("div", "theme-card-hint", hint));
      c.append(meta);
      // Click only — no hover preview. Repainting the whole app on a mouse-over made the
      // theme feel like it was changing on its own, and left the picker guessing which
      // choice was the real one.
      c.onclick = () => { T.apply(key); paint(key); onPick && onPick(key); };
      cards[key] = c;
      grid.append(c);
    });
    paint(T.get());
    return { grid, paint };
  }

  // Same reasoning as the colour scheme: how much room the timeline should take is a
  // property of the screen you are at, not of the project, so it belongs here rather than
  // in the settings that travel with one.
  function timelineRow() {
    const P = window.TimelinePeek;
    const row = el("div", "es-row");
    const lbl = el("label", "chk es-toggle");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = !!(P && P.get());
    cb.onchange = () => P && P.set(cb.checked);
    lbl.append(cb, el("span", null, "Show timeline on hover"));
    row.append(lbl);
    row.append(el("div", "es-hint",
      "The timeline sits as a strip — title, transport and Composer stay visible — and opens "
      + "to full height while the pointer is on it. It also opens while you drag a file over "
      + "it, so dropping onto a lane works the same as when it is pinned open."));
    return row;
  }

  function mount(body) {
    const wrap = el("div", "sw-stack");
    wrap.append(el("div", "sw-rows-label", "Colour scheme"));
    const { grid, paint } = buildPicker();
    wrap.append(grid);
    wrap.append(el("div", "sw-hint",
      "Stored in this browser, not in the project — a rented instance you open the editor on "
      + "starts on Dark until you set it there too."));
    wrap.append(el("div", "sw-rows-label", "Timeline"));
    wrap.append(timelineRow());
    body.append(wrap);
    const off = T.onChange(() => paint(T.get()));
    return () => off();
  }

  window.SettingsWindow.register({
    id: "appearance", group: "", order: 0, title: "Appearance",
    subtitle: "Light, dark, or follow the system.",
    keywords: "theme colour color scheme light dark auto appearance contrast palette "
            + "timeline hover peek collapse strip auto-hide autohide",
    iconBg: "linear-gradient(180deg,#8fd0ff,#3a7fd5)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M8 1.6a6.4 6.4 0 1 0 0 12.8V1.6z" fill="#fff"/><circle cx="8" cy="8" r="6.4" fill="none" stroke="#fff" stroke-width="1.3"/></svg>',
    mount,
  });

  // Applied at load, before the first paint of the layout, so a timeline that should start
  // as a strip is never briefly full height. The drag listeners need the zone to exist.
  if (window.TimelinePeek) {
    window.TimelinePeek.apply(window.TimelinePeek.get());
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => window.TimelinePeek.install());
    } else {
      window.TimelinePeek.install();
    }
  }

  // The onboarding wizard's theme step is this same picker.
  window.AppearanceSettings = { buildPicker };
})();
