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

  function mount(body) {
    const wrap = el("div", "sw-stack");
    wrap.append(el("div", "sw-rows-label", "Colour scheme"));
    const { grid, paint } = buildPicker();
    wrap.append(grid);
    wrap.append(el("div", "sw-hint",
      "Stored in this browser, not in the project — a rented instance you open the editor on "
      + "starts on Dark until you set it there too."));
    body.append(wrap);
    const off = T.onChange(() => paint(T.get()));
    return () => off();
  }

  window.SettingsWindow.register({
    id: "appearance", group: "", order: 0, title: "Appearance",
    subtitle: "Light, dark, or follow the system.",
    keywords: "theme colour color scheme light dark auto appearance contrast palette",
    iconBg: "linear-gradient(180deg,#8fd0ff,#3a7fd5)",
    icon: '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M8 1.6a6.4 6.4 0 1 0 0 12.8V1.6z" fill="#fff"/><circle cx="8" cy="8" r="6.4" fill="none" stroke="#fff" stroke-width="1.3"/></svg>',
    mount,
  });

  // The onboarding wizard's theme step is this same picker.
  window.AppearanceSettings = { buildPicker };
})();
