// Studio-style rating picker for Movie Editor (mirrors web/funpack_rating_picker.js).
(function () {
  const FORGET_LABEL = "-Just forget it-";

  // `label` is the canonical value sent to the backend (unchanged); `display` is the
  // clearer name shown to the user. Hints note that "action" includes camera MOVES and
  // "details" includes appearance / setting / camera FRAMING (_v2_axis_categories).
  const CATEGORIES = [
    {
      id: "positive",
      label: "Positive",
      accent: "76,175,125",
      ratings: [
        { label: "Perfect", reward: "+1.0", hint: "Exactly what you asked for. Strongly reinforces this result." },
        { label: "Nailed it", reward: "+0.75", hint: "On target, just short of perfect. Reinforces." },
      ],
    },
    {
      id: "missing",
      label: "Missing (asked for it, didn't get it)",
      accent: "212,168,75",
      ratings: [
        { label: "Missing action", reward: "+0.05", hint: "Prompted motion didn't happen — includes CAMERA moves (pan/zoom/dolly). Pushes more motion." },
        { label: "Missing details", reward: "+0.35", hint: "Asked-for details absent — fine detail, appearance, setting, or camera framing." },
        { label: "Missing quality", reward: "−0.30", display: "Bad quality / artifacts", hint: "Body horror, melting, blur, low quality — the IMAGE itself is degraded." },
        { label: "Missing action + quality", reward: "−0.55", hint: "Motion missing AND the image is degraded." },
      ],
    },
    {
      id: "wrong",
      label: "Wrong (did something, but wrong)",
      accent: "210,90,90",
      ratings: [
        { label: "Wrong action", reward: "−0.10", hint: "Subject or CAMERA moved, but the wrong way. Suppresses that motion." },
        { label: "Wrong action + quality", reward: "−0.40", hint: "Wrong motion that also damaged anatomy / quality." },
        { label: "Wrong details", reward: "+0.20", display: "Not-asked details", hint: "Detail / appearance / framing is fine in general — just not what THIS prompt asked." },
        { label: "Wrong appearance", reward: "0.0", display: "Wrong character/subject", hint: "Wrong person, outfit, or place. Stops auto-injecting appearance words. NOT for body horror — use Bad quality." },
      ],
    },
  ];

  const NUCLEAR = [
    { label: "Awful", reward: "−0.90", hint: "Everything wrong. Strong negative." },
    { label: FORGET_LABEL, reward: "none", hint: "Clears the rating — no learning signal." },
  ];

  // canonical label -> display name, for rendering a stored value (scene button, chips).
  const DISPLAY_NAMES = {};
  for (const cat of CATEGORIES) for (const r of cat.ratings) if (r.display) DISPLAY_NAMES[r.label] = r.display;
  const displayName = (label) => DISPLAY_NAMES[label] || label;

  const NO_LOVED = new Set([
    "Perfect", "Nailed it", "Awful", FORGET_LABEL,
    "Missing quality", "Missing details + quality", "Missing action + quality", "Wrong action + quality",
  ]);

  let activePicker = null;
  let activeCleanup = null;

  function rewardColor(reward) {
    if (reward === "none") return "var(--faint)";
    const n = parseFloat(String(reward).replace("−", "-"));
    if (Number.isNaN(n)) return "var(--faint)";
    if (n > 0) return "#6ecf8a";
    if (n < 0) return "#e07070";
    return "var(--faint)";
  }

  function formatLabel(value) {
    const v = String(value || "").trim();
    if (!v || v === FORGET_LABEL) return "";
    if (v.endsWith("|loved")) return displayName(v.slice(0, -6)) + " ♥";
    return displayName(v);
  }

  function buttonLabel(value) {
    const v = String(value || "").trim();
    if (!v || v === FORGET_LABEL) return "Rate scene…";
    return "★ " + formatLabel(v);
  }

  function closePicker() {
    activePicker?.remove();
    activePicker = null;
    activeCleanup?.();
    activeCleanup = null;
  }

  function makeOption(r, accent, onPick, allOptions) {
    const wrap = document.createElement("div");
    wrap.className = "me-rating-opt-wrap";
    wrap.style.setProperty("--accent", accent);

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "me-rating-opt";
    btn.title = r.hint;
    btn._fpBase = r.label;
    btn._fpLoved = false;

    const labelEl = document.createElement("div");
    labelEl.className = "me-rating-opt-label";
    labelEl.textContent = r.display || r.label;

    const rewardEl = document.createElement("div");
    rewardEl.className = "me-rating-opt-reward";
    rewardEl.style.color = rewardColor(r.reward);
    rewardEl.textContent = r.reward;

    const hintEl = document.createElement("div");
    hintEl.className = "me-rating-opt-hint";
    hintEl.textContent = r.hint;

    btn.append(labelEl, rewardEl, hintEl);
    btn.addEventListener("click", () => onPick(r.label));
    wrap.append(btn);
    allOptions.push(btn);

    if (!NO_LOVED.has(r.label)) {
      const heart = document.createElement("button");
      heart.type = "button";
      heart.className = "me-rating-heart";
      heart.textContent = "♥";
      heart.title = `${r.label} — loved it`;
      heart._fpBase = r.label;
      heart._fpLoved = true;
      heart.addEventListener("click", () => onPick(r.label + "|loved"));
      wrap.append(heart);
      allOptions.push(heart);
    }
    return wrap;
  }

  function open(event, currentValue, onPick) {
    closePicker();

    const picker = document.createElement("div");
    picker.className = "me-rating-picker";
    activePicker = picker;

    const header = document.createElement("div");
    header.className = "me-rating-picker-head";
    header.append(Object.assign(document.createElement("span"), { textContent: "Rate this generation" }));
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "me-rating-picker-close";
    closeBtn.textContent = "×";
    closeBtn.addEventListener("click", closePicker);
    header.append(closeBtn);
    picker.append(header);

    const raw = String(currentValue || "");
    const currentLoved = raw.endsWith("|loved");
    const currentBase = currentLoved ? raw.slice(0, -6) : raw;
    const allOptions = [];

    const setActive = (base, loved) => {
      for (const el of allOptions) {
        el.classList.toggle("active", el._fpBase === base && el._fpLoved === loved);
      }
    };

    const handlePick = (value) => {
      const isLoved = value.endsWith("|loved");
      const base = isLoved ? value.slice(0, -6) : value;
      setActive(base, isLoved);
      onPick(value);
      closePicker();
    };

    for (const cat of CATEGORIES) {
      const section = document.createElement("div");
      section.className = "me-rating-section";
      section.style.setProperty("--accent", cat.accent);
      const sectionLabel = document.createElement("div");
      sectionLabel.className = "me-rating-section-label";
      sectionLabel.textContent = cat.label;
      section.append(sectionLabel);
      const grid = document.createElement("div");
      grid.className = cat.id === "positive" ? "me-rating-grid me-rating-grid-pos" : "me-rating-grid";
      for (const r of cat.ratings) grid.append(makeOption(r, cat.accent, handlePick, allOptions));
      section.append(grid);
      picker.append(section);
    }

    const nuclear = document.createElement("div");
    nuclear.className = "me-rating-nuclear";
    nuclear.style.setProperty("--accent", "140,100,100");
    for (const r of NUCLEAR) nuclear.append(makeOption(r, "140,100,100", handlePick, allOptions));
    picker.append(nuclear);

    setActive(currentBase, currentLoved);
    document.body.append(picker);

    const cx = event?.clientX ?? window.innerWidth / 2;
    const cy = event?.clientY ?? window.innerHeight / 2;
    requestAnimationFrame(() => {
      const ph = picker.offsetHeight;
      const pw = picker.offsetWidth;
      picker.style.left = Math.max(8, Math.min(cx, window.innerWidth - pw - 8)) + "px";
      picker.style.top = Math.max(8, Math.min(cy, window.innerHeight - ph - 8)) + "px";
    });

    const onOutside = (e) => { if (!picker.contains(e.target)) closePicker(); };
    activeCleanup = () => document.removeEventListener("mousedown", onOutside, true);
    setTimeout(() => document.addEventListener("mousedown", onOutside, true), 0);
  }

  window.MovieRatingPicker = { open, close: closePicker, formatLabel, buttonLabel, FORGET_LABEL };
})();
