// Keyboard reachability: a trap for overlays, and roving tabindex for groups.

const FOCUSABLE = [
  "a[href]", "button:not([disabled])", "input:not([disabled])",
  "select:not([disabled])", "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

// Hiding is inherited: a control inside a hidden or aria-hidden subtree is
// unreachable however focusable it looks on its own, and trapping focus onto one
// strands the keyboard on something invisible.
const HIDDEN = '[hidden], [aria-hidden="true"]';

export function focusables(container) {
  return [...container.querySelectorAll(FOCUSABLE)].filter((n) => {
    const hidden = n.closest(HIDDEN);
    return !hidden || !container.contains(hidden);
  });
}

/**
 * Keep Tab inside `container` until released, and put focus back where it was.
 *
 * Restoring matters as much as trapping: closing a dialog and dropping focus to
 * <body> silently ends keyboard navigation, and the only way back is the mouse.
 */
export function trap(container, { initial } = {}) {
  const previous = document.activeElement;

  const onKeyDown = (event) => {
    if (event.key !== "Tab") return;
    const items = focusables(container);
    if (!items.length) { event.preventDefault(); return; }
    const first = items[0];
    const last = items[items.length - 1];
    const active = document.activeElement;

    // Wrapping is also the fix for focus having escaped the container entirely
    // (a click outside, a removed node): either edge pulls it back in.
    if (event.shiftKey && (active === first || !container.contains(active))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || !container.contains(active))) {
      event.preventDefault();
      first.focus();
    }
  };

  container.addEventListener("keydown", onKeyDown);
  const target = initial || focusables(container)[0] || container;
  if (target && target.focus) target.focus();

  return function release({ restore = true } = {}) {
    container.removeEventListener("keydown", onKeyDown);
    if (restore && previous && previous.isConnected && previous.focus) previous.focus();
  };
}

/**
 * One tab stop for a group, arrows to move within it -- what a segmented
 * control, toggle group or menu needs so Tab does not walk every option.
 */
export function roving(container, { orientation = "horizontal", onSelect } = {}) {
  const NEXT = orientation === "vertical" ? "ArrowDown" : "ArrowRight";
  const PREV = orientation === "vertical" ? "ArrowUp" : "ArrowLeft";

  const items = () => focusables(container);

  const setIndex = (i) => {
    const list = items();
    if (!list.length) return;
    const clamped = (i + list.length) % list.length;
    list.forEach((n, j) => n.setAttribute("tabindex", j === clamped ? "0" : "-1"));
    list[clamped].focus();
    if (onSelect) onSelect(list[clamped], clamped);
  };

  const onKeyDown = (event) => {
    const list = items();
    const current = list.indexOf(document.activeElement);
    if (current === -1) return;
    if (event.key === NEXT) { event.preventDefault(); setIndex(current + 1); }
    else if (event.key === PREV) { event.preventDefault(); setIndex(current - 1); }
    else if (event.key === "Home") { event.preventDefault(); setIndex(0); }
    else if (event.key === "End") { event.preventDefault(); setIndex(list.length - 1); }
  };

  items().forEach((n, i) => n.setAttribute("tabindex", i === 0 ? "0" : "-1"));
  container.addEventListener("keydown", onKeyDown);
  return () => container.removeEventListener("keydown", onKeyDown);
}
