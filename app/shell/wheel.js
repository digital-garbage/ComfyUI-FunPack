// The picker wheel: everything the app can do, under the pointer.
//
// v4 had three pinned buttons in the toolbar and a fixed idea of what could go
// in them. This is the other way round -- it shows whatever announced itself,
// so a part of the app that grew a new action is in the wheel without the wheel
// being told.
//
// Middle mouse by default, anywhere. The middle button is otherwise autoscroll,
// which does nothing useful over a panel.

import { composer } from "../composer/composer.js";
import { offered, run } from "./actions.js";

const MIDDLE = 1;

export function createWheel({ button = MIDDLE, root = document } = {}) {
  let open = null;

  const close = () => { if (open) { open.close("dismissed"); open = null; } };

  function show(x, y) {
    const items = offered();
    // The kit refuses a wheel of one -- there is nothing to aim between -- and a
    // page where only one part of the app loaded is exactly when that happens.
    // No wheel is a better answer than a thrown error under the pointer.
    if (items.length < 2 || open) return null;
    open = composer.wheel.picker({
      x, y, items,
      onPick: (item) => { open = null; run(item.id); },
      onClose: () => { open = null; },
    });
    return open;
  }

  const down = (event) => {
    if (event.button !== button) return;
    // Autoscroll otherwise, which is nothing anyone wants over a panel.
    event.preventDefault();
    show(event.clientX, event.clientY);
  };
  // Chrome opens autoscroll on mousedown and only cancels it if the AUXCLICK is
  // also prevented.
  const aux = (event) => { if (event.button === button) event.preventDefault(); };

  root.addEventListener("mousedown", down);
  root.addEventListener("auxclick", aux);

  return {
    show, close,
    get isOpen() { return Boolean(open); },
    destroy() {
      close();
      root.removeEventListener("mousedown", down);
      root.removeEventListener("auxclick", aux);
    },
  };
}
