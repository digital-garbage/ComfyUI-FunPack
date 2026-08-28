// Pointer drags for windows, sliders and splitters.
//
// Pointer capture rather than document-level listeners: capture follows the
// pointer outside the window and guarantees a matching pointerup, so a drag
// cannot be left running because the cursor left the viewport mid-gesture.

/**
 * drag(handle, { onStart, onMove, onEnd, button })
 *
 * onMove receives { dx, dy, x, y } relative to where the drag began.
 * Returns a dispose function.
 */
export function drag(handle, { onStart, onMove, onEnd, button = 0 } = {}) {
  let origin = null;
  let pointerId = null;

  const onPointerDown = (event) => {
    if (event.button !== button || origin) return;
    origin = { x: event.clientX, y: event.clientY };
    pointerId = event.pointerId;
    if (handle.setPointerCapture) handle.setPointerCapture(pointerId);
    handle.addEventListener("pointermove", onPointerMove);
    handle.addEventListener("pointerup", onPointerUp);
    handle.addEventListener("pointercancel", onPointerUp);
    event.preventDefault();
    if (onStart) onStart({ x: origin.x, y: origin.y, event });
  };

  const delta = (event) => ({
    dx: event.clientX - origin.x,
    dy: event.clientY - origin.y,
    x: event.clientX,
    y: event.clientY,
  });

  const onPointerMove = (event) => {
    if (!origin || event.pointerId !== pointerId) return;
    if (onMove) onMove({ ...delta(event), event });
  };

  const onPointerUp = (event) => {
    if (!origin || event.pointerId !== pointerId) return;
    const final = delta(event);
    stop();
    if (onEnd) onEnd({ ...final, event, cancelled: event.type === "pointercancel" });
  };

  function stop() {
    handle.removeEventListener("pointermove", onPointerMove);
    handle.removeEventListener("pointerup", onPointerUp);
    handle.removeEventListener("pointercancel", onPointerUp);
    if (pointerId !== null && handle.releasePointerCapture && handle.hasPointerCapture?.(pointerId)) {
      handle.releasePointerCapture(pointerId);
    }
    origin = null;
    pointerId = null;
  }

  handle.addEventListener("pointerdown", onPointerDown);
  return function dispose() {
    stop();
    handle.removeEventListener("pointerdown", onPointerDown);
  };
}
