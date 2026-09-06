// The way in, on a first visit.
//
// v4's could not be dismissed: a page refresh was the only way out, which makes
// it a trap rather than a wizard. This one closes on Escape, on the backdrop, on
// its own close button and on "Start empty" -- and it only ever appears when
// there is nothing to come back to, so the way to never see it again is to have
// a project, which is what it makes.

import { composer } from "../composer/composer.js";

const CHOICES = [
  { id: "empty", icon: "▭", label: "Empty project",
    hint: "One scene, nothing in it." },
  { id: "scenes", icon: "▦", label: "A few scenes",
    hint: "Three to write into, in order." },
];

/**
 * open({ onPick }) -> the modal handle.
 *
 * `onPick` is given the choice id, or null when the wizard was dismissed --
 * dismissing is a real answer, and the app carries on with what it has.
 */
export function open({ onPick } = {}) {
  let answered = false;
  const say = (id) => {
    if (answered) return;
    answered = true;
    if (onPick) onPick(id);
  };

  const window_ = composer.modal.generic({
    title: "Welcome to FunPack",
    subtitle: "Multi-scene video on a real timeline.",
    size: "md",
    body: composer.gallery.cards({
      items: CHOICES,
      onActivate: (item) => { say(item.id); window_.close("picked"); },
    }),
    actions: [composer.button.md({ label: "Start empty", tone: "ghost",
                                   onClick: () => window_.close("dismissed") })],
    onClose: () => say(null),
  });
  return window_;
}
