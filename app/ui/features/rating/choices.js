// What a person can say about a render, and what each answer teaches. The saved word is the project's `rating` (v4 files
// use the same words); what is sent to the taste store is only liked / disliked, plus which way a dislike went.
export const FORGET = "-Just forget it-";
export const CHOICES = [
  { value: "10", label: "Liked", hint: "This generation was good." },
  { value: "1", label: "Disliked", hint: "This generation was not." },
  { value: "Disliked: bad image", label: "Disliked — bad image", hint: "Composition was fine; the picture itself was ruined." },
  { value: "Disliked: bad composition", label: "Disliked — bad composition", hint: "Well drawn, but the shots, layout or movement were wrong." },
];

/** A saved rating as words ("" when there is none). */
export function nameOf(value) {
  const v = String(value || "").replace(/\|loved$/, "").trim();
  if (!v || v === FORGET) return "";
  const hit = CHOICES.find((c) => c.value === v);
  if (hit) return hit.label;
  return /^\d+$/.test(v) ? `${v}/10` : v;
}

/** What a saved rating teaches the taste store: {rating, axis}; null = it teaches nothing, "clear" = forget. */
export function tasteOf(value) {
  const v = String(value || "").replace(/\|loved$/, "").trim();
  if (!v || v === FORGET) return "clear";
  if (v === "Disliked: bad image") return { rating: "disliked", axis: "image" };
  if (v === "Disliked: bad composition") return { rating: "disliked", axis: "composition" };
  if (/^\d+$/.test(v)) return { rating: Number(v) >= 6 ? "liked" : "disliked", axis: null };
  return null;
}
