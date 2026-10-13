// Alphabetical, ignoring case and accents; two names that differ only in case fall back to the exact text so the order never shuffles.
export const alpha = (a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: "base" }) || String(a).localeCompare(String(b));
