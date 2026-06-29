// Pure scene→root matching for the global-prompt re-parse, extracted from store.js so it
// can be unit-tested in node (no DOM/state deps). The global prompt is authoritative; when
// it is re-parsed into scene slots, each slot must reuse the old scene root that owns its
// CONTENT — not the root that merely sits at the same array index. Matching by POSITION
// first was the "stale scene" bug: a reorder/insert/delete shifted indices and glued one
// scene's text onto another scene's anchor+render (then that wrong mapping got saved, so it
// survived restarts and key-clears). See scene_align.test.js for the regression cases.
//
// matchParsedToRoots(parsedTexts, rootTexts) → matchByIdx, where matchByIdx[i] is the index
// of the old root that parsed slot i should reuse, or -1 for none (→ ghost/new upstream).
// Both inputs are arrays of ALREADY-normalized text (caller applies its own normalizer so
// this stays the single source of matching truth without duplicating normalization).
(function (root) {
  function matchParsedToRoots(parsedTexts, rootTexts) {
    const n = parsedTexts.length;
    const m = rootTexts.length;
    const matchByIdx = new Array(n).fill(-1);
    const usedRoot = new Array(m).fill(false);

    // Pass 1 — exact text match, nearest original position breaks ties. Pins every
    // unchanged scene to its own root through any reorder / insert / delete.
    for (let i = 0; i < n; i++) {
      const pt = parsedTexts[i];
      if (!pt) continue; // empty slots can only be positional (Pass 2)
      let best = -1;
      let bestDist = Infinity;
      for (let ri = 0; ri < m; ri++) {
        if (usedRoot[ri]) continue;
        if (rootTexts[ri] !== pt) continue;
        const dist = Math.abs(ri - i);
        if (dist < bestDist) { best = ri; bestDist = dist; }
      }
      if (best >= 0) { matchByIdx[i] = best; usedRoot[best] = true; }
    }

    // Pass 2 — leftover slots inherit the leftover roots IN ORDER (in-place text edits keep
    // their render/anchor). A slot with no root left stays -1 → a fresh scene upstream.
    let rr = 0;
    for (let i = 0; i < n; i++) {
      if (matchByIdx[i] >= 0) continue;
      while (rr < m && usedRoot[rr]) rr++;
      if (rr >= m) break;
      matchByIdx[i] = rr;
      usedRoot[rr] = true;
    }

    return matchByIdx;
  }

  const api = { matchParsedToRoots };
  if (typeof module !== "undefined" && module.exports) module.exports = api; // node test
  if (root) root.SceneAlign = api;                                          // browser global
})(typeof window !== "undefined" ? window : null);
