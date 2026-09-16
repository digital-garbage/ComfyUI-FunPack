// The wheel's starter set: a modest, obviously-useful list so the feature
// isn't empty the first time anyone opens it. Anything else offers itself
// from wherever it's built -- this file is not the only place an action can
// come from, just where the first ones do.
(function () {
  const A = window.FunPackActions;
  const S = window.Store;
  if (!A || !S) return;

  A.offerAction({ id: "generate-all", label: "Generate", icon: "▶", run: () => S.generate(null) });
  A.offerAction({ id: "generate-selected", label: "Generate selected", icon: "▶", run: () => S.generateSelected() });
  A.offerAction({ id: "render-final", label: "Render", icon: "⧉", run: () => S.renderFinal() });
  A.offerAction({ id: "auto-montage", label: "Auto Montage", icon: "⚡", run: () => window.MontageDialog?.open() });
  A.offerAction({
    id: "undo", label: "Undo", icon: "↶",
    run: () => { window.EditorHistory?.undo(S); },
  });
  A.offerAction({
    id: "redo", label: "Redo", icon: "↷",
    run: () => { window.EditorHistory?.redo(S); },
  });
  A.offerAction({ id: "open-models", label: "Models & Pipeline", icon: "▤", run: () => window.ModelsModal?.open() });
  A.offerAction({ id: "open-engine", label: "Engine settings", icon: "⚙", run: () => window.SettingsWindow?.open("engine") });
  A.offerAction({ id: "sampler-quick", label: "Sampler", icon: "⏱", run: () => window.SamplerQuickbar?.toggle() });
})();
