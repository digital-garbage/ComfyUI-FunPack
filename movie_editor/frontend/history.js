// Undo / redo for project + session editor state (sceneRenders, sceneGhosts).
(function () {
  const MAX = 60;
  let undoStack = [];
  let redoStack = [];
  let applying = false;
  let coalesceDepth = 0;
  let coalesceSnapshot = null;

  function snap(store) {
    const st = store.get();
    return {
      project: JSON.parse(JSON.stringify(st.project || null)),
      sceneRenders: JSON.parse(JSON.stringify(st.sceneRenders || {})),
      sceneGhosts: JSON.parse(JSON.stringify(st.sceneGhosts || [])),
      selectedSceneId: st.selectedSceneId,
      selectedSceneIds: [...(st.selectedSceneIds || [])],
    };
  }

  function restore(store, snap) {
    applying = true;
    try {
      const st = store.get();
      st.project = snap.project;
      st.sceneRenders = snap.sceneRenders;
      st.sceneGhosts = snap.sceneGhosts;
      st.selectedSceneId = snap.selectedSceneId;
      st.selectedSceneIds = [...(snap.selectedSceneIds || [])];
      store.notify();
      store.notifyHistoryState();
    } finally {
      applying = false;
    }
  }

  function record(store) {
    if (applying || !store.get().project) return;
    if (coalesceDepth > 0) return;
    undoStack.push(snap(store));
    if (undoStack.length > MAX) undoStack.shift();
    redoStack = [];
    store.notifyHistoryState();
  }

  function beginCoalesce(store) {
    if (applying || !store.get().project) return;
    if (coalesceDepth === 0) coalesceSnapshot = snap(store);
    coalesceDepth++;
  }

  function endCoalesce(store) {
    if (coalesceDepth <= 0) return;
    coalesceDepth--;
    if (coalesceDepth === 0 && coalesceSnapshot) {
      undoStack.push(coalesceSnapshot);
      if (undoStack.length > MAX) undoStack.shift();
      redoStack = [];
      coalesceSnapshot = null;
      store.notifyHistoryState();
      store.scheduleSaveFromHistory?.();
    }
  }

  function isCoalescing() { return coalesceDepth > 0; }

  function undo(store) {
    if (!undoStack.length) return;
    redoStack.push(snap(store));
    restore(store, undoStack.pop());
  }

  function redo(store) {
    if (!redoStack.length) return;
    undoStack.push(snap(store));
    restore(store, redoStack.pop());
  }

  function clear() {
    undoStack = [];
    redoStack = [];
  }

  window.EditorHistory = {
    record, undo, redo, clear, beginCoalesce, endCoalesce, isCoalescing,
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
    isApplying: () => applying,
  };
})();
