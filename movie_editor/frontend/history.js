// Undo / redo for project + session editor state (sceneRenders, sceneGhosts).
(function () {
  const MAX = 60;
  let undoStack = [];
  let redoStack = [];
  let applying = false;

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
    undoStack.push(snap(store));
    if (undoStack.length > MAX) undoStack.shift();
    redoStack = [];
    store.notifyHistoryState();
  }

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
    record, undo, redo, clear,
    canUndo: () => undoStack.length > 0,
    canRedo: () => redoStack.length > 0,
    isApplying: () => applying,
  };
})();
