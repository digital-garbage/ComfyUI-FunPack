import { composer as c } from "../../composer/composer.js";

const requestBody = (app, text) => {
  const selected = app.project.selected;
  const prefix = (app.promptPrefix || []).flatMap((fn) => { try { return fn(selected); } catch { return []; } });
  return {
  text,
  anchor: [...prefix, app.project.anchor].filter((x) => x && String(x).trim()).join(" "),
  postfix: app.project.postfix,
  postfix_enabled: app.project.postfixEnabled,
  variables: app.project.variables,
  seed: 1,
  };
};

export function promptBuilder(app, own) {
  const page = c.region.stack({ gap: "sm" });
  const project = app.project;
  let enabled = false, analysis = null, base = "", draft = "", learned = false, ratingStats = {}, reviewNeeded = false;
  let busy = false, alive = true, builtBody = null;
  let targetProject = project.project && project.project.id, targetScene = project.selectedId;
  own(() => { alive = false; });

  const source = c.textarea.md({ label: "Prompt idea", rows: 4,
    value: project.selected ? project.selected.text || "" : "",
    placeholder: "Describe the scene or start with a shortcut trigger.",
    onInput: () => { base = ""; draft = ""; learned = false; } });
  const result = c.textarea.md({ label: "Review and edit the draft", rows: 7, value: "",
    placeholder: "Build a draft to start. You can edit it here before using or teaching it.",
    onInput: (value) => { draft = value; } });
  const note = c.hint.default({ text: "" });

  function draw() {
    if (!alive) return;
    const info = analysis || {};
    const categories = (info.categories || []).slice(0, 5).map((x) => `${x.name} (${x.count})`).join(" · ") || "No shortcut categories yet";
    const terms = (info.words || []).slice(0, 8).map((x) => `${x.text} ×${x.count}`).join(" · ") || "No repeated terms yet";
    const positions = (info.positions || []).slice(0, 5).map((x) => `${x.text}: ${Object.entries(x).filter(([k]) => k !== "text").map(([k, v]) => `${k} ${v}`).join(", ")}`).join(" · ") || "No position patterns yet";
    const neighbors = (info.neighbors || []).slice(0, 5).map((x) => `${x.left} → ${x.right} (${x.count})`).join(" · ") || "No repeated neighbors yet";
    const phrases = (info.phrases || []).slice(0, 4).map((x) => `“${x.text}”`).join(" · ") || "No replacement phrases yet";
    const targetIsCurrent = () => project.project && project.project.id === targetProject && project.selectedId === targetScene;
    const loadScene = c.button.sm({ label: "Load selected scene", tone: "ghost", onClick: () => {
      targetProject = project.project && project.project.id; targetScene = project.selectedId;
      source.setValue(project.selected ? project.selected.text || "" : "");
      base = ""; draft = ""; learned = false; builtBody = null;
      result.setValue(""); note.setText("Loaded the currently selected scene.");
    } });
    const build = c.button.sm({ label: busy ? "Building…" : "Build draft", tone: "primary", disabled: busy,
      onClick: async () => {
        busy = true; draw();
        try {
          builtBody = requestBody(app, source.node.value);
          targetProject = project.project && project.project.id; targetScene = project.selectedId;
          const built = await app.api.promptBuilderDraft(builtBody);
          base = built.base; draft = built.draft; learned = built.learned;
          reviewNeeded = built.prompt_review_needed === true;
          result.setValue(draft);
          note.setText(reviewNeeded
            ? `This reviewed version has ${built.prompt_votes.bad} dislikes and ${built.prompt_votes.good} likes. FunPack paused its automatic reuse; review and save a new edit if you want to teach a replacement.`
            : learned ? "Loaded your reviewed version for this exact expanded prompt." : "Started with FunPack’s normal shortcut expansion. Edit the draft, then choose Learn from this edit.");
        } catch (err) { note.setText(`Could not build a draft: ${err.message}`); }
        finally { busy = false; draw(); }
      } });
    const use = c.button.sm({ label: "Use in selected scene", tone: "neutral", disabled: !base || !draft,
      onClick: () => {
        if (!targetIsCurrent()) return note.setText("The selected project or scene changed. Load the selected scene before applying this draft.");
        const selected = project.selected;
        if (!selected) return note.setText("Select a scene first.");
        project.setText(selected.id, draft);
        source.setValue(draft); base = ""; learned = false; builtBody = null;
        note.setText("Draft copied into the selected scene. Review it there before generating.");
      } });
    const learn = c.button.sm({ label: "Learn from this edit", tone: "neutral", disabled: !base || !draft || busy,
      onClick: async () => {
        try {
          if (!builtBody) return note.setText("Build a draft before teaching an edit.");
          const saved = await app.api.promptBuilderLearn({ ...builtBody, accepted: result.node.value });
          note.setText(`Saved ${saved.edit_count} edit${saved.edit_count === 1 ? "" : "s"}. FunPack will reuse this review for the same expanded prompt when automatic use is enabled.`);
        } catch (err) { note.setText(`Could not save this review: ${err.message}`); }
      } });
    const toggle = c.toggle.default({ label: "Use learned drafts automatically for exact matches", checked: enabled,
      hint: "This changes the prompt sent to generation only when it exactly matches a prompt you reviewed.",
      onChange: async (value) => {
        try { const state = await app.api.promptBuilderEnabled(value); enabled = state.enabled; draw(); }
        catch (err) { note.setText(`Could not update this setting: ${err.message}`); }
      } });
    const clear = c.button.sm({ label: "Forget learned drafts…", tone: "ghost", disabled: !info.examples,
      onClick: async () => {
        if (!(await c.modal.dialogue({ title: "Forget learned drafts", message: "Delete the reviewed prompt pairs and turn automatic use off?", tone: "danger", confirmLabel: "Forget" }).result)) return;
        try { const state = await app.api.promptBuilderClear(); enabled = state.enabled; ratingStats = state; analysis = { ...info, examples: 0 }; note.setText("Learned prompt pairs deleted."); draw(); }
        catch (err) { note.setText(`Could not clear learned prompts: ${err.message}`); }
      } });
    page.set([
      c.label.section({ text: "Prompt builder" }),
      c.hint.default({ text: "Builds from your scene text and saved shortcuts. There is no language-model rewrite; nothing is added unless it came from a shortcut or your own edit." }),
      toggle,
      c.field.default({ label: "Shortcut profile", hint: `${info.shortcuts || 0} enabled shortcuts · ${info.spacy ? "spaCy phrase patterns available" : "word and category counts (spaCy model unavailable)"}`, control: c.text.sm({ text: `${categories}\nTerms: ${terms}\nPositions: ${positions}\nNeighbors: ${neighbors}\nCommon phrases: ${phrases}` }) }),
      c.toolbar.default({ items: [c.label.section({ text: "Prompt idea" })], trailing: [loadScene] }),
      source,
      c.toolbar.default({ items: [build], trailing: [clear] }),
      result,
      c.toolbar.default({ items: [learn, use] }),
      note,
      c.hint.default({ text: `${info.examples || 0} reviewed prompt${info.examples === 1 ? "" : "s"} and ${ratingStats.captured_prompts || 0} recent generation prompt${ratingStats.captured_prompts === 1 ? "" : "s"} stored locally. Linked feedback: ${ratingStats.prompt_likes || 0} likes, ${ratingStats.prompt_dislikes || 0} dislikes; ${ratingStats.prompt_review_needed || 0} prompt${ratingStats.prompt_review_needed === 1 ? "" : "s"} paused after repeated negative feedback. Learned rewrites are exact-match only.` }),
    ]);
  }

  Promise.all([app.api.promptBuilderStatus(), app.api.promptBuilderAnalysis()]).then(([state, stats]) => {
    enabled = state.enabled; ratingStats = state; analysis = stats; draw();
  }).catch((err) => { analysis = {}; note.setText(`Could not load the local prompt profile: ${err.message}`); draw(); });
  return page;
}
