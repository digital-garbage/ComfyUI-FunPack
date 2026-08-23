# Changelog

## [Unreleased]

### Fixed
- **Changing an attention or SLA setting no longer re-reads the checkpoint.** They live on
  the loader node, so ComfyUI invalidated the whole node and the file was loaded again — a
  minute of dequantizing a GGUF to alter a sparsity ratio. The weights depend only on the
  file (identity, size, mtime) and the dtypes; everything else is applied to a clone. One
  entry, replaced whenever the key changes.
- **A GGUF architecture ComfyUI-GGUF has not been told about now loads quantized anyway.**
  The pack refuses anything off its tested list — `Unexpected architecture type in GGUF file:
  'minimax_h3'` — which cost the quantized path entirely and expanded the whole checkpoint
  instead. The name is added to its IMAGE list (never the text one) for a file whose header
  declares it. Since that steps around the pack's own guard, what the backend hands back is
  checked here instead: anything that is not a state dict of named tensors raises, and the
  existing fallback turns that into the slow-but-working path. The status line says the
  override is in effect, because a wrong generation should be traceable to it.
- **When it does have to expand a GGUF, it expands ONCE.** The dequantized weights are
  written beside the file as an ordinary safetensors and reused on every later load, which
  then runs at safetensors speed. It costs the model's dequantized size in disk — the same
  size that path was already going to occupy in VRAM — and the status line names the file to
  delete. A cache older than the model is ignored; one that cannot be written is a slow
  load, not a failed one.
- **GGUF dequantizes across threads instead of one tensor at a time.** The native path
  expands every quantized tensor at load with numpy, serially, which is where the wait went.
  numpy releases the GIL for that work, so it now runs on up to 8 threads — bounded rather
  than one per core, because each worker holds a dequantized tensor and its copy.
- **The updater installs only what is MISSING, and never upgrades anything.** It ran
  `pip install -r requirements.txt`, which upgrades any package below its version floor —
  and on an install full of compiled extensions (torch, comfy-kitchen, comfy-aimdo,
  onnxruntime, opencv) a numpy or transformers bump underneath them does not raise, it
  segfaults hours later with nothing connecting it to the update. Absent packages are now
  installed by name; a package that is present but older than FunPack asks for is reported
  with the command to upgrade it yourself, and left alone. Nothing missing means pip is
  never run at all.
- **The stand-in tokenizer no longer downloads Gemma for a Qwen model, or reaches the
  network mid-run.** With no CLIP wired, Studio fetched a 12B *Gemma* tokenizer whatever the
  model was — wrong vocabulary for MiniMax H3, which encodes with Qwen3-VL. There is now an
  H3 entry, and every source is tried from the local HuggingFace cache FIRST. Only if nothing
  is cached does it go online, and it says so: `from_pretrained` has no token, no timeout and
  no progress, and it runs on ComfyUI's execution thread, so an uncached fetch is an
  unbounded stall in the middle of a generation with nothing in the log to explain it.
- **A LoRA weight that cannot fit is now dropped, not attempted.** FunPack already detected
  and reported shape mismatches, then handed them to ComfyUI anyway — which builds the full
  `lora_A @ lora_B` delta and only then discovers it will not reshape into the weight. On a
  curve-form MiniMax H3 checkpoint that is a 96768x2688 tensor per block, 51 times (~25 GB
  in bf16, ~49 GB in fp32), allocated and thrown away while dynamic VRAM staging streams the
  model in. Nothing about the render changes — those adapters could never have applied.
- **negative_erase could put an amplified-noise vector into the conditioning.** "Keep prompt
  strength" restored each token's norm after the erase — but a token pointing almost exactly
  along the negative is left with nothing but rounding error, and restoring its norm scaled
  that residue by thousands. The gain is now capped: past it the token stays quiet, because
  it really was mostly the thing you asked to remove. A non-finite result is refused outright
  rather than passed on, since the sampler captures conditioning for the refinement key and
  would have banked the bad vector.
- **Bypass now works as an A/B switch.** Wire two alternatives at one input — MiniMax H3's
  ref-to-video and first-last-to-video both feeding the sampler's latent — and bypass the
  one you are not using. A bypassed slot no longer claims the input, no longer counts as a
  second source in guided mode, and no longer has to pass a value through for a consumer
  something else already feeds or that does not require one. Its own unwired inputs are also
  no longer demanded: it is auto-wired if that is unambiguous, and otherwise left alone.
  Bypassing the ONLY source of a required input still blocks, with the same message.
- **Blocking messages are no longer cropped.** The generation readout kept a long error on
  one unwrapped line, so most of it ran off the canvas. The error state now wraps, scrolls
  past 40vh, can be selected, and carries a copy button. The running readout is unchanged.

### Added
- **`FUNPACK_FAULTHANDLER=1` makes a silent death talk.** Installs Python's faulthandler, so
  a crash in native code prints every thread's stack instead of nothing — and registers
  SIGUSR1, so `kill -USR1 <pid>` dumps where a HUNG ComfyUI actually is, from another
  terminal, without py-spy, gdb or a restart. Off by default; costs nothing while idle.
- **`FUNPACK_PINNED_MEMORY` caps ComfyUI's pinned host-memory budget.** ComfyUI page-locks up
  to 90% of system RAM for weight streaming; pinned pages cannot be swapped or reclaimed, so
  once that budget is committed a further allocation wedges the host instead of killing the
  process — no traceback, nothing in the log. `--disable-pinned-memory` is the supported
  switch, but a rented image often bakes its launch command in where it cannot be edited.
  FunPack is imported after the budget is computed and before any model is staged, so the
  same control works from here: `off` to disable, `64` for 64 GB, `50%` for half the RAM.
  It only ever lowers the number, and says what it did.
- **The negative prompt does something at CFG 1 (experimental).** MiniMax H3 always runs at
  CFG 1.0, so the negative branch is never evaluated and the negative prompt is dead weight.
  Studio ▸ "Use the negative prompt" encodes it anyway, pools it to one direction, and takes
  that direction out of the positive conditioning. Default mode is `project` — it removes
  only the part of each word that points at the negative and leaves the rest alone — with
  `subtract` kept for comparison. The vision span of an H3 conditioning is never touched.
  Off by default and unproven: expect concrete things to behave better than quality words.
- **Studio ▸ "Skip Studio's positive processing".** With both CLIP and a positive CONDITIONING
  wired, CLIP has always won and the wired conditioning was ignored. This inverts that: the
  wired conditioning owns the positive while CLIP keeps encoding the negative and the
  references. It skips shortcuts, $variables and the scene split, which is the point.
- **A different sampler for the second pass.** The Scene Chain Sampler takes an optional
  `second_pass_sampler`; unwired, pass 2 reuses pass 1's, which is the old behaviour. In the
  Editor, Second pass ▸ Own sampler reveals the full algorithm panel for pass 2 — what
  builds a shot well is often not what finishes it. Off, Studio mirrors the high pass's
  algorithm into that output, so a project that already used a second pass is unchanged.
  The console names the pass-2 sampler when it differs.
- **Numbered reference slots.** A node input can be wired to "Reference image 1" instead of
  to a particular file — it resolves to whatever is marked first among the image references,
  so re-ordering marks in the Media Bin re-points every socket without opening a node page.
  Numbered per kind, so marking an audio file never shifts the image slots. The picker
  offers one past the highest number that node already uses, and an unmarked slot leaves its
  socket unconnected in silence — never auto-wired to some other image, never reported.
- **Export settings… now covers the sampling too** — render geometry, the sampler and its
  schedule, the second pass when it is running, and the Studio / Chain Sampler overrides.
  Only the selected algorithm's settings are printed: a pass config always carries the
  hybrid / distilled / normalizing blocks and just one of them is live.
- **Models & Pipeline ▸ Export settings…** renders the loaded pipeline as a PNG: every
  loader with its full filename, every LoRA with its weight, the typed-in values of any
  custom node, and the host's PyTorch / CUDA / attention / GPU. Only inputs that were typed
  are printed — a wired input is marked `‹wired›` rather than showing the stale widget
  behind the socket. The card is drawn in the theme the app is showing, watermarked with the
  FunPack version, and carries the same data as JSON in a tEXt chunk. The dialog previews it
  with Download and Copy image.
- **Quality sharpness now works with a stock KSampler.** The unsharp mask that recovers fine
  detail lived inside Hybrid Euler 2S and Distilled Flow only. It reads the current x0
  prediction, the previous one and the step's sigma — nothing that needs a sampler's loop —
  so it now runs through a denoiser proxy on any `sampler_name`. `sharpen last %` sets the
  window as a fraction of the schedule, the same meaning as Hybrid's `high quality pct`.
  Audio on a packed AV latent is excluded, as in-loop. Off by default; a sampler that cannot
  be wrapped keeps sampling and says so.
- **Settings ▸ Custom Nodes: install, update and remove ComfyUI node packs.** `＋ Add node`
  asks for a repository URL and clones it into `custom_nodes`, installing its
  `requirements.txt` if it has one; each row offers Update (a fast-forward pull, refused if
  the pack has local changes) and Remove. It is three git operations, not a catalogue — you
  supply the URL, and nothing about the repository is vetted first, which the dialog says.
  Removal names the full path it is about to delete and cannot reach outside `custom_nodes`:
  the name must be a single path segment that resolves to a direct child directory, so a
  symlink cannot redirect it, and FunPack cannot delete itself. Node packs register at
  import, so the panel says a restart is needed rather than restarting under you.
  **Check for updates** fetches each pack's origin and shows how far behind it is, on the
  row and on its Update button. It is a button rather than part of the listing because it
  costs a network round trip per pack; the fetches run four at a time, and a pack that
  cannot be compared says why (detached HEAD, no remote, origin unreachable) instead of
  quietly reading as up to date.

- **FunPack's loaders take `.gguf` files**, for diffusion models and text encoders alike, and
  a text-encoder list may mix a `.gguf` with `.safetensors` slots (the usual LTX-2.3 shape).
  Core's extension set has no `.gguf`, so those files were on disk and invisible to every
  picker. `gguf` is now one of FunPack's requirements, so a fresh install reads GGUF
  containers with nothing extra to set up. The runtime is whichever backend is present:
  ComfyUI-GGUF keeps the weights quantized in VRAM, which is the point of GGUF; failing that,
  the `gguf` package dequantizes at load — the file loads, but at full size, and the status
  output says which of the two happened. With neither, the loader names both remedies instead
  of failing obscurely.
  Model-family detection reads GGUF containers too, using the same architecture signatures.
  A GGUF renamed to `.safetensors` is recognised by its container magic and loaded correctly
  rather than failing in the safetensors parser with a UTF-8 decode error. When ComfyUI-GGUF
  refuses an architecture it has no handling for (MiniMax H3, today), the load falls back to
  dequantizing rather than stopping, and says both what refused and what happened instead.

- **Simple mode's Advanced settings button opens the panel it names.** It slides in the
  Editor's own Properties column — which does nothing at all when that column was collapsed
  in Editor mode, since Simple mode hides the dock tabs that would uncollapse it. Opening the
  panel now uncollapses it, and the saved dock state is handed back on the way out of Simple
  mode. The panel also shows its folded sections again: they are hidden to keep the pinned
  column tidy, which left a button called "Advanced settings" opening onto the few rows that
  are not in a fold.

- **Pin up to three shortcuts to the timeline toolbar.** Settings is a window with a sidebar,
  and its useful places — a specific node's page, one Engine category — are several clicks
  deep: fine to walk once, tedious while dialling something in. `📌 Pin to a button` in the
  Settings header puts whatever is open into one of three slots, chosen from a dialog that
  shows what each slot currently holds and asks before replacing one. Pinning while a node
  is open pins the NODE, and pinning inside Engine pins the CATEGORY — "Sampler algorithm",
  not "Engine" — since the section itself was never the slow part. Slot 1 is always the leftmost button
  and slot 3 the one nearest Assets; an empty slot closes up and the remaining buttons keep
  their own numbers, so a shortcut's position never moves. Hovering says where it leads.
  Pins ride with the project, like the other editor preferences. A node opened this way is a
  destination, not a stop inside Settings: Save and Cancel both close the window and return
  you to the editor, rather than leaving you in the Models list you never asked for.

- **Reverse a clip**, audio included, in `+ Add → Effects`. Unlike the other clip effects
  there is no preview-side equivalent, so the monitor switches that clip to a server-rendered
  segment — what you watch is the reversed encode, not a forward one standing in for it.
  ffmpeg holds every frame of a reversal in memory, so clips past a frame limit are refused
  with the count, the limit, and the fix, rather than being rendered forwards silently.

- **Clip geometry on the timeline: flip, crop, and fill-frame**, in `+ Add → Effects`
  alongside the existing zoom/blur/fades. Flip horizontal and Flip vertical mirror the clip;
  Crop edges trims a percentage off each side and rescales (a punch-in that composes with
  Ken Burns rather than replacing it); Fill frame covers the output frame and crops the
  overflow instead of letterboxing. The preview and the render share one filter definition,
  so what plays is what renders. Flips and fill are switches — applying one again turns it
  off — and the timeline clip now shows a chip listing the effects on it, which clears them
  all when clicked. `Remove all effects` is also a preset, since Ken Burns previously had no
  way off. Presets are back-filled into libraries created before they existed.

- **SLA block-sparse attention for MiniMax H3**, as a value in the FunPack Diffusion Model
  Loader's `attention` list rather than a node to wire. ComfyUI ships no sparse-attention
  backend for H3, which is why lightx2v's SLA turbo LoRA gives no speedup on its own — the
  LoRA is the adaptation to sparsity, not the acceleration. Roughly 3.7x the attention
  throughput at 768p/15s. It is a toggle beside the attention backend, not one of its
  values: SLA takes H3's long packed self-attention and the chosen backend (sage3, int8,
  flash) takes the text refiner, masked calls and any trailing dense steps, so the two
  compose. Five settings folded under Advanced and validated at their defaults; skipped
  with a stated reason on anything that is not H3 or without Triton, leaving the chosen
  backend installed on its own. Kernel and block map vendored from LightX2V (Apache-2.0) via
  ComfyUI-H3-SLA-Attention (MIT).

- **MiniMax H3's latent upscaler loads without its custom node pack**, so `second_pass_op`
  and segmented detailing work on H3 — the operation needs an upsampler whose latents are
  the model's width, and the only published 24-channel one shipped behind its own node.
  Architecture and normalisation statistics are read off the checkpoint; verified
  bit-identical against the reference implementation.
- **H3 scenes continue from the previous one.** The previous scene's last latent frame
  becomes the next scene's frame-0 keyframe pin — the only continuity conditioning H3
  accepts. A carried latent tail is not conditioning on this model, so the seam matched
  while the rest of the shot knew nothing about the scene before it.

- **A resample factor for the between-pass operation** (`second_pass_upscale`, 1.0-4.0).
  Only upsamplers that take a factor honour it: MiniMax H3's resizer does, Lightricks' LTX
  one is a fixed 2x network and reports that it ignored the value rather than rounding
  quietly. Latent width and height snap to even, since a patchified model cannot take odd.

### Changed
- **FunPack's own nodes now say what they are.** A card in the Models shelf reads
  `🧠 Diffusion model` / `🔤 CLIP model` / `🎞️ VAE` / `🧩 LoRA` / `🔑 Refinement key`, tinted
  to read as built-in, instead of showing the class name. Third-party nodes keep their class
  name — the difference is the signal: these are the pieces that come wired.

- **The model family is detected from the checkpoint instead of chosen.** Selecting LTX while
  loading an H3 file wired the entire graph for the wrong model, and the mismatch surfaced as a
  stray port rather than as a family error. The family now comes from the diffusion model's own
  safetensors header — key-name signatures only, the same ones ComfyUI's `model_detection` uses,
  so it costs the same on a 40 GB file as on a small one and never loads weights. Changing the
  checkpoint rewires the pipeline and migrates the project's frame geometry, which previously
  only happened if you went through the wizard.

  A file that cannot be identified proposes **no** family: the previous one stands and the panel
  says why. Nothing silently becomes LTX. A video-only Lightricks checkpoint is wired on the AV
  graph, as before, but is now named as video-only so an empty audio branch is expected rather
  than mysterious.

- **Settings the loaded model cannot use are no longer shown.** MiniMax H3 drops Bounded
  Attention, Best-FaceID (with its four sub-settings) and `v2a_grad_scale`; LTX drops
  `h3_audio_clock`. Previously each was offered, left switchable, and then reported once per
  generation as inert — a control that is not offered needs no explanation. Stored values are
  untouched, so switching family back restores the setting. The main-window chips for these are
  gone with them: "turn it off in Settings → Engine" named a row that no longer renders.

- **Operations that fail now say so, and standing conditions stop repeating.** Two rules that
  fight each other unless they share one mechanism, so they now do (`funpack_log.py`). A failure
  reports what was attempted, why it stopped, and **what the output looks like as a result** —
  collapsed to one line per run, because these fire inside per-step wrappers. A standing
  condition ("this model is not LTX", "H3 has no cross-attention to hook") is stated once and
  restated only when it stops being true, instead of on every generation.

  Newly audible: anchor pin restore, guide and mid-scene-guide append, velocity bias, quality
  sharpness, the audio-protection mask, the audio clock correction, momentum guidance, template
  resolution match, learned-direction steering, conditioning memory and repel, Absolute steer,
  and the temporal-style classifier. Each of those could previously fail and leave a run that
  looked exactly like one where the feature was switched off.

### Fixed
- **Updating FunPack in the app installs the dependencies the update needs.** It pulled new
  code and never new requirements, so a release that added one left the node pack unable to
  import with nothing said. When an update touches `requirements.txt`, pip runs against the
  interpreter ComfyUI is using — never a bare `pip`, which in a venv installs somewhere else
  entirely — before the restart, and a failure names the command to run by hand rather than
  turning a completed update into an error. Update and branch switch also moved off the event
  loop; a fetch over a tunnel plus an install was freezing every stream in the meantime.

- **A LoRA that matches by name but not by shape now says so.** Key matching proves a LoRA is
  *for* a model, not that its weights fit it; a mismatched pair is dropped during the merge,
  which was visible only as one generic warning per key while FunPack's own status line still
  read like a clean load. The count now reaches that status line. The common case is named
  outright: MiniMax H3's pruned "curve-form" checkpoints read adaLN from a compact
  time-curve basis, so a turbo LoRA trained against the full-width model matches all 51 adaLN
  keys and merges into none of them. They cannot be projected — the basis that built the
  table is not in the checkpoint — so the message says that and points at a converted LoRA.

- **Context windowing and ALG's guide blur are switched off and hidden on MiniMax H3.**
  Core's windowing unpacks the LTXAV stream specifically and measures its window on LTX's 8x
  latent ratio; ALG's guide blur acts on guide frames appended to the latent, and an H3 guide
  is a condition row, so that tail is always empty. Context windowing also reported the wrong
  reason on H3 — it blamed the ComfyUI version for what is a model difference.

- **JoyAI-Echo is switched off and hidden on MiniMax H3, where it was doing damage.** The
  memory bank places frame *i* at sequence position *i*, but H3's packed layout pins only the
  first or last frame — so every frame past the first was refused, and the one that landed
  *replaced the scene's i2v anchor*, because H3 keys pins by frame index. Not an inert toggle:
  a harmful one. The sampler now forces it off on H3 and says why, and Engine Settings stops
  offering the whole group.
- **JoyAI-Echo's tooltips now say it requires the JoyAI-Echo LoRA.** Without it the injected
  memory frames change nothing on any model, because the base weights were never trained to
  read them as memory — the controls read as a working feature.

- **Closing Models & Pipeline mid-edit no longer leaves its edit buffer behind.** Escape, the
  ✕, or switching to another section tore the section down without ending the edit, so the
  next visit inherited a dirty flag and a baseline belonging to a config that had already
  been replaced — a spurious "unsaved changes" prompt, and a Cancel that could splice stale
  slots into freshly loaded ones.

- **Prompt templates can be renamed, deleted, and turned off.** The Composer's Templates bar
  applied a template and immediately forgot it, so there was no state to leave and no way to
  leave it — and clearing the prompt box by hand did nothing, because an empty global prompt
  is refused (an empty parse would wipe the timeline on a stray keystroke). The bar now shows
  which template is applied, offers `— None —` to clear the whole global prompt — anchor,
  transitions and scene texts — as a deliberate, undoable action, and gives the applied
  template Rename and Delete. Saving pre-fills the applied
  name, so updating a template is the default gesture rather than a trick. Deleting one
  leaves the prompt alone.

- **"No, I'll use my own pipeline" left the setup modal on screen.** Picking a model family
  starts a chain of requests that reopens the modal with fresh prerequisites; dismissing it
  mid-flight closed it, then the reply landed and put it straight back — and since opting out
  installs nothing, the prerequisites were still missing, so it always reopened. Closes the
  user asks for now supersede anything in flight. The choice is also recorded before the
  save round-trip rather than after it, and a save that fails says so instead of leaving the
  project quietly still on the built-in pipeline.

- **An output can be wired to several destinations while adding a node**, not only after.
  The Add-node panel offered one destination per output and told you to finish the job on
  the node's page; it now uses the same multi-destination editor the node page does, with
  the same rule about what the built-in pipeline will honour.

- **Double-click an image in the Media bin to make it the selected clip's anchor.** Dragging
  was the only way in, and it could not be relied on: an `<img>` is natively draggable, so
  inside the card the browser started its own image drag whose payload every drop target
  rejects — worse the larger the image. The image now opts out of dragging, and the drag
  ghost is the thumbnail rather than a snapshot of the full-size card.

- **Preview playback could stall at "Video is loading…" and stay there**, with the transport
  buttons dead. A media reset leaves a `<video>` with no metadata and no load running, and
  every recovery path keyed on the element having errored — which a reset does not do. Play
  went down the same wait-for-metadata branch, so pressing it changed nothing. Elements are
  now checked after a reset and restarted if no load followed; Play restarts a stalled one
  outright.

- **A linked input driven by `Project · Prompt (global)` claimed to encode "the same text
  Studio would" and did not.** Studio appends the postfix to every scene itself, so the
  global prompt is strictly less than what it encodes — and the postfix is usually where
  audio and style directions live, which went missing with no clue. The hint is now accurate
  per source, and points at `Project · Prompt + postfix` when a postfix is set.

- **Studio said nothing when both CLIP and a positive CONDITIONING were wired to it.** CLIP
  wins and the wired conditioning is never read — but the ownership label that knew this was
  computed and discarded, so a graph feeding Studio from an i2v node looked like it was
  working while the prompt path quietly supplied everything. Now stated once per run and
  carried in the encode status.

- **The late-step gate every rating-driven mechanism shares was measuring the wrong thing
  on H3**, so embed guidance, score slider, DynaShift and output guidance were inert or
  nearly inert while reporting themselves active. `max(0, 1 - 2*sigma)` reads sigma as
  schedule progress, which holds on LTX and fails on H3: its schedules are
  `shift*t / (1 + (shift-1)*t)`, so a large shift keeps sigma high until the final leap.
  Measured coverage of the old gate — shift 6 / 4 steps (turbo): 0 of 4; shift 12 / 12
  (H3's default): 0 of 12; shift 3 / 20: 4 of 20, the first two at gate 0.14 and 0.31. The
  gate now reads position on the schedule's own base grid, recovered from the schedule
  rather than from a shift constant (the shift is only reliable when MiniMaxH3SigmaShift is
  wired, which video sampling does not require). LTX keeps the gate it was validated with.
  Every run now reports its steering window, and says so outright when nothing will steer.

- **Embed guidance crashed every H3 run that reached a steering step**, and score slider
  silently did nothing on the same models. H3 refines the conditioning inside
  `extra_conds`, so the DiT consumes 5376-dim hidden state while the taste store captured
  the raw 5120-dim text conditioning; embed guidance raised the size mismatch, score
  slider caught it and returned the base prediction every step. Learned directions are now
  carried into the consumed space through the model's own text preprocessor, once per
  scene. DynaShift's prompt-similarity weighting was affected too — it compared the raw
  banked prompt against the refined one and weighted every negative equally.
  The value function had the same fault one layer down: fitted on raw conditioning, it was
  handed the refined tensor and threw every step, so the run steered on the fixed direction
  while the report claimed the value function was driving. It is now asked for its gradient
  in the space it was fitted in, once per scene rather than once per step.

- **A resolution-changing second pass no longer fails on MiniMax H3.** A keyframe pin is
  packed as condition rows, so its token count belongs to the grid it was encoded on, and
  `upscale_2x` handed pass 2 a pin from pass 1 — `value tensor of shape [168, 96] cannot be
  broadcast to indexing result of shape [672, 96]`. The pins are resampled onto the new
  grid rather than dropped: on H3 the pin IS the anchor, and pass 2 has to keep holding it.
- **Multi-scene chains work with a second pass again.** `second_pass_op="upscale_2x"` hands
  back a scene at twice the latent size, but every later scene is still built from the latent
  template at the original size — so the carried overlap frames, the anchor's continuation,
  the soft join, the JoyAI memory frame and per-scene guide sources were all being spliced
  into a chunk on a different grid, and the second scene died on the shape mismatch. Anything
  crossing a scene boundary is now brought back to the template's grid on the way. Each scene
  still samples and outputs at 2x; only the carried material is resampled, and only downwards
  — the direction that survives it, since those frames exist to say "continue from here"
  rather than to carry detail. Runs with no resolution-changing op are bit-identical.
- The About panel reads **FunPack 3 "Auspicious Asparagus"** rather than putting the major
  on the codename line.


## [3.5.1] "Auspicious Asparagus" - 2026-08-11

Compatibility and polish on top of 3.5.0. LTX-2.5 works, and the two places it would have
broken silently are fixed. Releases now carry a codename per major version.

### Added
- **Release codenames**, shown in Settings ▸ About.
- **About reports the machine ComfyUI runs on**: chip, memory, GPU (name, VRAM, compute
  capability), free disk, OS, ComfyUI, Python, torch, CUDA, and which fast-attention backend
  is installed. On a rental that is the host, not the browser.
- **A schedule for every sampler.** Steps and scheduler belong to the pass, not to the
  KSampler branch, so any sampler can now be given a computed schedule instead of only a
  hand-typed sigma list. The frames field became usable in the same pass.
- **ALG on any sampler.** The i2v anchor blur used to be locked to FunPack's Distilled Flow
  sampler. It now runs on whatever sampler is wired — a stock KSampler, Hybrid Euler 2S, a
  two-evals-per-step sampler like `heun` — by lifting the guidance out of the sampler loop
  and onto a denoiser proxy driven by the step's sigma.
- **H3 audio clock.** MiniMax H3 denoises video and audio on two different flow schedules
  but hands the sampler one sigma grid, so the DiT reconciles them with a start-of-step
  slope that badly overshoots on a few-step schedule — heard as distortion. `h3_audio_clock`
  swaps that for the chord actually spanning the step. One scalar multiply, no extra model
  call. It works with stock ComfyUI samplers too, and stands itself down with a console note
  on samplers that evaluate twice per step rather than guessing.
- **Projects remember your Editor settings**, so a fresh rental does not reset your
  preferences and shortcut revolver.
- **Every Engine setting says what it does, what it costs and what it needs**, in one line.

### Fixed
- **LTX-2.5 compatibility.** 2.5 reuses the same model classes behind new config flags, so
  nearly everything binds unchanged — but two places assumed 2.3 specifics and would have
  failed *silently*. The video/audio conditioning split was a hardcoded width table; from
  2.5 on ComfyUI reads those widths off the checkpoint, and an unrecognised pair made the
  split return the full width, steering the audio text context along with the video. The
  only symptom would have been degraded audio. The split is now measured off the live model,
  with the table kept as a fallback. Separately, decode-time noise reached only the conv
  decoder: 2.5's diffusion decoder takes no timestep and seeds itself, but is an `nn.Module`,
  so the settings were accepted and then ignored — a knob that read as live and did nothing.
  It is now capability-tested and says so when it cannot apply.
- **`cut_opening_frames` no longer leaves a noisy first frame.** The cut was made in latent
  space, and the LTX video VAE is causal: latent frame 0 is the temporal origin, while every
  later frame was generated as a continuation. Slicing the front off the latent promoted a
  continuation frame to position 0, which then decoded with origin handling it was never
  generated for. The cut now happens on decoded pixels, after a decode that saw every frame
  in the context it was sampled in — the way MiniMax H3 already did it. The count was wrong
  too: a latent cut could only remove whole latent frames and shortened the clip by a
  different amount than the span it removed. It is exact now, N means N. Video comes from the
  IMAGES output on a cut run (the Editor already wires it that way); the latent's audio
  stream is cropped by the same amount of time so sound and picture still start together.
- **Guide keyframes are dropped when `second_pass_op` changes the resolution.** They are
  recorded as token indices into pass 1's grid, so `upscale_2x` left them addressing the
  wrong tokens — which recent ComfyUI rejects outright and older builds mis-placed silently.
  Pass 1 still uses every guide; only pass 2 loses them, and the scene report says so.
- **One ALG control instead of two.** The Distilled Flow panel had its own switch while the
  chain sampler's anchor blur did the same thing on any sampler and already drove it — two
  controls for one behaviour, with precedence depending on which you had touched. The Editor
  now shows one. Projects carrying the old switch are migrated with their strength and
  threshold, and told so.
- **H3 image conditioning actually reaches the model.** Two independent breaks, both silent,
  both ending with a generation that ignored the input image entirely and left the prompt to
  carry the whole shot. The Movie Editor always splits a run into per-scene conditionings,
  and that split re-encoded every scene *without* the anchor image or the ref2va references —
  so neither the keyframe pin nor the `<Picture i>` presentation ever reached the sampler,
  even for a single-scene project. The split now carries the anchor onto the opening scene
  and the references onto every scene. LTX projects are untouched: there the anchor travels
  in the latent, and adding vision to scene 1 would change existing output.
- **A wired `MiniMax H3 Image to Video` node no longer throws its image away.** That node
  carries its first/last frame pins on its CONDITIONING output, which this pipeline dropped
  because the sampler's positive comes from Studio. New optional **Chain Sampler ·
  h3_keyframes** input (auto-wired for H3 projects) salvages the pins — first frame onto the
  opening scene, last frame re-indexed onto the closing scene's own final frame.
- **Spare H3 token tags are trimmed, not thrown away.** An image prompt comes back one tag
  long, and treating the surplus as corruption discarded the whole vector — which on exactly
  those prompts left the DiT modulating the picture as if it were text.
- **Projects remember which model family they are for.** Every project on disk had no family
  recorded while the global default said MiniMax H3, so the builder read them as LTXAV and
  routed the audio VAE at a node H3's graph never builds. That is the reported "phantom Audio
  VAE port": one root cause, four separate bugs, all fixed.
- **A wired `positive_conditioning` satisfies Studio's CLIP requirement**, and an
  fl2va/ref2va node satisfies H3's AV latent requirement — neither should have been blocking
  generation.
- **Warn only when Studio has no conditioning source at all**, rather than whenever one
  particular input is empty.

### Changed
- **Guide strengths span the full 0..1 range.** Mid-scene guide, identity pin, prior-scene
  guide, the per-guide strength field, and the JoyAI memory floor were clamped to 0.25–0.5.
  That band is the measured audio-safe sweet spot, not a physical limit; it stays in the
  tooltips and out of the code.
- **Project texts reach nodes that encode on their own**, expanded — so a custom encoder in
  the graph sees the same prompt the pipeline does.
- The H3 `v2a_grad_scale` warning chip is gone. The knob does nothing without JoyAI audio
  memory on any model, so flagging it as an H3 limitation blamed the wrong thing.

## [3.5.0] - 2026-08-04

A second model family. FunPack was built around LTX-2 / LTXAV and now supports MiniMax H3
as a first-class alternative: a project picks a family, and everything downstream — which
nodes the graph emits, which model files the setup asks for, which sampler settings are
even applicable, what a valid scene length is — follows from that choice. Nothing about
LTX-2 changes.

### Added
- **MiniMax H3 is live.** H3 support was written against ComfyUI's pull request; the model
  has since merged (ComfyUI v0.30.0) and the weights are published, so the pipeline setup
  now offers it as a real, released family with the actual filenames to download. Nothing
  about LTX-2 / LTXAV changes — a project picks a family, and both are first-class.
- **i2v anchors work on H3.** LTX anchors an image by writing it into the starting latent;
  H3 has no such path — its anchor is a condition row packed beside the text. A scene's
  anchor image (and the Cutting Room's per-scene anchors) now becomes H3's own frame-0
  keyframe pin, so image-to-video actually conditions instead of quietly generating from
  noise. A first-frame anchor and a last-frame guide now coexist rather than overwriting
  each other, which is exactly H3's first-and-last-frame mode.
- **Reference images can be encoded at full detail** (Settings ▸ Engine ▸ References):
  *Match output size* as before, or *Max detail (2048px)*, the reference pipeline's own
  sizing. Reference rows ride through every sampling step, so max detail is not free.
- **Frame geometry follows the model.** Frames per scene snap to the chosen family's grid
  (LTX 8k+1, H3 17k+5) in both frontends and in the graph builder, and H3 renders at its
  fixed 24 fps. Picking a model family brings the project onto its geometry and says so.
- **The model family is the first question in Pipeline setup**, not something to find in
  Models afterwards — it decides which nodes the graph emits and which files to download,
  so everything else follows from it. Reachable later via Models ▸ Model family ▸ Setup…
- **Reference media is marked in the Media Bin and wired like any other input.** Any image,
  video or audio takes an `R` mark beside the identity pin, mark order is the numbering the
  prompt uses, and each marked item appears as an input source in Models & Pipeline for
  every socket its kind can fill. At generation the files are copied into ComfyUI's input
  folder and one loader per reference is injected, chosen by what the destination socket
  asks for. Replaces the Reference media tab: references are wired now, not configured.
- **Reference video clips work,** decoded once at 24 fps into both views the model needs —
  the 2 fps timestamped frames the text encoder sees and the full-rate latent block the DiT
  packs — with the clip's own soundtrack riding along as its `<Audio j>` label. Capped at
  15s, because reference tokens ride every sampling step.
- **The KSampler pass can build its own schedule** from `steps` + a scheduler, the way
  ComfyUI's own scheduler does. A SAMPLER object is only the step function, so this pass
  previously had no schedule at all unless you typed a sigma list by hand. The schedule
  dropdown is the switch between the two: `use_user_sigmas` (the default) runs the Sigmas
  field as typed, anything else computes — so a hand-written schedule can stay parked in
  the field and be switched back to without retyping it.
- **A live elapsed timer on the Generate buttons** — "▶ Generating (44s)" — counting from
  the press, in both frontends. After a reload it re-attaches to the running generation and
  says it is counting from the reconnect rather than claiming a total it never saw.
- **A non-finite latent now stops the run where it happened.** A chunk that samples to
  completion and returns NaN/Inf used to sail through everything and surface as an ffmpeg
  AAC error naming nothing that produced it, after the whole montage had been paid for —
  and the video half never complained at all. The sampler now checks both the incoming
  latent and the returned one and says which it was, with the scene label, the stream, how
  many values, and the causes worth testing in order. The hand-typed sigma schedule is
  checked too: an interior zero or a repeated value is an instant Inf, since the solvers
  divide by sigma.
- **A node's own prompt field can be driven by the project prompt** — the linked-input
  "Driven by" list gained Prompt / Negative prompt / Seed, filtered to the widget kinds
  each can legitimately fill.
- **Warnings arrive before a generation is spent, not after.** Beside Generate: when
  Best-FaceID identity transfer cannot do anything, when a scene's source mode wants an
  anchor image but none is picked, and when a sampler setting is inert on the chosen model.

### Changed
- **The sampler-name list comes from ComfyUI** instead of a hardcoded eight written before
  the rest existed — that had been hiding 36 samplers, including `res_multistep`, which is
  what ComfyUI's own MiniMax H3 templates use. A saved name missing from the live list
  stays selectable rather than being swapped for the first option.

### Fixed
- **The log panel no longer eats your selection.** It rewrote its whole body every 1.5s,
  which drops whatever you had highlighted — copying three lines out of a running log was a
  race against the next tick. Updates are now withheld while a selection is held, an idle
  log doesn't touch the DOM at all, and there is a ⏸ Pause toggle plus copy-just-the-
  selection.
- **A core override left behind by a model-family switch is no longer emitted.** Overrides
  are saved per core id, not per node class, so switching family left the replaced node's
  input names on its successor — invisible in the Models panel, fatal at generation
  (`VAEDecodeAudio.execute() got an unexpected keyword argument 'audio_vae'`). The builder
  refuses to write an input the class does not declare, and the family switch prunes them.
- **H3 token tags are reconciled with the conditioning Studio hands on.** The token count
  moves with both the prompt and the resolution, and a mismatch was a bare "list index out
  of range" at step 0 — which is why it looked intermittent and why editing either one made
  it come and go.
- **H3 decodes audio from the sampler latent,** matching ComfyUI's own reference graphs
  rather than handing the audio VAE a different object than they do.
- **Autogrow list entries are addressed by their dotted socket id** (`ref_images.ref_image_0`).
  The bare name is rejected by validation, or slips through and arrives at `execute()` as an
  unexpected keyword. Configs saved under the old name are renamed in place on load.
- **A saved model file that isn't on this machine no longer blocks the whole prompt.** An
  ArcFace projector name with `models/loras` empty made ComfyUI reject everything, even with
  Best-FaceID switched off; editor-supplied values now fall back to the node's default with
  a visible note.
- **The i2v guide stack works again** — a lost dataclass had left it dead.
- **The between-pass sharpen no longer smears** through a box filter on the way back down.
- **The in-app updater survives an upstream history rewrite.** `git pull --ff-only` cannot
  cross one, so the update dead-ended in a git hint; it now asks whether any local commit's
  content is actually missing upstream, realigns when nothing can be lost, and refuses with
  a count when there is real local work.
- **A video reference lost its soundtrack between Studio and the sampler.** Studio had
  already announced the track to the text encoder as `<Audio 1>`, but the reference list it
  handed on dropped the field — so no audio rows were packed for it and every later
  `<Audio j>` in the prompt pointed one reference earlier than written.
- **Easy Gen never knew it was on H3.** Its model config lives on the project rather than in
  a separate slot, which the shared capability check did not read — so every H3-specific
  warning silently reported LTX and never appeared.
- **An off-grid scene length on H3 failed with arithmetic instead of an explanation.** The
  length is now snapped to the model's grid, and a genuine template mismatch names the
  latent node and the number it needs.
- **The run says which H3 checkpoint its conditioning needs.** H3 ships as two DiTs —
  `fl2va` (anchors/keyframes) and `ref2va` (reference media) — that load identically and
  never reject each other's conditioning, so a mismatch only shows up as a poor generation.

### Maintenance
- **The 22 chronic test failures were stub collisions, not redundant tests.** Every one was
  a real test of a live feature: 18 failed only in a full-suite run because each module
  built its own partial `comfy` stub and the first to import won. The package tree is now
  built once and shared. The suite is green and every file passes alone and in reverse order.
- Dead code removed: the orphaned wildcard engine, refiner-state template cluster, unused
  template payload/summary helpers, and unreferenced movie_editor backend helpers.

## [3.4.1] - 2026-07-31

### Added
- **A second sampling pass on the Chain Sampler.** Give it a schedule and every scene is
  sampled twice: pass 1 runs the main sigmas in full, then pass 2 runs its own schedule in
  full, starting from pass 1's finished clip. Nothing is cut short and nothing is derived —
  total steps are simply the two added up — and pass 2 re-enters through ComfyUI's ordinary
  img2img noise scaling, so **its first sigma is a literal strength dial** (0.8 reworks the
  shot, 0.4 polishes, 0.2 is detail work). The i2v anchor stays pinned across the split. In
  the Cutting Room it is one field, **Second pass schedule**, under FunPack Studio ▸ Sampler
  algorithm: typing a schedule enables the pass, clearing it turns the pass off.
- **An optional operation between the two passes** (`none` by default — nothing runs unless
  you pick one). `sharpen` runs one forward of Lightricks' trained 2× latent upsampler and
  resamples straight back to the original size: no video-model calls, a fraction of a step,
  and pass 2 then re-denoises the result, which is what makes it stick. `upscale_2x` keeps
  the 2×, so pass 2 runs at four times the pixels and the scene decodes at double resolution
  (3–5× the cost of the second half, and the i2v pin is dropped — the scene report says so).
  Both use the same upsampler file segmented detailing does, and the model picker now appears
  alongside the operation instead of only under segmented detailing.
- **Cut the opening off i2v scenes** (`cut_opening_frames`, Chain Sampler ▸ Timing & Seed).
  The anchor is a pinned frame at position 0: it carries identity, style and composition
  better than anything that weakens it on the way in — and it is also literally the first
  frame you see, so every i2v scene opens on the exact reference still. The scene is
  generated exactly as normal, anchor at full strength, **no extra sampling**, and the
  opening is then cut off the finished clip: an i2v generation that reads as t2v. Nothing is
  regrown, so the scene comes out that much shorter and the audio is cropped to match.
- **Context windows** for scenes longer than the model's comfortable window (ComfyUI core's
  own mechanism, audio-aware on LTX — nothing ported). Engages only past the window length,
  so shorter scenes pay nothing.
- **The progress readout says what is running**, not just a step count: `sampling 17/26 ·
  scene 2/3 · pass 2 of 2`. The second pass also announces itself on the ComfyUI console the
  moment it starts, rather than only in the run report afterwards.
- **Prompt `$name` variables in Easy Gen** — the shorthand-for-a-full-phrase layer the
  Cutting Room's Composer already had (`$vid = "High quality, high fidelity realistic video,
  motion blur, cinematic fog"`). A **$ Variables** button in the prompt bar opens its own
  window with editable name/value rows, a count badge, and a live warning for names used but
  never declared or for a variable that references itself. Same `project.variables` field as
  the Cutting Room, so a project carries its variables between the two UIs; substitution
  still happens inside Studio dead-last — after shortcut expansion and after the scene split
  — which is what makes a `$var` work *inside* a shortcut's replacement while a value
  containing a comma or a trigger word can never move a scene cut.
- **An Interrupt button in Easy Gen.** The progress panel now carries the same
  **■ Interrupt** control the Cutting Room's player has — Easy Gen previously had no way to
  stop a run short of the ComfyUI queue. A stopped run reports "Generation stopped." instead
  of a failure (an interrupted job is recorded as an error with no media, which is
  indistinguishable from a real crash at the API level), and any partial media it did write
  is still shown.

- **Easy Gen re-attaches to a running generation after a page reload** — the Cutting Room
  already did.

### Changed
- **Variable values are now auto-growing text boxes** (both UIs) instead of one-line inputs —
  a long phrase was effectively uneditable in a single-line field.
- **Plateau step-cache and context windows are mutually exclusive** — the cache cannot tell
  one window from another within a step, so it is skipped with a note in the scene report.

### Fixed
- **Context windows never worked.** The schedule names were ComfyUI core's spelled backwards
  (`uniform_standard` where core says `standard_uniform`), so every choice except `batched`
  raised a `ValueError` out of the sampler and **failed the whole render**. Core's names are
  the choices now; the old spellings are still accepted and mapped onto them, so a saved
  project keeps generating. An unknown name is refused with a reason naming what your
  ComfyUI accepts, instead of escaping as an exception.
- The same call also passed a keyword core's window handler does not always take, which was
  reported as *"ComfyUI core too old"* — silently switching off a feature that core fully
  supports. Unsupported keywords are dropped now, with a note naming the one setting that
  degrades.
- **A rejected prompt now says why.** ComfyUI's top-level error for a validation failure is a
  fixed blob ("Prompt outputs failed validation", no details); the part naming the node, the
  widget and the bad value is in `node_errors`, which the editor never read — so every
  rejected prompt reported the same nothing. Both are shown.
- **Bypass no longer refuses a node over an output nothing uses.** A node that emits an extra
  output the graph never wires (an IC-LoRA loader's `latent_downscale_factor`, say) could not
  be bypassed at all, because a passthrough was demanded for every output rather than for the
  ones actually consumed.
- **The progress bar counted only the first pass**, so a second pass overflowed it and then
  jumped backwards at the next scene.
- **Identity transfer** follows ComfyUI's new RoPE matrix layout.
- **Models & Pipeline** treats ComfyUI V3 MultiType widgets as widgets rather than required
  sockets, so a node using them no longer reads as unsatisfied.
- The latent upsampler loads on ComfyUI v0.29.0, which moved it onto DynamicVram and made
  `operations` a required argument.

## [3.4.0] - 2026-07-24

### Added
- **Easy Gen** (the simplified single-scene UI) reaches near feature-parity with the Cutting
  Room for the workflows it's meant for:
  - Shortcut/transition JSON library **import & export** in Settings ▸ Shortcuts (merge or
    replace) — Easy Gen previously had no way to load a shortcut/transition library at all.
  - The full **Engine settings** panel (Studio refinement/adjustments/sampler algorithm,
    Chain Sampler continuity/timing/guidance/decode/experimental, Best-FaceID identity
    transfer) — restored via the same shared frontend module the Cutting Room uses.
  - A **Log** button (ComfyUI backend log, same as the Cutting Room's).
  - An explicit **Save** button, and a **Gallery** picker to reuse already-uploaded media as
    the i2v anchor without re-uploading — with a one-click **continuity identity pin**
    button right on each thumbnail, mirroring the Cutting Room's Media Bin.
  - An **Export…** button to download the project as a `.funpack_project.json` file (e.g. to
    move a project off a rental GPU box) — the backend route already existed but had no UI.

### Changed
- **Loop temporal style now works alongside guides.** Scenes carrying appended guide frames
  (carry-i2v-guides, mid-scene guide, JoyAI memory, custom guide stacks) no longer disable
  the loop roll: only the content region rolls while the pinned guide tail (and the audio
  memory tail) stays canonical, so guides keep informing every position of the cycle without
  ever being rendered into it. The i2v anchor was always supported - it rides the denoise
  mask, which rolls in step with the latent.
- **Easy Gen's Studio always runs in Prompt-only mode.** Easy Gen has no rating UI, so every
  Engine setting that's a no-op without a trained refinement key/rated history (refinement
  key, value guidance, steer mode, reference injection, embed/score/output guidance, taste
  retrieval, DynaShift, sampler-panel velocity bias/rescue) is hidden there and enforced at
  the pipeline level, not just the UI — every Easy Gen generate call forces Studio's `mode`
  to "Prompt only" and the rating-gated Chain Sampler knobs off via node overrides, so a
  project configured with these on in the Cutting Room still generates plainly from Easy
  Gen. A note points to the Cutting Room or the ComfyUI graph for the full learned refiner.
- The "expose to editor" eye buttons in Models & Pipeline are hidden under Easy Gen — there's
  no per-scene inspector for an exposed control to ever surface in, so they were inert
  clutter that could look actionable and silently do nothing.

### Fixed
- The shortcut autocomplete menu could open off-screen below the prompt field (Easy Gen's
  prompt box sits in a footer bar at the bottom of the viewport) with no way to see or click
  a suggestion. Now flips above the field when there isn't room below.
- A node's **bypass** toggle could silently do nothing: when the pass-through mapping was
  ambiguous, or the bypassed node's own matching input wasn't wired to anything, the failure
  was only ever recorded in a report field no frontend code read or displayed — the node
  stayed fully active with no visible error. Generation now blocks with a clear reason
  instead of a silent no-op.
- Easy Gen could silently revert Models & Pipeline edits (bypass, exposed widgets, LoRA
  settings): any later project save sent a stale in-memory copy of the models config back to
  the server, clobbering whatever was just changed there independently.
- Easy Gen's preview restarted playback (ignoring a pause) on every background health poll
  or prompt keystroke, since each one rebuilt the video element from scratch.
- A "fetching process...aborted by the user agent" unhandled-rejection error banner could
  appear when switching between Gallery/generated media in Easy Gen.
- Downloaded generation results always suggested "result.mp4" as the filename regardless of
  the actual render (no `Content-Disposition` on the `/result` route) — the underlying file
  was already uniquely named per run and already lived in ComfyUI's temp dir, just wasn't
  surfaced to the browser's save dialog.
- Easy Gen's Generation settings showed a "Seed" field always reading "1" and looking fixed.
  It was bound to the Project's legacy `seed` field, which nothing in generation has ever
  read (the real seed resolution only reads `sampler_inputs.seed`, already correctly random
  when unset) — removed the dead field in favor of the working one under Engine ▸ Timing &
  Seed, which already says so.

## [3.3.0] - 2026-07-15

### Added
- **Loop temporal style** - the "loop" option in Temporal style (inert since it was added)
  now produces seamlessly looping clips, the LTX2.3 equivalent of WanVideoWrapper's Loop
  Args: on eligible denoise steps the latent is cyclically rolled along time before the
  model forward and the prediction unrolled after, so the model repeatedly smooths the
  video's seam (last frame | first frame) as if it were an ordinary interior cut. Video and
  audio roll by their own frame counts and stay time-aligned; i2v anchor masks roll in step.
  Near-zero overhead (no extra forward passes). Also honored by the "auto" director when a
  scene's prompt asks for a loop ("seamless loop", "cycling", "repeating"). On scenes that
  carry guides (mid-scene guide, carry-i2v-guides, JoyAI memory, custom guide stacks) the
  roll stays off - guide frames are pinned to absolute positions - and the run log shows
  `temporal_loop_roll(inert: guides pin frames)`.
- **Plateau step-cache** (experimental, Engine ▸ Experimental) - reuses the base model's
  forward across the near-noise plateau steps of the LTX 8-step schedule (sigma >= threshold,
  default 0.975), skipping 3-4 of 8 forwards per scene for a negative-overhead speedup.
  Guidance wrappers still post-process every step's prediction.
- **Wired widget locks** - node widget fields fed by an incoming wire are now locked in the
  Models & Pipeline editor and show which node feeds them, instead of silently accepting
  edits that the wire would overwrite on the next run.
- **Per-prompt taste direction** (`taste_nearest_prompt`, experimental, Engine ▸ Guidance) —
  a retrieval layer over the learned taste steering. Instead of pushing Embed guidance /
  Score slider along the single global liked-direction average, each liked rating now also
  records a `(prompt fingerprint → that run's liked direction)` entry, and with this on each
  scene retrieves the similarity-weighted direction of its nearest rated prompts (a forest
  prompt pulls what worked on forests, not the mean across every prompt). Non-parametric
  (a plain ring buffer keyed on pooled conditioning) — no extra model forward, just a cosine
  lookup + vector mean, and no small value function to collapse into a spurious attractor.
  Falls back to the global liked direction when nothing rated is close enough. Rides the same
  refine_v2 clip-state file as `liked_dir`, so it's removed atomically with the key.

### Changed
- **Best-FaceID identity transfer** (experimental, Engine ▸ Experimental) is now a full native
  port of ComfyUI-BFSNodes' LTX Identity Transfer overlap+source_phase+ArcFace conditioning,
  replacing 3.2.1's source-phase-only approximation. The reference face is now injected as
  separate, non-rendered overlap tokens (never blended into a real frame) instead of tagging
  Continuity's Identity pin keyframe, plus an optional ArcFace projector channel (`identity_projector`,
  `id_strength`, `arcface_mode`) for a second identity signal. Still keyed off Continuity's
  existing Identity pin image — no new picker. Requires the `insightface` package (added to
  requirements.txt; downloads the buffalo_l model on first use) only when an ArcFace projector
  is selected — the overlap tokens alone work without it.
- Frame overlap now carries through anchor changes (hard scene cuts with a new anchor image
  used to drop the latent overlap; `carry_overlap_through_anchor`), and an Identity pin
  configured on an anchor-swap scene resolves again instead of being skipped.

### Fixed
- **Preview stability (rounds 3-5)** - the remaining causes of stale / black / undecodable
  scene previews, especially "the second half stops loading":
  - `/result` faststart-remuxes raw ComfyUI renders (VHS writes the MP4 index at the end of
    the file; browsers cannot seek that) and never serves a video mid-write - a file whose
    index isn't on disk yet answers 503 until the save finishes, instead of undecodable bytes.
  - A failed remux answers 502 with rate-limited re-attempts instead of permanently falling
    back to the raw file; temp-dir failures and the loopback fallback path can no longer
    serve raw index-at-end video either (the fallback also gains Range/seek support).
  - The player materializes video elements only in a window around the playhead instead of
    one per timeline clip - a montage-scale timeline (100+ clips) used to create 100+
    elements and fire ~180 preview-segment encodes at once, starving the browser's
    per-origin connections (and hitting Chrome's hard 75-media-players-per-page cap), which
    left the back half of the timeline permanently "loading". Timeline filmstrips load
    through a small shared queue for the same reason.
  - The monitor shows honest readiness slates ("Video is processing…" / "Video is loading…" /
    "Preview unavailable") instead of a black or blinking frame, and a long GIL-bound save
    can no longer exhaust the retry budget.
- Exposed controls silently reverting: the autosave snapshot was taken before edited model
  values synced back to the project, so an exposed-widget edit could be undone by its own
  autosave.
- ALG crash on multi-scene runs (tensor truthiness in the `alg_latents` fallback).

## [3.2.1] - 2026-07-06

### Added
- **Shortcut Ideas bulb** (💡 next to prompt fields) is now a habit-aware co-pilot: alongside
  the existing missing-category browser, it surfaces shortcuts you've historically paired with
  what's already in the prompt, and what usually follows the previous scene in your own
  projects — mined on demand from saved project text, no new persistent state.
- **Media gallery: one-click continuity pin** — a 📌 hover action on image cards sets/unsets
  the project's Continuity identity pin directly from the Assets gallery, without a trip to
  Engine settings.
- **Shortcut revolver** — an opt-in no-repeat cycling mode for multi-replacement shortcuts
  (sequential or shuffled). A shortcut with several replacements now draws through all of them
  before repeating, instead of a seeded random pick that could repeat early.
- **Guide blur** (`alg_blur_guides`) now stands alone from the sampler's anchor ALG: it works
  with `alg_enabled` off, and has its own strength / sigma-threshold controls independent of
  the anchor's.
- **Best-FaceID identity transfer** (experimental, Engine ▸ Experimental) — tags the existing
  Continuity "Identity pin" guide with the source-phase RoPE rotation Best-FaceID-style
  identity LoRAs were trained on. A native port on the Scene Chain Sampler, no external
  custom-node dependency; the LoRA itself loads the normal way via Models.

### Changed
- Models & Pipeline "+ New node" search collapses to a compact summary once a node is picked,
  instead of leaving the full results list open above the Values/Wiring panel.
- Prompt autocomplete no longer caps suggestions at 8 — a prefix shared by many shortcuts now
  shows every match (the menu already scrolls).

### Fixed
- **Preview stability**: the preview-segment ffmpeg re-encode ran synchronously inside the
  request handler, freezing ComfyUI's event loop and stalling every other streaming response —
  now backgrounded with per-key locking. Ghost clips (and deep-trimmed lone clips) get proper
  segments instead of deep-seeking the raw chain file. Segment URLs encode the clip duration so
  re-trimming a clip can't serve an hour-old cached segment. Pooled preview videos default to
  `preload=metadata` (only the current clip and its neighbours preload fully), so a full
  timeline no longer starves the browser's connection pool mid-playback.

## [3.2.0] - 2026-07-05

### Added
- **Unified macOS-style Settings window** (⌘, or the Settings button) — one door to every
  setting. Icon-rail sidebar that expands on hover, with grouped sections: About FunPack,
  Editor, Engine, Models & Pipeline, Refinement & Taste, Updates & ComfyUI, Temp Files. The
  FunPack menu and the old Settings dropdown are gone — no duplicated entries anywhere; the
  top bar is File / Edit / View / Help / Settings.
- **About FunPack** — version (now served via `/api/git/status` from pyproject), commit,
  branch, and a Software Update deep link.
- **Models & Pipeline redesigned** around its own sidebar: Linked inputs and Pipeline at the
  top, "＋ New node", and every configured node with a status dot and hover rename. Adding a
  node opens a Setup dialog with search where widget values AND output wires / input sources
  are set before it lands in the pipeline. Link mode is a persistent bar that follows you
  across node pages. Import ComfyUI Workflow lives here as a header action.
- **Engine settings redesigned** the same way: always-visible categories (Overview · Studio:
  Refinement / Adjustments / Sampler algorithm · Chain Sampler: Continuity / Timing & Seed /
  Guidance / Decode / Experimental) with macOS-style rows and live changed-count badges — no
  more scrolling a wall of collapsible cards.
- **DynaShift** (experimental) on the Scene Chain Sampler — negative latent memory: Awful and
  wrong-appearance ratings bank the run's video latent, and later runs steer x0 away from
  matched banked frames per late step ("a negative prompt at CFG=1").
- **output_guidance** on the Scene Chain Sampler — a value function trained on the model's own
  prediction, nudging sampling toward liked outputs; exposed in Engine ▸ Guidance.
- **Value function training**: pairwise ranking loss + ensemble disagreement gate.
- **Distilled Flow**: `quality_sharpness` unsharp-mask port from Hybrid Euler 2S.

### Changed
- **Split at playhead actually cuts the clip in place** (both halves keep playing their own
  portion) instead of only adding a marker.
- Settings fields render single-column and shrink correctly, so the window works on small
  screens; no control can push the layout out of reach anymore.
- Removed block steering and the NVFP4 loader after failed live validation.

### Fixed
- **Preview player: rapid play/pause could restart the video from the beginning.** An aborted
  stream fetch (or the retry path's cache-busted reload) reset the `<video>` element to 0 and
  the playhead followed. The player now stashes the position when the element resets, restores
  it as soon as metadata is back, and resumes playback if it was playing.
- Sampler: exception-safe per-scene wrapper lifecycle on interrupt + output_guidance gradient
  calibration.
- Editor: typing no longer interrupted by autosave re-renders; renders no longer become
  unviewable after transient errors; media thumbnails release their network connections
  (Chrome's 6-per-origin pool starvation).
- Models "+ Add" requirement buttons doing nothing on an empty project; the
  disabled-built-in-pipeline state is surfaced instead of hidden.

## [3.1.3] - 2026-06-29

### Added
- **ALG anchor de-staticking** (experimental) on the Distilled Flow sampler — adaptive latent
  guidance that swaps the model's latent image per step (blurred ↔ sharp) to keep i2v anchors from
  freezing the shot. Opt-in, off by default; `alg_blur_guides` extends it to guide-attention
  frames.
- **Momentum Guidance** (experimental, arXiv:2602.20360) on the Distilled Flow sampler — an
  EMA-of-velocity smoothing paired with ALG over complementary sigma windows. Opt-in, off by
  default.
- **Bounded Attention** (experimental) on the Scene Chain Sampler — reduces attribute bleed
  between multiple subjects via text cross-attention masking, with exact-boundary per-span
  encoding and activation/skip logging.
- **Auto Montage.** Build a trailer-style cut from already-rendered clips; works on generated
  scenes, not just imported video clips.
- **Bypass mode for Models-modal nodes**, exposable to the main editor via the existing
  eye-button / exposed-widget pattern.
- **Conditioning Adjustments** (per-phrase universal steering) are now exposed in Engine Settings.
- **Reconnect to a running generation after a UI reload.** If the editor is reloaded mid-run it
  re-attaches to the in-flight generation in ComfyUI's queue — Generate stays blocked, Interrupt
  is offered, and the result is recorded onto the right scenes when it finishes.
- **Temp files browser** (Settings ▸ Temp files) — a media-bin-style view over ComfyUI's temp
  directory (scene previews and other transient outputs); open any item in a new tab or save it to
  disk before a restart wipes it.
- **Switch Branch / Update FunPack from the welcome screen**, so you can swap branches or update
  without first loading a project.

### Changed
- JoyAI-Echo memory strength is no longer capped at 0.5 (now up to 10.0).

### Fixed
- **Guide scenes crashed on Blackwell (sm_120) GPUs.** xformers has no masked (tensor-bias)
  attention kernel there, so guide/mid-scene-guide runs aborted while plain anchors worked; masked
  attention is now auto-routed to SDPA.
- Removed a module-level persistent CLIP encode cache that could carry stale conditioning across
  requests — the UI is the only source of truth.
- Custom shortcut refinement keys are now trained on single-scene runs too, not only multi-scene
  runs.
- Splitting a clip could leave a phantom/duplicated video transition on the first half.
- A stale scene anchor/render could survive a global-prompt reorder; scene matching is now
  content-first.
- A previous scene's preview became unwatchable once the next scene generated (temp files now use
  a unique per-run prefix).
- Audio-track resize could overshoot, and keyboard focus could get stuck on the Start(s) input.

## [3.1.2] - 2026-06-28

### Fixed
- **Cutting Room: splitting a clip could undo itself.** An async probe that adopts a render's
  real encoded file duration could resolve *after* a split and silently rewrite the first half's
  length back to the full original — pushing the second half later, which looked like "a
  full-length scene got appended to the end of the timeline" instead of an in-place split. The
  probe now carries an edit token that the split bumps, so a stale probe result is discarded
  instead of overwriting a fresh split.

## [3.1.1] - 2026-06-26

### Added
- **Prompt `$variables` + global-prompt templates.** Define `$name` variables in the Composer and
  reference them anywhere in the global prompt or scene text; resolved last (after shortcuts and
  scene splitting), so existing trigger/split behavior is untouched. Recursive/cyclic references
  fall back to the literal text instead of erroring. Save the current prompt as a reusable
  template and load it back from a dropdown above the prompt box.
- **JoyAI-Echo memory mode** on the Scene Chain Sampler — a cross-shot audio/video memory bank
  built on guide attention: a video memory bank carries visual identity across shots, paired with
  an audio memory (protected via a masked tail) and a `v2a_grad_scale` control over the native
  video-to-audio attention hook (1.0 = no-op). Toggle in Engine Settings.
- **Contrastive-pair FreeSliders.** The minus pole of the score-space taste slider is now
  synthesized from a learned `bad_dir`, giving a real contrastive pair instead of a single learned
  direction.
- **Scene postfix.** Shared text can be appended to every scene, toggleable, with shortcut
  expansion and key fold-in covered by tests.
- **Cycle-guard + smart auto-wire** for full-control graphs, so automatic wiring no longer fights
  manual connections.
- **Path-outcome planner (phases 1–2d).** Per-scene path-outcome memory with seed-lever-avoidance
  and Thompson/UCB explore-exploit seed routing, plus conditioning-variant routing — ratings now
  steer future generations away from disliked/repeated paths.
- **BachVid + KV-Lock.** Training-free per-key raw K/V identity bank (capture/bless/inject) gated
  by a variance-of-x0 scheduler that drives injection strength.
- Clip length on the timeline now adopts the real encoded file duration instead of the plan
  estimate; project-mode scene length honors `frames_mode` so scenes track project length.

### Changed
- Scene boundaries are now passed structurally end-to-end; the generic `scene N` injection marker
  is removed entirely, and a malformed/incompatible split can never reach the encoder.
- A corrupt or incompatible saved project now opens with a clear error instead of a 500.
- "Just forget it" (refiner reset) now truly leaves no trace.

### Fixed
- Leaked Technique-5 forward hooks could survive a reset/key-delete and cause progressive quality
  drift across runs; hooks are now tagged and stripped on every run.
- Deleting a refinement key was non-atomic and could orphan its `value_fn`/blessed banks (still
  applied after "deletion") and leave behind an invisible keyless Absolute store; deletion is now
  atomic and the keyless Absolute store is surfaced/clearable in the UI.
- JoyAI memory inputs were inserted in the middle of the sampler's `INPUT_TYPES`, desyncing widget
  order in the ComfyUI graph; moved to the end.
- Timeline clip order no longer scrambles after editing the global prompt.

## [3.1.0] - 2026-06-21

### Added
- **Cutting Room rebuilt on an OpenCut-style NLE shell.** Three-column layout — Assets (media
  bin) | Preview (program monitor) | Properties + Timeline — with the prompt-craft tools moved
  into a **Composer** floating window (drag, minimize-to-title-bar roll-up, maximize) holding
  **Compose** (the global prompt), **Shortcuts**, **Splits**, and **Files** tabs. Layout borrows
  from OpenCut — credited in the README.
- **Prompt autocomplete.** Typing in the global prompt or a scene prompt suggests matching
  shortcut triggers (trigger + replacement + category); Enter/Tab accepts and finishes the
  trigger with a trailing space, ready for the next. Includes an Add-shortcut picker (browse by
  category) and works mid-prose, not only after a delimiter. Toggle in Editor settings.
- **"Anchor as guide" scene source.** The scene image still feeds the pipeline like any anchor
  (so nodes that need it — e.g. an Image Transform deriving width/height — still get it) and
  steers the scene from a frame-0 guide, while the i2v node is bypassed so the latent stays empty
  (text-to-video). Per-scene guide strength; declare your i2v node + the input/state to force in
  Editor settings → Anchor as guide.
- **Mixed mode also attaches the scene's own image as a frame-0 guide**, so even the first scene
  (nothing prior to carry) gets guide attention from its image, on top of the carried prior-scene
  guides. The Mixed anchor image now also shows an inspector picker (parity with image modes).
- **Per-browser Editor settings** — prompt autocomplete, shared-anchor toggle, and the
  Anchor-as-guide i2v bypass config.
- **Richer text-overlay styling** — bold/italic, alignment, outline, shadow, and a background box.
- **Composer file management** — Replace/Merge on import, Delete-all, and a Files tab to audit and
  purge FunPack files on disk.
- **Inline clip reorder controls** on the timeline; **managed Shortcut categories / sub-categories**
  for the Compose tab; the node search/filter is now available in every Add-Model mode (not just
  "Any node").

### Changed
- **Canonical scene splitter.** `split_scenes` was rewritten as a single piece-walker shared by
  every consumer (preview + generation), removing the old offset-projection band-aids and divergent
  split paths. "Wrong appearance" ratings now act as a consistency anchor.
- Generated renders are **immutable and full-length** — later plan/trim edits never truncate an
  existing clip. Timeline cut order is independent of plan order; a clip shows what it was generated
  with, not live plan text; removing a scene from the plan keeps its generated clip on the timeline.
  i2v anchors drop on the **Plan**, not the timeline.
- Exposed controls honor real min/max/step, and autosave no longer eats in-progress input.

### Removed
- Removed the `FunPack Scene Builder` node and the Studio Scene Builder feature (the per-key
  scene database, saved scenes, the Studio **Scene** tab, and scene-database wildcard cleanup).
  It was redundant — helpful on paper but effectively unused, with no way to drive it from the
  Cutting Room, and it risked interfering with shortcuts. Shortcuts, transition splitting, and
  refinement keys are unchanged. Studio now always uses the connected `positive_prompt`.
- Removed the **character feature** (no UI surface remained) and the **"Convert to video / scene"**
  buttons.
- Removed **FunPackNormalizingSampler** (redundant).

### Fixed
- "Reset Studio session" silently wiped keys it never listed. The confirmation read the per-scene
  `scene_refinement_keys` (which drops keys via its divergence fallback), while the backend reset
  wipes the scene-count-independent **pool** (`prompt_scene_shortcut_keys.all_keys` — every key
  fired in the prompt). So a second key you were training could be cleared without ever appearing
  in the confirmation. The preview now exposes `refinement_key_pool` (mirrors the backend reset
  exactly) and the confirmation lists from it, so what's shown is precisely what gets wiped.
- A single-transition "silent" split leaked the trigger word into conditioning.
- Shortcut autocomplete never suggested inside prose prompts; `split_scenes` dropped a scene when a
  transition was keyed by expansion; styled text overlays crashed on newer Pillow (float bbox → int).
- The global prompt is now materialized before generate (the screen is the source of truth); the
  timeline re-renders after prompt edits; stale LoRA/combo slot values coerce to live choices; the
  Composer "Minimize" rolls the window up in place instead of acting like Close; the autocomplete
  menu can no longer be orphaned on screen when a panel re-renders.

### Notes
FunPack 3.x development focuses on the Cutting Room frontend; pre-3.0 ComfyUI graph nodes remain
supported with bugfixes only.

## [3.0.2] - 2026-06-15

### Fixed
- "Reset Studio session" now lists the keys it will wipe off a **fresh** preview. The modal read
  `scene_refinement_keys` from the cached preview, which can lag the prompt (debounced refresh, or
  `_distributeGlobalPrompt` carrying the old keys forward), so a just-added key-bound shortcut was
  missing from the wipe list (it would only offer to reset `default`). Arming the reset now awaits a
  preview refresh before computing the list.
- Exposed project-settings dropdowns now refresh with the model list. A node input exposed to the
  main editor (e.g. a wired LoRA loader's `lora_name` + strength) snapshotted its combo options at
  expose time, so "Refresh model list" updated the live spec inside the Models menu but left the
  exposed dropdown stale (a newly installed LoRA was detectable in the node settings but not in the
  project-settings control). Refresh now re-pulls each exposed control's and shared link's combo
  `choices` from the freshly loaded spec and persists, and all three refresh entry points (Models
  modal button, `ModelsModal.refresh`, menubar "Refresh model list") go through one shared path.
- Reward poisoning from prompt-repair ratings (drift / wrong-character / distorted gens). The
  `Wrong *` ratings (appearance/details/action/combos) are prompt-REPAIR signals — "good gen, but
  the words/identity were off" — yet they were feeding their 0.0/low reward into value-function
  training AND the keyless Absolute taste store. That stamped visually-good conditioning as a
  low-value "valley," so the VF ascent (now always the default key, applied to every scene) pushed
  conditioning away from good regions into drift, and Absolute stored good gens as global
  bad-taste. Wrong-* now carry `skip_value_function`: they still drive prompt repair + relative
  per-key direction/category memory, but no longer train any reward asset. Quality ratings
  (Missing-*/Perfect/Nailed/Awful) are unchanged.
- Refinement-key attribution rewritten to the simple, correct model — fixes both "every key
  trained on every clip" AND "custom uploaded keys not detected." The old code re-split the raw
  prompt independently and compared scene counts, then guessed (all-keys **union** → cross-training,
  or, briefly, empty → keys vanished, including single-scene-with-keys). `resolve_scene_refinement_keys`
  now just reads each scene: split with the shortcut-aware `split_timeline_verbatim` (same scenes
  generation uses) and collect the keys whose shortcuts fired in each scene's text (anchor keys
  apply to all). No scene-count comparison, no union/empty guessing — a single scene with a custom
  key is detected, and each scene gets exactly its own keys. Only when the editor's scene count
  genuinely can't be aligned (advisor/repair restructured the prompt) does it fall back to the
  project default key.
- Refinement training law: a scene owned by a custom key trains **only** that key — the project
  **default** key is left untouched for that scene and learns solely from scenes with no custom
  key. Previously every rated scene also trained the default (and, via the union, every other key),
  which is what bloated keys and cross-contaminated taste.

## [3.0.1] - 2026-06-14

### Added

**Multi-refinement-key support in the Movie Editor.** Refinement keys are now a first-class, per-shortcut training signal. (1) The Engine settings → FunPack Studio card has a **Refinement key** field, so a project can set/wire its own key (previously the key was fixed and unreachable even in full-control mode); it feeds the FunPackRefinementKeyLoader for Studio / Chain Sampler / SaveRefinementLatent. (2) The Shortcuts editor gains a **"Use non-default refinement key"** checkbox + key-name field — firing that shortcut means its key is being trained. (3) **Multi-key per scene:** a scene whose prompt fires several shortcuts bound to different keys now has its conditioning steered by *each* key and **averaged/merged** into one (one key ⇒ substitute the default; none ⇒ default). Rating that scene trains **every** participating key. A key counts for a scene only if one of its shortcuts fired in that scene's text (anchor-bound keys count for every scene); attribution falls back to the safe union if the prompt was rewritten or the split diverges.

**Per-scene refinement-key preview.** The timeline preview now shows, on each scene, the refinement key(s) it will actually steer with before you generate — explicit keys (with `(avg)` when more than one), or the project default key (greyed, `(default)`) when no shortcut key fired in that scene. The preview reuses the exact generation-time resolver (`resolve_scene_refinement_keys`), so what you see matches what runs, including the safe-union fallback.

**Session Reset now wipes every key a run trains, not just the default.** Because per-scene multi-key learning trains *every* refinement key whose shortcut fired in the prompt, a Studio Session Reset now clears all of those keys too (project/default key + each non-default key activated by a shortcut), so no stale per-key state survives a reset (`FunPackVideoRefinerV2._v2_reset_prompt_keys`). The Movie Editor's "Reset Studio session" now confirms first, listing exactly which keys will be wiped: *"This action will reset Studio learning for keys: default, key1, key2… To avoid resetting a non-default key, remove the shortcut that activates it from the prompt. Proceed?"* — disarming a mis-armed reset does not re-prompt.

### Fixed

**Shortcut replacement phrases were split on commas, breaking comma-containing replacements.** A replacement like `"The video rapidly cuts, showing the next view."` was torn into two variants at the comma (Movie Editor `splitLines` and the backend `_shortcut_replacements` string fallback both split on `,`). The editor then showed it across two lines, expansion picked one half at random (so it never matched a comma-form transition trigger), and editing it back to one comma phrase reverted on save because the comma re-split every time. Replacements are now "one per line" only — commas stay inside the phrase; triggers still accept comma-separation as before. Existing torn replacements self-heal on the next edit + save.

**Scene prompt text vanished from the timeline after replacing an i2v anchor.** Replacing a scene's i2v anchor is a source-only change — it doesn't alter the prompt-preview key or the selection — so `commit()` replaced `state.project` with the saved project *without* re-rendering the timeline, leaving the stale optimistic DOM (the clip's prompt text could read blank/"empty scene" until a reload or regenerate forced a full re-render). Anchor/source changes now flag the commit to re-render from the authoritative saved state once the save lands.

**Editor timeline dropped a scene when a transition trigger was also a shortcut.** If a scene-cut marker (e.g. `qcut`, `cut`) was *also* a shortcut — common now that cut markers can carry refinement keys (e.g. `qcut` → "cuts" key) — the lossless splitter expanded the trigger word away *before* detecting transitions, so the split was lost or misplaced. Classic symptom: "scene 2 removed from the timeline, scene 1 shows scene 2's prompt" until reload. `split_timeline_verbatim` now also scans the original (verbatim) text for triggers, catching trigger-shortcuts at their true position; genuine shortcut-driven splits (trigger only appears after expansion) and plain triggers are unchanged, and the result still round-trips losslessly.

## [3.0.0] - 2026-06-11

### Added

**FunPack Cutting Room (Movie Editor)** — a full browser-based montage editor served from ComfyUI at `/funpack/movie`. Build multi-scene projects on a real NLE timeline (ruler, proportional clips, trim, split, drag-to-space pauses, crossfades, per-clip effects), preview the whole sequence in a program monitor with seamless scrubbing, and generate or stitch final video without leaving the UI. Includes a Media bin (upload, filter, sort, rename, export), Characters bible with per-scene assign, global prompt editor with lossless split markers, prompt Shortcuts library, Engine settings (FunPack Studio + LTXAV Scene Chain Sampler panels), Models & Pipeline wiring (built-in LTX pipeline or imported ComfyUI workflow), overlay tracks (text/image compositing with WYSIWYG preview), separated and inserted audio lanes, per-scene Refiner ratings, refinement key import/export, ComfyUI log viewer, git-based FunPack update UI, interactive welcome tour with sandbox demo, and first-run pipeline dependency installer (ComfyUI-Manager + LTXVideo / Video Helper Suite / KJNodes packs).

**Interactive Guessing** — a new Batch Training mode. Instead of varying the seed, it freezes *everything* (including the noise seed) and sweeps the **conditioning's spread** (its "sigma") on video channels along a **linear ramp** across the N rungs — up to amplify toward overbake, down to dampen — so you can see exactly where it breaks. Each rung records its factor. When you rate the ladder, it learns your **safe steering ceiling** for that key and **auto-caps future steering**: a soft pre-limit on absolute/relative steer strength plus a hard clamp on the output conditioning's spread (the guarantee). Audio is never touched (reuses the LTXAV video/audio channel split). The Batch Training tab gains **Mode** (Regular / Interactive Guessing), **Direction**, **Range**, and a **Learning** toggle — Learning off (either mode) means pure generation that teaches nothing. Combined with the new IMAGES output (all batch videos come out the Chain Sampler's IMAGES port), you can now generate N slight variations and just watch them without rating.

**FunPack Normalizing Sampler** — a new `SAMPLER` (selectable in Studio alongside Hybrid Euler 2S / Distilled Flow / KSampler) built for distilled few-step LTXAV at CFG=1. Video-only latent normalization counteracts overbaking / oversaturation / colour-drift; audio stays on plain euler. Node `FunPackNormalizingSampler` + Studio sampler panel.

### Changed

The Chain Sampler's **IMAGES output now returns every batch video** concatenated, not just the last one — for both regular batches and Interactive Guessing.

**Latent normalization stacks on the Hybrid and Distilled samplers too.** The same video-only anti-overbake normalization from the Normalizing sampler is now an opt-in `normalize_strength` / `normalize_start_sigma` on the Hybrid Euler 2S (CONST/RF path) and Distilled Flow samplers. Default off; audio is never touched.

**Distilled Flow `AB2 ramp` — graduated 2nd order.** New opt-in toggle that ramps the AB2 contribution linearly from 0→1 across the schedule. No effect at `order=1`. Exposed on the Distilled Flow sampler node and in Studio's sampler panel.

**FunPack development direction (3.0+).** New work focuses on the Cutting Room frontend and its pipeline integration. Pre-3.0 ComfyUI graph nodes remain supported but will receive **bugfixes only** — no major UI reworks on legacy node popups unless a fix requires it.

### Fixed

**Audio corruption in the FunPack Hybrid Euler 2S and Distilled Flow samplers on LTXAV.** AB2 and Heun are now confined to the video stream; audio rides plain 1st-order euler. Distilled Flow's optional `s_noise` is likewise video-only.

Movie Editor: timeline pause UX (drag clips apart to create pauses, drag flush to remove), preview minibar crash on gap segments, pipeline dependency installer with ComfyUI-Manager bootstrap and cancel, overlay/audio lane stability, multi-scene chain preview offsets, and numerous NLE polish fixes across the dev cycle.

## [2.7.8] - 2026-06-04

### Added

Made all conditioning steering **audio-safe on LTXAV**. LTXAV conditions video and audio from two separate text cross-attentions that the model carves out of one conditioning tensor by splitting its channel dim (`comfy/ldm/lightricks/av_model.py` `_prepare_context`: `torch.split(context, [v_context_dim, a_context_dim], -1)` — leading channels → video `attn2`, trailing → audio `audio_attn2`). Previously every steer (relative/absolute pull, value-function ascent + search, embed_guidance, the attn2 direction patch, and manual Conditioning Adjust) shifted the *whole* tensor, corrupting the audio's own text conditioning and degrading audio. Now a shared `protect_audio_channels` confines every edit to the video channel-slice and restores the audio slice from the unsteered conditioning — effectively "modified conditioning for video, original for audio", with no model patching or extra forward pass. The split is auto-detected from the channel width (7680→3840, 6144→4096) and logs the detected layout once; single-stream LTXV (unrecognised width) is a clean no-op, so video-only models are unaffected. Note: this protects the audio's *direct* text conditioning (the dominant lever); the per-block cross-modal `video_to_audio_attn` can still carry a weaker second-order influence from heavily-steered video.

Added an **Absolute / Relative steer mode** to Refiner V2 and Studio. *Relative* (the default, unchanged behaviour) is per-prompt: it learns and applies the best conditioning for one specific prompt. *Absolute* is prompt-agnostic — it accumulates a single global "taste" prior across **every** rated generation and pulls conditioning toward it regardless of the prompt. In Absolute mode a rating means "this has (or lacks) details I like **in general**", not prompt adherence: a Perfect means "I love this, give me more of it everywhere", and a low rating means "this is missing the details I like in general". The pull layers two engines — a pooled liked/disliked direction (the learned, automated analogue of the manual Conditioning Adjust phrase shift) and the keyless value function on top — and is applied at the conditioning output (`_v2_finalize_conditioning`). *Both* keeps the per-prompt fit and layers the global prior under it; pure *Absolute* bypasses per-prompt memory entirely. The global store learns from rated runs even with no refinement key wired (it is keyless by design). The Scene Chain sampler's `embed_guidance` gains an `embed_guidance_source` (`relative` / `absolute`) so the per-step nudge can draw from the global taste direction too. Studio exposes `Steer mode` + `Absolute strength` in the refiner panel.

Ported **velocity bias + reactive rescue** to the `FunPack Distilled Flow` sampler (`sample_funpack_distilled_flow`), reusing the same capture/memory/rescue machinery as the Hybrid Euler-2S and LTXAV/RF samplers. The sampler now exposes `velocity_bias_mode`, `velocity_bias_strength`, `velocity_bias_source`, `velocity_refinement_key`, `rescue_mode`, `rescue_threshold`, and `rescue_strength`, and these are wired into Studio's **Distilled Flow** sampler config panel (with the same blank/`default` velocity-key → wired-refinement-key fallback the Hybrid panel uses). Steering is applied magnitude-preservingly and, on packed LTXAV latents, confined to the video stream (audio-safe). Note: few-step distilled schedules may only land on one or two velocity targets, so apply/rescue fire less often than on an 8-step run; they no-op cleanly when no target matches. With everything off the sampler is byte-identical to the previous deterministic ODE.

## [2.7.7] - 2026-06-03

### Added

Added **Batch Training** — a controlled-batch RLHF workflow built around the principle that the cleanest learning signal comes from rating several generations that differ in exactly one thing. When a batch is active, the Scene Chain sampler runs the chain `N` times with everything frozen except the noise seed, producing `N` directly-comparable videos. Studio then shows a rating panel where every variant is scored, and on submit the value function trains on the batch's shared (frozen) conditioning with each variant's reward — same conditioning, N rewards, which is the variance-reduced comparison signal. Spans the full pipeline: the engine (controlled N-run mode on the sampler), a Studio variant producer that packs the variants into conditioning, the `/funpack/batch` server routes and rating window, and the value-function intake on submit. Phase 3b adds deeper learning from batch ratings via axis and direction memory. Batches live in ComfyUI's temp directory and are wiped on restart.

Added a **reactive in-flight rescue** system to the Hybrid Euler-2S sampler. During sampling the trajectory is compared against conditioning-clustered memory of past good and bad runs; when a step drifts toward a known-bad trajectory it is nudged back. Rescue is rating-gated (separate good/bad trajectory banks, learned only from rated runs), prompt-aware (clustered by conditioning so the right memory is consulted), runs in both sampler phases, and persists its trajectory banks to disk. The full feature set was also ported to the LTXAV rectified-flow (CONST-model) path via a sampler mirror, so velocity capture and rescue now work for LTXV/LTXAV, not just the discrete-step models.

Added **Monte Carlo conditioning search** after the value-function gradient ascent, plus a VF final gate, so the conditioning chosen for generation is the best of a sampled neighbourhood rather than the raw ascent endpoint.

Added a Studio **Variability macro** and an **active-feature readout** for the Hybrid sampler, summarising which steering features (velocity bias, rescue, embed guidance, value guidance) are live for the current configuration.

### Changed

Unlocked `velocity_bias_strength` from a `0.35` cap to `3.0` for deliberate creative action injection. Velocity bias is an artistic / action-injection control (and carries an emergent scene-cut prior — cuts survive the trajectory mean and get reintroduced unprompted); it is not a consistency tool.

Refiner V2 now **always trains the value function**; `value_guidance` only gates whether the learned reward is *applied* during sampling, and defaults on. This means rated runs keep teaching the value function even when guidance is off.

Decoupled `eta_final` decay from the quality boundary — it now anchors to schedule progress instead, so ancestral-noise decay tracks the sampling timeline rather than a quality threshold.

### Fixed

Audio-safe LTXAV sampling: ancestral noise and trajectory steering are now applied to the video latent only, never the audio latent, preventing the joint-attention audio corruption that steering on the combined tensor caused.

Velocity-bias anti-softening: the bias is now applied magnitude-preservingly with sigma-decay and a quality-sharpness term, and sourced from the nearest trajectory cluster, so it injects motion without washing out detail.

Audit fixes: corrected a velocity commit key mismatch, sparse rescue targets, and a redundant aspect-bucket computation. Batch Training fixes: distinct per-variant seeds in both the split-scene and single-scene paths (identical seeds were producing identical videos), correct activation alongside `split_by_transitions`, the node rating is ignored while a batch is in progress, and the in-node panel was de-duplicated from the Studio Refiner tab.

## [2.7.6] - 2026-05-30

### Added

Added **pre-generation conditioning ascent** driven by the value function. Before sampling, the positive conditioning is moved by gradient ascent toward higher predicted reward, in both the single and scene-split Refiner V2 output paths. Displacement is capped to prevent reward hacking (the value function will otherwise push conditioning into degenerate high-score regions). The value function can be exported and imported from Studio and is cleared on session reset.

Added **VF-driven conditioning shaping**: confidence scaling and gradient-aligned phrase boosting in conditioning memory, attention-weight accumulation, and VF-driven temperature, so the learned reward influences which phrases and attention weights are emphasised.

Added **concept-in-context conditioning guidance** and **concept-pair bad-direction repulsion** — conditioning is nudged toward liked concepts in context and away from concept pairs learned to be bad.

Added a **motion floor** to embed guidance: when temporal variance falls below a threshold the guidance auto-boosts it, fighting static/frozen output. Activation is logged with step, sigma, and variance ratio.

Reworked the rating UI into a **rating picker popup**. Added a `Wrong` action and an explicit quality rating, a `Nailed it` rating (prompt-adherence positive, weaker than `Perfect`), and replaced the standalone `Loved it` rating with a per-option **heart modifier** (an axis-blind quality endorsement layered on top of any rating). The heart is disabled for quality-degraded ratings and for `Awful`.

### Changed

Disabled `perfect_freeze` — prompt changes made after a `Perfect` rating are now respected instead of the conditioning being frozen.

### Removed

Removed the experimental latent value function (VGG-Flow-style per-step latent steering). It was added in this cycle but pulled: the LTX token format does not expose the latent in a form the per-step steering could act on.

### Fixed

Fixed an `UnboundLocalError` (`eff_key` not yet assigned when the value function loads), value-function loading needing `inference_mode(False)`, several rating-picker bugs (stale outside-click listener re-opening the picker, active state not updating on option change, clicks landing on a canvas button widget rather than a DOM element), and `Save Refinement Latent` not executing.

## [2.7.5] - 2026-05-29

### Added

Added **embed guidance** — a per-step nudge of the conditioning toward the learned "liked-quality" direction during sampling. Near-free overhead; requires a wired `refinement_key_input` with enough liked generations to have formed a direction. Exposed as `embed_guidance` / `embed_guidance_strength` on the Scene Chain sampler.

Added an **online value function** for reward-guided sampling — a small MLP trained on rated generations that predicts reward and can steer sampling toward it. Required several fixes to train and run gradients inside ComfyUI's `inference_mode` execution context.

Added **mid-scene guide** (`mid_scene_guide` / `mid_scene_guide_strength`), replacing the broken `self_consistency` feature. It uses the LTX guide-attention mechanism rather than post-block hidden-state injection (which corrupted audio through joint attention). At ~`0.25`–`0.3` strength it preserves static-element layout across a scene; capped at `0.5`.

Added a **vision-conditioning toggle** to the Studio popup, and moved reference-image conditioning into Studio, where the source image already lives.

### Changed

Removed the predefined transition-phrase list — scene splitting is now fully driven by user-defined and auto-detected transitions.

### Removed

Removed the **per-scene vision re-encoding and `clip` input** from the Scene Chain sampler, and reverted the per-scene reference-image encoding in Studio (both added in 2.7.4). They were unstable in practice.

Removed `self_consistency` (corrupted audio via joint attention — superseded by `mid_scene_guide`), the `i2i_scene_cut` feature, and dead guide-keyframe / guide-conditioning code paths.

### Fixed

Fixed `i2i_strength` inversion (higher now means stronger reference influence) and replaced the 1-frame i2i anchor with 2-frame i2v anchor generation for hard cuts.

## [2.7.4] - 2026-05-28

### Added

Added **K/V in-context conditioning** for LTXAV identity blocks during i2v generation. Reference hidden states are captured at the start of each scene's denoising pass and prepended as extra attention tokens to the identity-formation blocks ([14, 20, 21, 30, 33]). This forces the model to attend to the reference character's appearance during every self-attention step in those blocks. Result: strong character consistency across scene cuts, view changes, and orientation changes with no LoRAs required.

Added **Gemma3 vision prompting**. When `source_image` is connected to Studio and the CLIP was loaded with a Gemma3-12B checkpoint via `DualCLIPLoader`, Studio automatically encodes the reference image through the built-in SigLIP vision encoder and feeds the resulting vision tokens into the text conditioning. This means the model conditions on both the prompt text and the actual pixel content of the reference frame. No extra node is required — `DualCLIPLoader` already loads the vision weights when present in the checkpoint.

Added **per-scene vision re-encoding** to `FunPack LTXAV Scene Chain Sampler`. Connect an optional `clip` input. After each scene is sampled, the next scene's conditioning is re-encoded using the previous scene's decoded last frame as vision context. This gives identical scene texts genuinely different conditioning based on runtime-generated content, so the model knows what state it came from when building the next scene.

Added **duplicate scene text differentiation**. When two or more scenes share identical text, the second and subsequent occurrences are encoded with a `"Returning to an earlier scene: "` prefix. The original text is preserved in metadata for logging. This breaks the shared conditioning cache entry so the model receives distinct input for each occurrence.

### Improved

Reworked `frame_overlap=0` soft continuation: the previous scene's last **4 frames** are now prepended with **mask 0.4** (partial denoising) instead of 1 frame at mask 0.0 (fully pinned). The model receives temporal context from the previous scene while retaining enough denoising freedom to commit to orientation and pose changes directed by the text prompt.

### Fixed

Fixed anchor text punctuation when joining scene segments. If the character description ends with `.`, `!`, `?`, or `,`, a plain space is used as the separator instead of injecting a redundant `, `. Previously a description ending with a full stop produced `"description., scene text"`.

### Removed

Removed `FunPackGemmaVision` node. It was redundant: `DualCLIPLoader` with a Gemma3-12B checkpoint already loads `vision_model` and `multi_modal_projector` weights through the normal `load_state_dict` path, so the manual weight injection the node performed was a second load of the same file. The vision capability check now detects the attributes directly instead of relying on an injected flag.

## [2.7.3] - 2026-05-27

### Fixed

Fixed `carry_i2v_guides` soft continuation when `frame_overlap=0`: the anchor mask is now fully pinned (0.0) to prevent denoising from disturbing guide tokens.

### Notes

Using `frame_overlap=0` together with `carry_i2v_guides=True` is confirmed to produce bad results and is not recommended. Both parameters now warn about this in their tooltips and in the sampler documentation. Use this combination only for deliberate testing.

## [2.7.2] - 2026-05-22

### Added

Added **Shortcuts** system. Activation phrases in the prompt are replaced with full cinematic descriptions before encoding. Multiple replacement options are randomly picked per seed. Empty replacement removes the matched phrase entirely. Longer phrases always win over shorter overlapping triggers. Managed in the new Studio Shortcuts tab with Add/Save/Delete/Import/Export.

Added **Transitions** system. User-defined transition phrases extend the built-in split list. Custom entries support a placement override - `start` (transition opens the new segment), `end` (transition closes the previous segment), or `silent` (split happens but the phrase is stripped from output entirely). Managed in the new Studio Transitions tab.

Added global **Transition placement** setting in Studio Refiner tab (`start` / `end` / `silent`), with per-entry override on each custom transition entry.

### Fixed

Removed all single-word temporal markers (`next`, `suddenly`, `later`, `finally`, etc.) from the built-in transition phrase list. They caused false splits on normal prose.

Fixed `_GENERIC_SCENE_LABEL_PATTERN` matching "scene proceeds", "scene features", "scene shows" and similar noun-verb constructions as scene labels.

Fixed dangling trailing transition segments (prompt ending with "...cuts to the") being kept as near-empty scenes.

Fixed stray comma artifacts after article words when transition phrase merging occurs.

Fixed custom transition triggers ending with punctuation (e.g. "Scene cut.") not being detected due to a misplaced word boundary.

## [2.7.1] - 2026-05-21

### Added

Added successful seed memory for `FunPack Studio`. When the `seed` output is connected, `Perfect` and `Loved it` ratings store the previous run's sampler seed under the active refinement key. Future runs occasionally reuse concept-matched successful seeds while keeping normal fresh seeds as the default path.

Added per-scene seed metadata for prompt-split mode. Studio and Refiner V2 keep the public `seed` socket as a single integer, while each detected scene conditioning entry can carry its own `funpack_scene_seed` for the Scene Chain sampler.

Added `use_same_seed` to `FunPack LTXAV Scene Chain Sampler`. When enabled, every scene uses the first provided scene seed or the base seed. When disabled, each scene uses scene seed metadata or falls back to `seed + scene_index`.

### Changed

Made Scene Chain i2v guide carry opt-in. The default path no longer appends protected i2v frames as hidden guide tokens, preserving cleaner scene cuts unless the experimental option is deliberately enabled.

### Fixed

Fixed compact i2v guide masks failing to concatenate with full spatial Scene Chain masks by broadcasting guide masks to the chunk mask shape.

Updated README release notes and added the Intent section.

## [2.7.0] - 2026-05-21

### Added

Added `FunPack LTXAV Scene Chain Sampler` for split-scene LTXV/LTXAV continuation in one ComfyUI run. It consumes multi-entry positive conditioning from `FunPack Studio` or `FunPack Video Refiner V2`, samples one scene chunk per conditioning entry, increments the seed per scene, preserves overlap from the previous chunk, and blends/appends chunks in latent space.

Added support for plain LTXV video latents and nested LTXAV video/audio latents in the Scene Chain sampler. For nested AV latents, video and audio tensors are continued together, with audio overlap derived from the video/audio latent length ratio.

Added broad order-only scene splitting for `split_by_transitions`. Scene labels such as `scene ten`, `scene -999999`, and `scene minus infinity` are transition cues, but their written labels never affect scene numbering or order.

Expanded transition phrase detection with scene progression, camera shift, zoom, final shot, and final transition phrases.

### Changed

`split_by_transitions=True` now returns one conditioning entry per detected scene through the existing `modified_positive` output. No new Refiner V2 or Studio output sockets were added.

The text before the first transition is treated as a shared character/global anchor and prepended to every detected scene conditioning. This is intended to improve character consistency across generated chunks.

Removed the hard 8-scene cap from Refiner V2 split output and Scene Chain sampler execution. `max_scenes` still defaults to `8`, but users can raise it for longer chains.

Standalone `then` is no longer a transition trigger. More specific phrases such as `and then`, camera transitions, scene labels, and explicit cut/transition language still split scenes.

### Fixed

Fixed Studio's Scene Builder mode dropdown not refreshing the active tab immediately after selecting a new mode.

### Warning

`FunPack LTXAV Scene Chain Sampler` is resource heavy. Long chains create large final latents and may run out of memory during VAE Decode even when sampling succeeds. Start with short scenes and a modest `max_scenes`, then increase carefully.

## [2.6.0] - 2026-05-16

### Added

Added `FunPack Studio` - a single node that replaces the typical chain of Refinement Key Loader, Scene Builder, Apply LoRA Weights, LoRA Loader, Video Refiner V2, and Conditioning Adjust with a tabbed popup editor. All settings are managed inside the popup; only the rating widget and Open Studio button are visible on the node face.

Studio inputs (in order): model, clip, advisor_clip, positive_conditioning, negative_conditioning, clip_vision_output, source_image, lora_stack, positive_prompt, negative_prompt, user_intent_prompt, feedback_prompt, refinement_key_input.

Studio outputs (in order): model (LoRAs applied + attn2 direction patch), modified_positive, negative (encoded from negative_prompt or passed through from negative_conditioning), seed (for wiring to sampler), high_pass_sampler, high_pass_sigmas, low_pass_sampler, low_pass_sigmas, loss_graph, status, training_info, encoded_prompts.

Studio popup tabs:
- **Session**: refinement key management and Scene Builder mode selector (Pass-through / Manual / Auto / Learning).
- **Scene**: scene preset load and save, phrase bank from session memory, positive prompt composer.
- **Refiner**: all Refiner V2 settings including negative prompt field, feedback, and intent override. Shows a banner and disables the intent field when Scene Builder is active.
- **Advisor**: enable/configure an internal HuggingFace CausalLM advisor. Uses the same model cache as the standalone Advisor LLM node.
- **LoRA**: full LoRA pipeline - session weight suggestions are read first, then LoRAs are applied to model and CLIP, then the direction patch is applied on top. Supports model type (ltx2/wan) and per-block settings.
- **Sampler**: configure Hybrid Euler 2S, Distilled Flow, or any KSampler for the high-pass and low-pass outputs independently. Sigma schedules entered as comma-separated floats.
- **Adjustments**: phrase-level conditioning adjustments with session phrase bank.

Three text inputs (refinement_key, feedback_prompt, user_intent_prompt) have override toggles: when off, connected inputs win; when on, popup values win.

The popup remembers its last active tab per node via localStorage. All field changes auto-save to the node widget after 600ms, so settings survive page refresh without requiring Close to be clicked.

Added `negative_prompt` encoding to Studio: when no pre-encoded `negative_conditioning` is connected, Studio encodes the `negative_prompt` text via CLIP internally, removing the need for an external CLIPTextEncode node.

Added `/funpack/available_loras` and `/funpack/phrase_memory` backend endpoints used by Studio's LoRA picker and phrase banks.

Added `FunPackConditioningAdjust` standalone node for phrase-level conditioning adjustments. Encodes each phrase via CLIP, computes a unit-norm direction from the base conditioning, and applies it at user-set strength. Positive pushes toward the phrase, negative pushes away. Popup editor with session phrase bank.

Added `seed` output to `FunPack Video Refiner V2` (via optional `_seed` parameter) so Studio can generate a seed and expose it as an output for wiring to samplers.

### Fixed

Fixed `FunPackAdvisorLLM` tokenize failing with `AttributeError` when `apply_chat_template` returns a `BatchEncoding` instead of a plain tensor in newer transformers versions. Explicit `hasattr(result, "input_ids")` check now handles both return types.

Fixed advisor generation producing no output for Qwen3 and other chain-of-thought models: thinking tokens (`<think>...</think>`) were not being stripped because they are special token IDs that disappear with `skip_special_tokens=True`. Switched to decoding with `skip_special_tokens=False` then stripping thinking blocks with regex. Truncated thinking blocks (no closing tag, token budget exhausted) are also stripped.

Fixed `FunPackAdvisorLLM` attention mask warning and erratic output (echoed prompts, missing spaces) caused by `pad_token == eos_token` without an explicit mask. Now passes `torch.ones_like(input_ids)` as attention mask to all generate calls.

Fixed `FunPackConditioningAdjust` adjustments not applying for LTX/Gemma3 conditioning: the node was reading `pooled_output` which is `None` for T5-based encoders. Now uses `conditioning.mean(dim=(0,1))` on the sequence tensor, matching how V2 handles conditioning internally.

Fixed Refiner V2 advisor diagnostic not being generated in Full mode: the LLM analysis prompt was asking for session-wide pattern recognition when no history existed yet. Now adapts - uses a simple per-run analysis on early sessions and session pattern analysis when history exists. Added a rule-based fallback so diagnostic history always accumulates even when the LLM produces empty output.

Fixed `perfect_repair_phrases` and `_v2_emphasized_prompt` injecting phrases regardless of the `prompt_repair` toggle. Both are now gated behind `prompt_repair=False`.

Removed the Perfect-rating advisor gate. The advisor previously skipped both analysis and repair when rating was Perfect and no text feedback was provided. Perfect is not a ceiling - the advisor now runs normally for Perfect ratings.

## [2.5.3] - 2026-05-16

### Fixed

Fixed `FunPackAdvisorLLM` wrapper not triggering advisor generation. Qwen3 and other chain-of-thought models emit `<think>...</think>` blocks that were not stripped, causing the parsed repaired prompt to contain reasoning text. This made body-similarity validation reject the result as "too far from intent." Fixed by stripping thinking blocks in `decode`. Also added `enable_thinking` kwarg support in `tokenize` (with TypeError fallback for models that don't support it) and expanded `max_new_tokens` by 2048 when thinking mode is active so the reasoning budget does not crowd out the actual response. Fixed `_v2_text_semantic_similarity` returning 0.0 for generation-only clips that have no `encode_from_tokens_scheduled` - these now return 1.0 (skip the semantic gate) instead of causing spurious rejections. Fixed `FunPackAdvisorLLM` missing from the standalone import block in `__init__.py`, which broke the test suite.

Removed Perfect-rating advisor gate. The advisor was silently skipping both the analysis pass and the repair pass whenever the rating was Perfect and no text feedback was provided, even when the user had an active advisor mode. Perfect is not a ceiling - if the user provides `feedback_prompt`, it must be honored regardless of rating. The only remaining guard is the `allow_prompt_change` check (Learning mode).

## [2.5.1] - 2026-05-16

### Added

Added `FunPackAdvisorLLM` node. Loads any HuggingFace CausalLM (including sharded checkpoints) as an advisor for Refiner V2. Connect the output to `advisor_clip`. Model is cached after first load so subsequent runs do not reload. Also fully compatible with the built-in `TextGenerate` and `TextGenerateLTX2Prompt` nodes - supports `skip_template`, `min_p`, `presence_penalty`, and progressive fallback for unsupported generation parameters.

Added `_v2_direction_readout` to training_info Adaptation section. Shows each direction memory slot in plain language: run count, magnitude, whether it is in direction mode or lerp fallback, and the role each axis is playing this run.

### Changed

Advisor prompt format rewritten to natural language. Both the repair and analysis user messages now read as plain enhancement requests rather than structured field-value pairs. Works with enhancement-type models (Sulphur, Qwen prompt enhancers) as well as instruction-following models. System prompt reduced to one sentence.

Direction-based conditioning now uses `max_new_tokens` instead of `max_length` in the `FunPackAdvisorLLM` wrapper so prompt length does not eat into the generation budget.

Model patch status expanded to show which direction slots are active with run counts and which phrase texts are being emphasized in cross-attention.

Adaptation status block rewritten to multi-line readable format showing strength, reward trend, streak, per-slot mode (direction vs lerp fallback), and axis adjustments applied this run.

### Fixed

Fixed `_v2_generate_advisor_text` returning `None` when layer 1 tokenization succeeded - `generate`/`decode`/`return` were inside the `except TypeError` block so they only ran when layer 1 failed.

Fixed session reset not clearing `intent_expansion_memory`, `session_source_mean_count`, `liked_dir`, and `bad_dir` - these fields were missing from `_v2_empty_state` and survived reset via `setdefault`.

Fixed advisor repetition loops: `repetition_penalty` raised from 1.05 to 1.3, added `no_repeat_ngram_size=5`, temperature raised from 0.5 to 0.7.

Fixed system prompt bleeding into advisor output by splitting prompts into `(system, user)` tuples and applying the model's native chat template. Three-layer fallback: native `system_prompt` kwarg, manual `apply_chat_template` via BFS, flat string with completion anchor.

Added persistent cross-run encode cache (`_V2_PERSISTENT_ENCODE_CACHE`, 4096 entry cap) so phrase encodings are not recomputed every run when CLIP and text are unchanged.

## [2.5.0] - 2026-05-15

### Added

Added a CLIP text-generation advisor to `FunPack Video Refiner V2`. The advisor runs two sequential passes: an analysis pass that identifies what specifically needs to change in the suggested prompt, followed by a repair pass that applies those findings. Both passes see the current suggested prompt, user intent, previous prompt, memory suggestions, and the full feedback history.

Added `advisor_clip` input to use a separate generative CLIP/Gemma model for the advisor while the main `clip` continues handling encoding and similarity checks.

Added `advisor_mode` dropdown: `Off`, `Only diagnostics`, `Only prompt`, `Full`. In `Full` mode both analysis and repair passes run. In `Only diagnostics` mode only the analysis pass runs and its finding is stored in feedback history for the next run. In `Only prompt` mode only the repair pass runs silently.

Added `feedback_prompt` optional input. When connected, the user's natural-language description of what was wrong is placed first in both advisor passes, with the system instructed to follow it exactly and override all other repair logic.

Added persistent `advisor_feedback_history` stored in V2 session state. Up to ten past feedback entries accumulate across runs, each labelled with the corresponding rating: `Missing action: he was supposed to hold her hand not her head`. Advisor-generated diagnostics from `Only diagnostics` runs are stored as `Advisor note:` entries so they carry forward into subsequent `Full` runs.

Added `Prompt only` execution mode to `FunPack Video Refiner V2`. All prompt shaping runs as normal but conditioning vectors are passed through unchanged. Learning still applies. Useful when conditioning adaptation should be paused while prompt refinement continues.

Added `prompt_repair` boolean input to `FunPack Video Refiner V2` (default on). Turning it off disables the rule-based phrase injection from phrase memory and passes no repair candidates to the advisor. Useful early in a session before enough context has been built.

Added `encoded_prompts` STRING output to `FunPack Video Refiner V2`. When the advisor ran and produced a suggestion, the output includes up to four labelled sections: `Positive prompt` (what was encoded), `Advisor suggestion (applied/rejected)` (what the advisor generated), `Advisor analysis` (the diagnostic text from the analysis pass), and `Pre-advisor prompt` (the prompt before the advisor rewrote it).

Added `eta_final` parameter to `FunPack Hybrid Euler 2S Sampler`. When set below `eta`, ancestral noise strength decays linearly toward this value as sigma approaches the quality phase boundary, smoothing the transition into deterministic refinement. Default `1.0` preserves existing behaviour.

### Changed

Replaced the Refiner V2 advisor system prompt with a structured repair format. The advisor now receives four explicit variables — `ORIGINAL_USER_INTENT`, `LAST_PROMPT`, `RATING`, and `OPTIONAL_NOTE` — and is instructed to rewrite `LAST_PROMPT` to fix the specific failure described by the rating. Memory suggestions, feedback history, and analysis context are folded into `OPTIONAL_NOTE`. The repair pass for `Only prompt` mode outputs a plain prompt string with no labels, matching the instruction to output only the final text.

Removed `negative_prompt` input and `modified_negative` conditioning output from `FunPack Video Refiner V2`. Negative conditioning has no effect at CFG=1.0 with NAG guidance and added a redundant AI generation call in every mode. Both the rule-based negative repair and the negative advisor pass are removed.

Increased advisor token budget: repair pass 800 → 1600 tokens, analysis pass fixed at 1200 tokens (was `repair // 2 = 400`). The analysis limit is now independent of the repair limit so it does not shrink if the repair budget changes.

`FunPack Hybrid Euler 2S Sampler` early phase now uses an order-2 ancestral denoised extrapolation (Adams-Bashforth 2-step) in addition to the existing Euler-A update. The previous step's denoised estimate is used to extrapolate a better score direction at zero extra model-call cost. The state resets after any motion pulse.

`FunPack Hybrid Euler 2S Sampler` quality phase now uses a progressive `correction_blend`: the first half of quality steps use a single-eval Euler ODE pass; the second half use the configured 2S correction. This reduces model calls in the quality phase while concentrating the expensive correction where sigma is lowest and it has the most impact.

### Fixed

Fixed `Only prompt` advisor mode running two AI generation calls per invocation (one for positive repair, one for the now-removed negative advisor), causing each run to take twice as long.

Fixed `encoded_prompts` always showing only `Positive prompt:` regardless of advisor activity. The final return path was calling `_v2_encoded_prompts_output` without the advisor keyword arguments.

Fixed Refiner V2 advisor generation: `do_sample` was `False`, forcing greedy decoding and silently ignoring temperature, top_k, top_p, and all sampling parameters. The model was always producing its highest-probability default output regardless of instructions or feedback. Changed to `do_sample=True` with temperature 0.5.

Fixed advisor validation silently rejecting valid feedback-driven repairs: the intent-distance check and protected-category checks now bypass when `feedback_prompt` is connected, allowing the advisor to implement what the user explicitly requested.

Fixed `_v2_find_perfect_example_for_intent` accessing a field (`loved_delta_sources`) that was never written. It now correctly reads from `perfect_anchors` and `loved_variants`.

Fixed `_v2_update_streaks` updating conditioning strength signals (`avg_reward_ema`, `good_streak`, `bad_streak`) in `Prompt only` mode, which contaminated conditioning adaptation for subsequent `Refine` runs. Rating and axis labels still update for repair continuity.

Fixed advisor rating label on first run or session reset: was forwarding the user's rating widget value even when there was no previous output to apply it to. Now passes `"No previous output (first run or session reset)"` when `has_previous_run` is false.

## [2.4.2] - 2026-05-15

### Added

Added `Learning` mode to `FunPack Video Refiner V2`. It still observes prompts, conditioning, ratings, phrase memory, and diagnostics, but passes positive and negative prompt conditioning through without prompt repair, Lucky composition, wildcard cleanup, or conditioning-vector adaptation.

### Fixed

Fixed `FunPack Scene Builder` mode handling so the live Mode widget stays independent from the selected saved scene, including queue-time `Learning` and `Auto` behavior.

Fixed `FunPack Scene Builder` rich prompt editing so the caret can move past a final inline phrase chip with the mouse or right arrow key.

## [2.4.1] - 2026-05-14

### Fixed

Improved `FunPack Scene Builder` database rows so long words and phrases show their full text as a hover hint, and double-click editing opens a wider multiline field with explicit OK/Cancel buttons.

## [2.4.0] - 2026-05-14

### Added

Added `FunPack Scene Builder`, a scene preset node that replaces `FunPack Template Manager`. It collects universal prompt phrase memory, lets users manually assign positive/negative scene phrases, passes the current LoRA stack through unchanged, and can auto-apply a saved scene from an intent prompt match.

Simplified `FunPack Scene Builder` so prompt and intent text are connection-only inputs, removed model-mode and per-block controls, and outputs only scene prompt data plus the pass-through FunPack LoRA stack instead of conditioning.

Added `Learning` mode to `FunPack Scene Builder`; it records connected prompt phrases into the selected refinement key's scene memory while passing positive prompt, negative prompt, and LoRA stack through unchanged. Refiner reset clears conditioning-delta learning while preserving the refinement key's Scene Builder memory.

Redesigned `FunPack Scene Builder` as a compact button-driven node with centered editor menus for scene name, mode, aliases, Positive Prompt, Negative Prompt, and Database controls. First use asks for a scene name before editing, connected prompts now teach useful words as well as phrase chunks, the editor refreshes the selected refinement-key database before opening, prompt editors highlight already-used chips, database words can be double-clicked for inline editing, and wildcard random choice is now a checkbox for adjacent entries instead of a text group.

Added searchable LoRA picking to `FunPack Apply LoRA Weights`. The compact row UI remains the primary workflow, and saved workflows still serialize through the existing `lora_list` JSON value.

Added optional `clip_vision_output`, `source_image`, and `negative_prompt` inputs to `FunPack Video Refiner V2`.

Added a final `modified_negative` conditioning output to `FunPack Video Refiner V2`. When negative repair has prompt text to encode and `CLIP` is connected, the node returns repaired negative conditioning; otherwise it returns an empty conditioning list.

Added advisory V2 vision context storage for source image dimensions, aspect ratio bucket, image fingerprint, CLIP Vision tensor summaries, and changed-image detection. Vision context is diagnostic only and is not blended into positive conditioning.

Added experimental early velocity bias capture/application controls to `FunPack Hybrid Euler 2S Sampler`, defaulting off.

### Changed

Removed public registration for `FunPack Template Manager`. Use `FunPack Scene Builder` for new scene/preset workflows.

Updated V2 prompt repair so repaired phrases preserve stopwords and phrase text while still using filtered semantic tokens for matching and categorization.

Reduced repeated Refiner V2 CLIP model calls by caching category and phrase encodes within each run.

Updated negative repair to persist poorly rated or wrong-context tags and append them to future negative prompts before encoding negative conditioning.

## [2.3.3] - 2026-05-08

### Fixed

Fixed Refiner V2 so `CLIP` and pre-encoded `positive_conditioning` can both be optional inputs. When `CLIP` is connected, V2 keeps owning prompt encoding as before. When `CLIP` is not connected but `positive_conditioning` is connected, V2 accepts the finished Gemma3/LTX2 conditioning, uses the prompt for analysis, and loads only the Gemma3 tokenizer.

## [2.3.2] - 2026-05-08

### Added

Added Refiner V2 original-intent alignment memory. When `user_intent_prompt` stays the same but an enhancer produces different `positive_prompt` variants, the refiner now remembers intent-enhance pairs, which variants rated well, which original-intent phrases were omitted, and which enhancer-only phrases were rejected.

### Fixed

Fixed Refiner V2 so learned original-intent omissions can be restored on later runs, while repeatedly rejected enhancer-only additions can be removed before encoding. Rejected enhancer-only full words and adjacent word pairs are stored as omit evidence for that original intent.

## [2.3.1] - 2026-05-08

### Fixed

Fixed Refiner V2 Prompt Repair so missing/wrong ratings only repair from the current prompt or explicit user intent, instead of pulling unrelated learned favorite actions, details, quality cues, camera moves, or styles from memory.

Fixed Prompt Repair memory matching so the same word with different neighboring prompt context is treated as different evidence.

Fixed vague raw user intent handling so prompts like `Figure it out` let the enhanced `positive_prompt` drive repair matching when available.

## [2.3.0] - 2026-05-08

### Added

Added `Wrong appearance` rating to `FunPack Video Refiner V2` for outputs contaminated by remembered clothing, character, subject, or background concepts.

Added `FunPack Refinement Key Loader`, with a selectable key dropdown, create-on-load behavior, and browser-side JSON import/export buttons.

Added a Discord-friendly Refiner V2 quick guide for new users.

### Changed

Updated Refiner V2 Prompt Repair so it only auto-adds safe repair concepts such as action, camera, details, quality, and style. Appearance, subject/character, and environment/background concepts are now blocked from Prompt Repair.

Updated `I'm Feeling Lucky` in Refiner V2 so appearance, subject/character, and environment/background memory is not auto-injected unless the user explicitly includes that phrase in the current prompt.

Updated legacy Void/Lucky token-bank selection to skip appearance, subject/character, and environment/background tokens.

Updated Refiner V2 and `FunPack Apply LoRA Weights` so both can accept a linked refinement key from `FunPack Refinement Key Loader`.

### Fixed

Fixed appearance bleed-over where highly liked clothing or character tags could reappear in unrelated image-to-video prompts.

## [2.2.1] - 2026-05-07

### Fixed

Fixed `FunPack Video Refiner V2` prompt phrase categorization so environment and appearance descriptions are not pulled into action learning by generic `-ing` or `-ed` words.

Updated Refiner V2 category similarity blending so CLIP category comparisons only help uncertain phrases instead of overriding strong local action, camera, appearance, environment, quality, or detail anchors.

Fixed `FunPack Video Refiner V2` so prompt-enhancer refusal text like "I'm sorry, I cannot help..." is passed through without being saved into prompt history, phrase memory, or future learning targets.

Improved `FunPack Video Refiner V2` training data output with clearer sections and extra line breaks for run state, learning, prompt analysis, adaptation, guidance, and LoRA diagnostics.

Updated `FunPack Video Refiner V2` to remember liked action/detail phrase clusters with their neighbors and use those ordered clusters before weaker ngram or token memory when repairing missing axes.

Added `Wrong details`, `Wrong action`, and `Wrong details + action` ratings for good-looking videos that do not match the requested intent; these preserve satisfied quality/composition signals while marking the mismatched action/detail context for repair.

## [2.2.0] - 2026-05-07

### Added

Added `FunPack Video Refiner V2`, a simplified prompt-owned refiner that accepts `positive_prompt` and a connected `CLIP`, owns prompt encoding internally, learns from ratings, and returns refined positive conditioning plus diagnostics.

Added `FunPack Template Manager`, a preset node for storing prompts, activation words, refinement keys, sigma schedules, and FunPack LoRA stacks with import/export support.

Added `I'm Feeling Lucky` mode to `FunPack Video Refiner V2`. It works as a preference composer that can inject learned user-preferred actions, camera moves, details, and styles even when the current prompt is vague.

### Changed

Updated LTX per-block LoRA loading so supported stacks now compare LoRA block fingerprints across the whole stack and apply type-aware conflict balancing before patches are loaded.

Fixed `FunPack Hybrid Euler 2S Sampler` restart timing so `restart_trigger_pct` is respected across the full sigma schedule instead of being clamped to the Euler-to-2S quality transition.

Improved `FunPack LoRA Loader` rerun performance by caching recently used raw LoRA files, model-mapped LoRA patches, and per-block fingerprint analysis.

Reworked `FunPack Video Refiner V2` ratings around explicit missing-axis signals: `Perfect`, single missing axes, paired missing axes, and `Awful`.

Removed the Refiner V2 `mode` input. V2 now accepts whatever connected `CLIP` the workflow provides and stores state in a CLIP-owned namespace.

Renamed visible Refiner and LoRA intent from `concept` to `action`. Old `Missing concept` ratings and old `concept` LoRA rows are still accepted as compatibility aliases, but V2 stores and displays `action`.

Updated `I'm Feeling Lucky` in Refiner V2 so Lucky only composes prompt text when enabled. When disabled, it may train memory from rated runs but does not compose or alter output.

### Fixed

Fixed `I'm Feeling Lucky` token-bank learning for changing prompt/conditioning workflows by falling back to prompt-order token placement when exact tokenizer position matching cannot find enough words.

Fixed `I'm Feeling Lucky` rating attribution for changing prompts so ratings update the previous prompt's learned tokens while the current prompt seeds new neutral discovery tokens.

Updated `I'm Feeling Lucky` filtering to learn poor adjacent token pairs instead of refusing individual tokens outright.

Updated `I'm Feeling Lucky` with uncapped token, pair, and context memory so it can learn which concepts belong together and call strong missing neighbors when prompt anchors are present.

Fixed `I'm Feeling Lucky` composition order so the current generation uses already-learned memory first, then seeds current prompt tokens for future runs.

Fixed `I'm Feeling Lucky` memory-first output so vague or empty incoming conditioning can use the longest compatible learned conditioning canvas instead of being limited to the current prompt's shape/content.

Added an optional `clip` input to `FunPack Video Refiner` so `I'm Feeling Lucky` can compose a learned prompt, re-encode it through the connected CLIP/Gemma text encoder, and refine from that freshly tokenized conditioning.

Improved `I'm Feeling Lucky` runtime by selecting learned conditioning canvases from saved tensor metadata before decoding, capping CLIP/Gemma Lucky prompts to a practical per-run concept count, and decoding only the token vectors selected for the current generation.

Reduced redundant `I'm Feeling Lucky` work by keeping Lucky runs in one stable memory history, skipping normal prompt-variant conditioning scans while Lucky is active, validating large Lucky memories once per loaded session, and updating context relationships locally instead of writing all-to-all token graphs every run.

Fixed `I'm Feeling Lucky` CLIP/Gemma re-encode crashes when the encoded Lucky prompt has a different sequence length than the incoming conditioning by resizing the refinement delta before applying it.

Updated `I'm Feeling Lucky` CLIP/Gemma prompt composition to preserve learned comma/semicolon-separated concept phrases instead of emitting loose word lists when phrase memory is available.

Added Lucky phrase placement memory so learned prompt phrases remember their rated order positions and CLIP/Gemma Lucky prompts can reassemble phrases into a more coherent prompt order instead of sentence salad.

Fixed `I'm Feeling Lucky` bootstrap learning so sessions that start with Lucky enabled now create a real discovery history entry, seed prompt tokens/phrases, and can learn from ratings without first running the classic refinement loop.

Updated Lucky memory so normal non-Lucky runs still seed reusable token, phrase, context, and placement memory for later Lucky runs.

Updated all missing-axis ratings so `Missing details`, `Missing concept`, `Missing quality`, and paired missing ratings now mark prompt tokens as wanted-but-underrepresented instead of weak neutral feedback; repeated missing feedback reserves Lucky composition room for those tokens and their compatible neighbours.

Fixed Lucky diagnostics so the collapsed Lucky memory stream reports real Lucky update counts and learned memory size instead of implying the session is still prompt 1 out of 1.

### Removed

Removed the old public `FunPack Video Refiner` node, the `FunPackGemmaEmbeddingRefiner` compatibility alias, and `FunPack Save Refinement Latent` from the registered node list.

Removed sigma refinement, latent refinement, manual scheduler controls, and feedback-question inputs from the active Refiner workflow. These systems are not part of Refiner V2.

## [2.1.3] - 2026-04-24

### Changed

`FunPack Apply LoRA Weights` now has more user-friendly, compact UI.

`FunPack Video Refiner` now has updated logic to work more stable when provided different prompts and conditioning with each new generation.

## [2.1.1] - 2026-04-24

### Added

Added `-Just forget it-` as a Video Refiner rating. Use it when a generation failed for reasons that should not be learned from, such as a broken reference, bad seed, or workflow mistake.

Added category feedback questions for prompt phrases that the refiner cannot confidently classify. The answer scale is `general`, `concept`, `style`, `quality`, `character`, and `details`.

Added a CLIP Vision output combiner node for workflows that need one combined CLIP Vision output from multiple inputs.

### Changed

Updated the Video Refiner rating categories so feedback can separate missing details, missing concept, missing quality, and fully failed output instead of treating all bad results the same way.

Reduced repeated category feedback prompts after the user has already answered enough about the same concept.

Refreshed README and refiner docs for 2.1.1.

### Fixed

Fixed LoRA weight row restore order when workflows are loaded.

## [2.1.0] - 2026-04-23

### Added

Added `FunPack Apply LoRA Weights` and `FunPack LoRA Loader`, a prompt-exact LoRA weight workflow designed to work with `FunPack Video Refiner`.

Added `FunPack Save Refinement Latent`, which stores latent tensor bundles by refinement key for optional latent refinement in `FunPack Video Refiner`.

Added hidden LTX per-block LoRA redistribution for supported `ltx2` model stacks. The UI still exposes normal LoRA weights, while the loader derives per-block strengths from the LoRA patch magnitudes when the model and LoRA layout support it.

The new workflow uses base LoRA weights on the first run for a prompt, then lets the refiner save prompt-specific suggested LoRA weights into its existing JSON state for later runs.

### Changed

Renamed the visible refiner title from `FunPack Gemma Embedding Refiner` to `FunPack Video Refiner`. The old node key is still available as a compatibility alias.

Split the old single `funpack.py` implementation into focused modules:

- `conditioning.py`
- `samplers.py`
- `image_processing.py`
- `model_management.py`

`funpack.py` remains as a compatibility re-export for older imports.

Updated `FunPack Video Refiner` so it can accept a FunPack LoRA stack and save next-run model LoRA weight suggestions based on prompt concepts, LoRA type hints, and user ratings.

Updated `FunPack Video Refiner` with optional latent input/output refinement. If no matching saved latent exists and both latent input and output are connected, the input latent is saved as the first reference and passed through unchanged.

Updated prompt analysis so quoted speech and backslash-wrapped phrases can be protected as whole prompt units.

### Documentation

Documented unintended and edge-case usage for the new refiner workflow, including disconnected latent paths, saved-latent-only runs, wrong LTX audio/AV latent connections, exact-prompt LoRA lookup behavior, base-weight mismatch behavior, zero-weight LoRA skipping, and unsupported per-block fallback behavior.

## [1.3.3] - 2026-04-22

### Changed

Expanded `/docs` so every node in `funpack.py` now has matching documentation, and refreshed the existing node docs to match the current inputs and outputs.

## [1.3.2] - 2026-04-19

### Changed

Changed the core logic of Self-Refiner.
Removed obsolete nodes.

## [1.3.0 & 1.3.1] - 2026-04-18

### Changed

Added new nodes - User Rating and Gemma Self-Refinement for LTX2.3 video workflows.

### Fixed

Device type mismatch in new nodes.

## [1.2.3] - 2026-01-30

### Fixed

Fixed Transformers library error when running Prompt Enhancer and Story Writer nodes.

## [1.2.2] - 2026-01-26

### Changed

Changed the logic of processing sequences in Story Writer node. Now doesn't append full instructions and previous context to previous messages with each loop iteration, now fully replaces messages with a system prompt and sequence history without appending.

## [1.2.1] - 2026-01-24

### Added

Added experimental LoRA recommendation feature and Sanity Check features to Story Writer node.

## [1.2.0] - 2026-01-23

### Added

Added new Story Writer node, based on existing Prompt Enhancer. It generates up to 5 prompts one after another, based on either user's prompt directly, or on the story generated from the user's prompt.

## [1.1.0] - 2026-01-02

### Added

Added Creative Template and Lorebook Enhancer nodes. The Creative Template is a wildcard-based node that replaces given keywords in the template with ones provided by user. Lorebook Enhancer is a node that takes SillyTavern format .json lorebooks and enhances your prompt by adding required knowledge.

## [1.0.0] - 2026-01-01

Initial release on Comfy Registry.
