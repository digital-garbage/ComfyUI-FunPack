# ComfyUI-FunPack v5 — "Cunning Cranberry"

FunPack v4.0.0 is going to be the latest release of classic FunPack.
Next update is another milestone update, or shall I say, a full rework.

Maintenance of current FunPack became a mess — forgetting to add or remove buttons from/to UI, finding functions that don't do anything but still present in the code absolutely randomly, trying to keep the UI neat while it has no strict boundaries on how it should look like, and of course — making it easy for users to understand how FunPack operates and how you can operate it — this is all time, money and token-consuming.

General idea behind the FunPack v5 rework is writing it from scratch, with a new architecture where a feature can't be forgotten, where a wire can't be missed, where one failing module never stops whole process from executing. Loaders and features would tell the UI by themselves what sorts of inputs they need and where to put it — and UI will compose them according to design guidelines it would have. Samplers would become independent from features — if a feature is available and enabled, sampler would call it. If it's not — it gets skipped and doesn't even appear in the UI so it won't confuse you.

Most of this comes from a FunUI backend draft that got shelved after understanding on how much time has to be invested into polishing it. Without a community support and wide usage it would become another dead project only one person in the world needs but no one else.

Come to think of it, FunPack already became something alike, with an exception — this project is more than alive, and while I can, I will support it.

Final goal is to make FunPack an ultimate answer for a novice's question "I don't know how to generate — what do I install and how do I start?" and for a professional's question "I need a UI with all the cutting edge techniques available on day 0 — which one?"

Preparation for the v5 had started.

**FunPack v5.0 "Cunning Cranberry" — coming fall 2026.**

---

This branch is a clean start. Classic FunPack lives on [`main`](https://github.com/digital-garbage/ComfyUI-FunPack/tree/main).
