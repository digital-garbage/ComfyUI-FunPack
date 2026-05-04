# FunPack Video Refiner

See [`FunPackGemmaEmbeddingRefiner.md`](FunPackGemmaEmbeddingRefiner.md) for the full documentation. The old filename is kept as a compatibility documentation path for users and workflows that still know the node by its former name.

Current reference behavior: original conditioning is kept for prompt/conditioning change detection, `Perfect` switches active refinement to the liked generated conditioning average, and `Awful` can roll back to the latest better-rated conditioning before boosting missing details, concept, and quality support.
