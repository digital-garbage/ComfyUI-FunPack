# FunPack Video Refiner

See [`FunPackGemmaEmbeddingRefiner.md`](FunPackGemmaEmbeddingRefiner.md) for the full documentation. The old filename is kept as a compatibility documentation path for users and workflows that still know the node by its former name.

Current reference behavior: original conditioning is kept for prompt/conditioning change detection, `I like it` / `9-10` switches active refinement to the liked generated conditioning average, and `I don't like it` rolls back to the latest better-rated conditioning before pushing away.
