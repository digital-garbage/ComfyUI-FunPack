# FunPack Video Refiner

See [`FunPackGemmaEmbeddingRefiner.md`](FunPackGemmaEmbeddingRefiner.md) for the full documentation. The old filename is kept as a compatibility documentation path for users and workflows that still know the node by its former name.

Current reference behavior: original conditioning is kept for prompt/conditioning change detection, `Perfect` switches active refinement to the liked generated conditioning average, and `Awful` can roll back to the latest better-rated conditioning before boosting missing details, concept, and quality support. Experimental `into_the_void` mode can also mix a few learned liked token embeddings into the final conditioning for preference discovery. Experimental `I'm Feeling Lucky` mode builds a random conditioning field from better-rated learned token embeddings and uses saved prompt order as a loose composition guide when available.
