# Interview prep & CV bullets

## CV bullets

Use these only after you've actually run `evaluation/evaluate.py` and have
real numbers — the bracketed placeholders below are exactly that: **never
copy a bracketed placeholder into a resume as if it were a measured
result.**

- Built GroundedRAG, a hybrid-search RAG system with independent citation
  verification: combines dense (FAISS) and BM25 retrieval via Reciprocal
  Rank Fusion, CrossEncoder reranking, and NLI-based claim verification
  that checks every generated claim against its cited evidence rather than
  trusting the LLM's self-reported citations.
- Implemented BM25 from scratch (Okapi formula, IDF, length normalization)
  rather than using a library, to keep the retrieval scoring fully
  inspectable for a debug panel showing per-query dense/BM25/hybrid/rerank
  ranks.
- Designed a self-correcting generation loop: claims that fail NLI
  verification trigger automatic regeneration with targeted feedback (up
  to N retries) before falling back to a safe "insufficient evidence"
  response — never returning an unverified answer.
- Built an evaluation harness comparing four pipeline variants (dense-only,
  hybrid, hybrid+reranked, fully-verified) on Recall@K, MRR, citation
  accuracy, and groundedness against an isolated test corpus; achieved
  [recall@5 of X.XX / groundedness of Y.YY%] *(fill in only after running
  `evaluate.py` yourself)*.
- Full-stack: FastAPI + PostgreSQL + SQLAlchemy backend, React + Vite
  frontend with a citation-verification UI (clickable evidence, support
  scores, SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED badges) and a retrieval
  debug panel; 185 backend test functions; Dockerized.

## Interview Q&A

**Why RAG instead of fine-tuning?**
Fine-tuning bakes knowledge into weights — it's expensive to update, can't
cite its sources, and risks hallucinating confidently on anything outside
the fine-tuning set. RAG keeps the knowledge external and retrievable, so
updating a document is just re-indexing a file, and — critically for this
project — the answer can be traced back to and checked against a specific
source, which a fine-tuned model's output can't be.

**Why BM25 alongside dense retrieval?**
Dense embeddings are good at semantic similarity but can blur together
exact identifiers — product codes, error numbers, rare proper nouns —
because those tokens don't carry much weight in embedding space. BM25
scores exact term overlap directly, so it catches queries like "what does
error code E-4021 mean" that a purely semantic retriever can miss.

**Why not just use dense/vector search alone?**
Because it systematically underperforms on exact-match and rare-term
queries (see above) — hybrid retrieval exists specifically to cover that
gap, and the evaluation harness's dense-vs-hybrid comparison is designed to
make that gap measurable rather than assumed.

**Why Reciprocal Rank Fusion instead of combining raw scores?**
Cosine similarity (roughly 0–1) and a raw BM25 score (unbounded,
query-dependent) aren't on comparable scales, so a weighted average like
`0.6·dense + 0.4·bm25` would let whichever retriever happens to produce
larger numbers dominate, arbitrarily. RRF sidesteps that entirely by fusing
*rankings* — `score(d) = Σ 1/(k + rank_i(d))` — which only needs each
retriever's ordering, not its score's scale.

**Why CrossEncoder reranking on top of hybrid retrieval?**
A bi-encoder (used for dense retrieval) embeds the query and each chunk
*independently*, then compares vectors — fast, so it can run over the
whole corpus, but it can't let the query and chunk tokens directly attend
to each other. A CrossEncoder processes the (query, chunk) pair *together*,
which is more accurate but too slow to run over everything — so hybrid
retrieval does the cheap first pass, and the CrossEncoder only reranks the
~20 survivors.

**How do you detect hallucinations?**
Every factual claim extracted from the answer is checked with an NLI model
against the specific chunk the model cited for it: the NLI model predicts
entailment/contradiction/neutral probabilities between the evidence and the
claim. A claim with no real support (low entailment) is UNSUPPORTED; one
the evidence actively contradicts is UNSUPPORTED regardless of entailment
score; this is independent of the LLM's own stated confidence or citation.

**How do you calculate groundedness?**
`(supported_claims + 0.5 × partially_supported_claims) / total_claims` —
partial credit for partially-supported claims, since a response that's
mostly-SUPPORTED-with-some-PARTIAL is meaningfully better than one with the
same count fully UNSUPPORTED, which a stricter supported-only ratio can't
distinguish.

**What happens when evidence is insufficient?**
The system prompt instructs the model to respond with an explicit
`INSUFFICIENT_EVIDENCE:` sentinel rather than guessing, which the parser
detects deterministically. Separately, if claims DO get generated but keep
failing verification after the regeneration budget is exhausted, the
pipeline falls back to the same kind of response — the system is designed
to prefer admitting it doesn't know over returning something ungrounded.

**How do you evaluate RAG quality?**
Retrieval quality via Recall@K and MRR against a keyword-defined ground
truth (any indexed chunk containing the expected fact's keywords counts as
relevant — this avoids hardcoding chunk IDs that would break on every
re-index). Answer quality via citation accuracy (fraction of claims that
come back SUPPORTED) and groundedness. All four pipeline stages (dense,
hybrid, hybrid+reranked, fully verified) are compared head-to-head on the
same question set.

**What is Recall@K? What is MRR?**
Recall@K: of all the chunks that are actually relevant to a question, what
fraction appear somewhere in the top K retrieved? Precision@K: of the top K
retrieved, what fraction are actually relevant? MRR (Mean Reciprocal Rank):
average of `1/rank` of the first relevant result across all queries —
rewards getting a relevant result near the top, not just somewhere in the
list.

**How does chunk size affect retrieval?**
Too large, and a chunk mixes multiple topics, diluting its embedding and
making it a weaker match for any single question; it also risks silent
truncation by the embedding model's own input limit (this project's default
of 180 words, not the more common 400–500, is chosen specifically because
`all-MiniLM-L6-v2` truncates at ~190 words). Too small, and a chunk loses
context needed to make sense on its own, and you multiply the number of
vectors to search. Overlap exists to keep a fact that lands near a chunk
boundary retrievable from whichever side the query embedding lands closer
to.

**How would you reduce latency?**
Cache embeddings for repeated/similar queries; batch embedding calls during
ingestion (already done); reduce `RERANK_TOP_M`/`HYBRID_TOP_N` if quality
allows it (tune via the evaluation harness, not by guessing); run the
embedding/reranker/NLI models on GPU; consider a smaller/distilled
CrossEncoder and NLI model if the accuracy trade-off is acceptable; cache
the LLM response for identical questions against an unchanged corpus.

**How would you scale this to millions of documents?**
FAISS's flat index (exact search) stops being fast enough — switch to an
approximate index (HNSW or IVF+PQ) at the cost of some recall, tune via the
evaluation harness's Recall@K to find an acceptable trade-off. BM25's
from-scratch implementation here recomputes corpus statistics on every
change, which is fine at this project's scale but would need an
incremental-update design (or a dedicated search engine like Elasticsearch/
OpenSearch) at that scale. Move from a single Postgres instance to
read replicas or a managed service.

**How would you replace FAISS with a distributed vector database?**
The `VectorIndex` class in `faiss_index.py` is a thin wrapper with
`add`/`remove`/`search`/`save`/`load` — swapping it for a client to
Pinecone/Weaviate/Qdrant/Milvus means reimplementing that same interface
against the new backend's API; nothing elsewhere in the codebase imports
FAISS directly (dense.py only talks to the `VectorIndex` interface), so the
blast radius of that change is contained to one file plus its tests.

**How would you handle concurrent users?**
FastAPI with Uvicorn already handles concurrent requests; the main
contention points would be: (1) the single in-process FAISS/BM25 index —
either needs per-process locking (already present via `threading.RLock`)
or moving to a shared service if scaled across multiple backend processes;
(2) the database, where PostgreSQL handles concurrent connections natively
(unlike the SQLite option, which is explicitly documented as single-writer
and not recommended beyond local testing).

**How would you cache embeddings?**
Chunk embeddings are already computed once and persisted in FAISS, not
recomputed per query. For query-side caching: a hash of the normalized
question text as a cache key against its embedding, with an LRU or
TTL-based cache — most valuable when the same or near-duplicate questions
repeat often.

**How would you handle document updates?**
Delete-and-reingest is the current model (`DELETE /documents/{id}` removes
it from both indexes via content hash dedup on upload, so re-uploading an
updated version is a clean replace). A true "patch" workflow would need to
diff the new content against the old chunks and only re-embed/re-index the
changed ones — not implemented, since dedup-by-hash already makes the
simple path correct, if not maximally efficient for large documents with
small edits.

**How would you secure uploaded documents?**
See [`docs/security.md`](security.md) for the full, honest answer —
including what's actually implemented (filename sanitization, magic-byte
validation, size caps, no secrets in git, generic error responses) and what
isn't (auth, rate limiting, a DOCX decompression-bomb guard).
