# GroundedRAG — Architecture

## 1. Why this exists

A plain "embed chunks → cosine similarity → stuff into prompt → ask LLM"
RAG pipeline has two structural weaknesses:

1. **Retrieval blind spots.** Dense embeddings are good at *semantic*
   similarity but often miss exact identifiers — product codes, names,
   dates, section numbers, API error codes — because those tokens don't
   carry much semantic weight in embedding space. BM25 catches exactly
   this case, which is why this project runs both and fuses them.
2. **No answer accountability.** Even with perfect retrieval, an LLM can
   still misstate, over-generalize, or blend what the context says. Most
   RAG demos never check this — they just print whatever citations the
   LLM claims to have used, which is trusting the same model to grade its
   own homework.

GroundedRAG's pipeline is built around fixing both: hybrid retrieval for
#1, and a dedicated claim-extraction + NLI-based verification layer for #2
that never trusts the generator's own citations.

## 2. End-to-end data flow

```
                      React Frontend (Vite)
                            │ REST (JSON)
                      FastAPI (app/api/v1)
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                     │
   Ingestion            Query pipeline        Evaluation
   (upload → parse       (below)              (evaluate.py,
    → chunk → index)                           isolated DB/index)
        │                   │
        ▼                   ▼
┌─────────────────┐   Query pipeline, in order:
│ PostgreSQL        │
│ (documents,        │  1. Dense retrieval (FAISS, top DENSE_TOP_K)
│  pages, chunks,     │  2. BM25 retrieval (from-scratch Okapi BM25, top BM25_TOP_K)
│  queries)           │  3. Hybrid fusion (Reciprocal Rank Fusion)
│                    │  4. CrossEncoder rerank → top RERANK_TOP_M
│ FAISS index         │     (short-circuits to INSUFFICIENT_EVIDENCE here if the
│ (dense vectors,      │      best score is below MIN_RERANK_SCORE)
│  IndexIDMap2+        │  5. LLM generation (context-constrained, inline [n] citations)
│  IndexFlatIP)        │  6. Claim extraction (sentence-level, or LLM-decomposed atomic claims)
│                    │  7. Citation verification (NLI entailment/contradiction per claim)
│ BM25 index           │  8. Groundedness scoring
│ (pickled corpus,     │  9. If unsupported claims remain → regenerate with feedback
│  recomputed stats)   │     (up to MAX_REGENERATION_RETRIES)
└─────────────────┘   10. Return answer + verified citations + groundedness + retrieval trace
```

## 3. Technology choices and why

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | Async, Pydantic-validated request/response models, free OpenAPI docs — the API contract is self-documenting. |
| Dense retrieval | sentence-transformers + FAISS | FAISS runs fully local with no paid vector-DB service; sentence-transformers gives direct control over the embedding model/dimension rather than hiding it behind a managed API. |
| Keyword retrieval | **From-scratch BM25** (`bm25_index.py`), not a library | The scoring formula (term frequency, IDF, length normalization) stays fully inspectable — important both for the retrieval debug panel and for being able to actually explain the math in an interview, not just name-drop "BM25". |
| Fusion | Reciprocal Rank Fusion | RRF combines *rankings*, not raw scores, so it doesn't require cosine similarity (~[0,1]) and an unbounded BM25 score to live on the same scale — a safer default than ad hoc weighted averaging. |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Scores the (query, chunk) pair jointly instead of comparing independently-embedded vectors — more accurate for final selection, at a cost that's affordable once hybrid retrieval has already narrowed the candidate set. |
| Claim verification | CrossEncoder NLI (`nli-deberta-v3-small`) | Entailment/contradiction/neutral probabilities between evidence and claim is a direct, well-studied way to answer "does this text actually support this statement" — more principled than asking the generator LLM to grade its own output. |
| Metadata store | PostgreSQL | Relational integrity across documents → pages → chunks → queries, and queryable directly from the UI/evaluation harness without touching the vector index. |
| LLM access | Provider-agnostic (`llm_client.py`) | `anthropic` / any OpenAI-compatible `/chat/completions` endpoint / `extractive` (no LLM, zero-cost fallback for local dev and the evaluation harness's baseline row) — verification logic never imports a vendor SDK directly. |
| Frontend | React + Vite + Tailwind, plain JS | No build-step overhead beyond Vite; Tailwind keeps the UI's citation badges, debug panel, and status indicators consistent without hand-rolled CSS. |
| Testing | pytest, 185 test functions | Every phase shipped with tests for its own logic, written against injectable functions (embedder, LLM, NLI model, vector/BM25 index) so the orchestration logic is verifiable without downloading any model or calling any API. |

Deliberately **not** used: LangChain's retrieval abstractions (the
retrieval/fusion/reranking/verification math is written out explicitly, per
this project's own "don't hide the logic" requirement), and a managed/paid
vector database (FAISS is enough at this scale and keeps the project
runnable with zero paid services beyond the LLM API itself).

## 4. Chunking strategy

Structure-aware by default (`CHUNK_STRATEGY=structure`): splits on paragraph
boundaries, tracks markdown headings as `section_title` context carried
across chunks, packs paragraphs up to `CHUNK_SIZE_TOKENS` words (180 by
default — `all-MiniLM-L6-v2` truncates input at ~190 words, so a larger
default would be silently cut at embedding time), splits oversized
paragraphs on sentence boundaries, and overlaps consecutive chunks by
`CHUNK_OVERLAP_TOKENS` words so a fact split across a chunk boundary is
still retrievable from either side.

A `fixed` strategy (plain sliding word window, ignoring structure) is kept
specifically as an evaluation baseline — Experiment 5 in `evaluate.py`
compares the two.

## 5. Citation verification — the core differentiator

For every claim extracted from the generated answer:

1. **Evidence gathering**: the chunk(s) the model cited for that claim's
   sentence, or — if it cited nothing — every chunk in context (a missing
   citation doesn't get a free pass).
2. **Windowing**: each evidence chunk is split into sentence-level windows
   (capped at `NLI_MAX_WINDOWS_PER_CHUNK`) so one supporting sentence isn't
   diluted by unrelated sentences sitting in the same chunk.
3. **NLI scoring**: every (window, claim) pair gets entailment/
   contradiction/neutral probabilities from the NLI model.
4. **Verdict**: an explicit contradiction (≥ `CONTRADICTION_THRESHOLD`)
   overrides a middling entailment score; otherwise entailment ≥
   `SUPPORT_THRESHOLD` → SUPPORTED, ≥ `PARTIAL_THRESHOLD` →
   PARTIALLY_SUPPORTED, otherwise UNSUPPORTED.

Verified directly against the worked example in this project's own spec:
*"Product X supports PostgreSQL"* against *"...supports MySQL and
PostgreSQL"* → SUPPORTED; *"...supports MongoDB"* against the same
evidence → UNSUPPORTED (`backend/tests/test_citation_verification.py`).

**Groundedness** = `(supported + 0.5 × partially_supported) / total_claims`
— partial credit for partially-supported claims, since a response with a
few PARTIALLY_SUPPORTED claims is meaningfully better than one with the
same count fully UNSUPPORTED, and a strict supported-only ratio can't tell
them apart. Verified against the spec's own worked example (8 claims, 7
supported, 1 partial → 93.75%).

**Self-correction**: if any claim comes back UNSUPPORTED, the pipeline
regenerates the answer with explicit feedback about which specific claims
failed, up to `MAX_REGENERATION_RETRIES` times. If still ungrounded after
that, it returns a safe `INSUFFICIENT_EVIDENCE` response — never a shaky,
unverified answer.

## 6. Database design

```
documents            — filename, file_type, sha256 (dedup), status, page/char/chunk counts
document_pages        — cleaned per-page text (FK -> documents)
chunks                — text, section_title, page_number, token_count (FK -> documents)
                        id doubles as the FAISS vector id AND the BM25 corpus key
queries               — question, status, groundedness, full result JSON, latency
```

Dense vectors live in FAISS and BM25 postings in a pickled corpus — both
keyed by `chunks.id` — not in Postgres; Postgres is the source of truth for
everything relational and queryable from the UI.

## 7. API surface

```
POST   /api/v1/documents/upload          Upload + parse + chunk (not yet embedded)
GET    /api/v1/documents                 List documents
GET    /api/v1/documents/{id}            Get one document
GET    /api/v1/documents/{id}/pages      Debug: cleaned per-page text
GET    /api/v1/documents/{id}/chunks     Debug: how the document was chunked
POST   /api/v1/documents/{id}/index      Embed + add to FAISS/BM25
DELETE /api/v1/documents/{id}            Delete (and remove from both indexes)

POST   /api/v1/chat/query                Ask a question — runs the full pipeline
GET    /api/v1/chat/history               Past queries
GET    /api/v1/chat/{id}                  One past query's saved answer/claims

GET    /api/v1/health                     Liveness check
```

Indexing is a separate step from upload deliberately: embedding is
comparatively slow, and you'll want to re-run it (e.g. after changing
`EMBEDDING_MODEL_NAME`) without re-uploading every file.

## 8. Configuration philosophy

Every tunable — chunk size/overlap, `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`,
`RERANK_TOP_M`, `MIN_RERANK_SCORE`, the NLI thresholds,
`MAX_REGENERATION_RETRIES` — lives in `app/core/config.py`, sourced from
environment variables (`backend/.env`). This is what lets `evaluate.py`
sweep these values (chunk size, top-K) without touching pipeline code, and
it's also what caught two real bugs during development: `MIN_RERANK_SCORE`
and `MAX_QUESTION_CHARS` were both declared but not actually wired in at
first — found during the Phase 14/15 testing and security review, not
shipped silently broken.
