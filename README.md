# GroundedRAG — Hybrid Search RAG with Citation Verification

An enterprise-style document Q&A system whose defining feature is that it
**checks whether its own answers are actually supported by the retrieved
documents** — claim by claim, with an independent NLI model — instead of
trusting the LLM's own citations.

Full pipeline: PDF/TXT/MD/DOCX ingestion → structure-aware chunking → dense
(FAISS) + BM25 hybrid retrieval via Reciprocal Rank Fusion → CrossEncoder
reranking → context-constrained LLM generation → claim extraction → NLI-based
citation verification → groundedness scoring → self-correcting regeneration
on unsupported claims. See [`docs/architecture.md`](docs/architecture.md)
for the full design rationale, and [`docs/security.md`](docs/security.md)
for what's hardened versus what's a known, documented gap.

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | Matches `backend/Dockerfile`'s base image |
| Node.js | 20 | Matches `frontend/Dockerfile`'s base image |
| PostgreSQL | 16 | Or run it via Docker Compose (recommended) |
| Docker + Docker Compose | Any recent version | Only needed for the Docker path below |

An LLM API key (Anthropic by default) for real answer generation — or set
`LLM_PROVIDER=extractive` to run with no LLM at all (see
[LLM provider options](#llm-provider-options) below).

## Project layout

```
grounded-rag/
├── backend/            FastAPI app — ingestion, retrieval, generation, verification
│   ├── app/
│   │   ├── api/v1/       REST endpoints (documents, chat, health)
│   │   ├── core/         config, database, logging
│   │   ├── models/       SQLAlchemy ORM models
│   │   ├── schemas/      Pydantic request/response models
│   │   ├── services/     the actual pipeline (ingestion/chunking/embeddings/
│   │   │                 retrieval/reranking/generation/verification/rag)
│   │   └── evaluation/   metrics.py (used by evaluation/evaluate.py)
│   ├── tests/            185 test functions
│   └── requirements.txt
├── frontend/            React + Vite + Tailwind UI (chat, citations, document manager)
├── evaluation/           Sample corpus, QA dataset, evaluate.py
├── docs/                 architecture.md, security.md
├── docker-compose.yml     postgres + backend + frontend
└── .env.example
```

## Environment variables

Copy the template and fill in what you need:

```bash
cp .env.example backend/.env
```

Every setting has a working default in `app/core/config.py` — you only need
to edit `LLM_API_KEY` to get started. Full reference:

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg2://grounded_rag:grounded_rag@localhost:5432/grounded_rag` | Or a `sqlite:///./data/grounded_rag.db` URL for zero-setup local testing |
| `LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` (any OpenAI-compatible endpoint) \| `extractive` (no LLM) |
| `LLM_API_KEY` | *(empty)* | Required unless `LLM_PROVIDER=extractive` |
| `LLM_MODEL` | `claude-sonnet-5` | Change to match your provider |
| `LLM_BASE_URL` | *(empty)* | Only for `openai` against a non-OpenAI endpoint, e.g. a local Ollama server |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `180` / `30` | Chunking budget (see architecture doc for why 180, not the more common 400-500) |
| `DENSE_TOP_K` / `BM25_TOP_K` | `20` / `20` | Candidates pulled from each retriever before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANK_TOP_M` | `5` | Final chunks handed to the LLM as context |
| `MIN_RERANK_SCORE` | `0.0` (disabled) | >0 skips generation entirely if even the best chunk isn't relevant enough |
| `SUPPORT_THRESHOLD` / `PARTIAL_THRESHOLD` / `CONTRADICTION_THRESHOLD` | `0.70` / `0.30` / `0.50` | NLI verdict thresholds |
| `MAX_REGENERATION_RETRIES` | `2` | Self-correction attempts before returning a safe fallback |

See `.env.example` for the complete list with inline comments.

## Installation

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`sentence-transformers` and `torch` are large installs (a few GB); the first
`pip install` will take a while. Models themselves (the embedder, reranker,
and NLI model) download on first use, not at install time.

### Frontend

```bash
cd frontend
npm install
```

## Database / vector-store setup

**Option A — Docker (recommended, handles Postgres for you):**

```bash
docker compose up -d postgres
```

**Option B — your own Postgres:** create a database and user matching
`DATABASE_URL`, or edit `DATABASE_URL` to point at an existing instance.

**Option C — zero-setup SQLite:** set
`DATABASE_URL=sqlite:///./data/grounded_rag.db` in `backend/.env`. Fine for
trying the project out; not recommended beyond that (no concurrent-write
safety).

FAISS and BM25 indexes are plain files under `data/index/` — created
automatically on first use, no separate setup needed.

## Running locally

**Backend:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```
API docs at `http://localhost:8000/docs`. Health check:
`http://localhost:8000/api/v1/health`.

**Frontend** (separate terminal):
```bash
cd frontend
npm run dev
```
UI at `http://localhost:5173`.

## Running with Docker Compose (all three services)

```bash
cp .env.example backend/.env   # edit LLM_API_KEY first
docker compose up --build
```

Backend on `:8000`, frontend on `:5173`, Postgres on `:5432`. The frontend
container waits for the backend's healthcheck before starting; the backend
waits for Postgres. First build is slow (torch + faiss-cpu); subsequent
builds are cached.

## Running tests

```bash
cd backend
pytest -v
```

Tests use an in-memory SQLite database and a temp upload directory — no
Postgres, FAISS, or real models required for the vast majority of them;
retrieval/reranking/generation/verification logic is tested against
injectable fake embedders, LLM calls, and NLI models so the orchestration
is verified without downloading anything.

## Uploading documents

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/api/v1/documents/upload
```
(or use the **Documents** tab in the UI). This parses and chunks the file —
check the returned `status`. Once it's `"chunking"`, embed and index it:

```bash
curl -X POST http://localhost:8000/api/v1/documents/{id}/index
```
(or click **Index** in the UI). `status` becomes `"indexed"` once both the
dense and BM25 indexes are built — only then is the document searchable.

## Running a query

```bash
curl -X POST http://localhost:8000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Does Product X support PostgreSQL?"}'
```

The response includes the answer with inline `[n]` citation markers, each
claim's independently-verified verdict (`SUPPORTED` /
`PARTIALLY_SUPPORTED` / `UNSUPPORTED`) with its supporting quote and score,
a groundedness summary, and the full retrieval trace (dense/BM25/hybrid/
rerank ranks per chunk) for the debug panel.

## How citation verification works

Short version: every factual claim in the generated answer is checked with
an NLI (Natural Language Inference) model against the specific passage the
LLM cited for it — not against the LLM's own self-reported confidence. The
model predicts whether the evidence *entails*, *contradicts*, or is
*neutral* toward the claim; an explicit contradiction overrides a middling
entailment score. If any claim comes back unsupported, the system
regenerates the answer with feedback about exactly what failed, up to twice,
before falling back to a safe "the documents don't provide enough evidence"
response rather than returning something ungrounded. Full detail, including
the verdict thresholds and the worked example from this project's own spec,
is in [`docs/architecture.md`](docs/architecture.md#5-citation-verification--the-core-differentiator).

## Evaluation

```bash
cd backend
python ../evaluation/evaluate.py
```

Compares four pipeline variants (dense-only, hybrid, hybrid+reranked,
full verified RAG) against a 22-question sample dataset, using an isolated
database/index so it never touches your real data. See
[`evaluation/README.md`](evaluation/README.md) for details — and note its
numbers are *measured*, never pre-filled: this script has not been run in
whatever environment built this repository, by design (see that README for
why).

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `uvicorn` fails to start with a DB connection error | Postgres isn't running, or `DATABASE_URL` doesn't match it. Run `docker compose up -d postgres` or check your local Postgres credentials. |
| Upload succeeds but `status` stays `"failed"` | Check `error_message` in the response — usually a scanned/image-only PDF (OCR isn't supported) or a corrupted file. |
| `POST /documents/{id}/index` returns 409 | The document has no chunks (chunking must have failed — check `error_message` on the document). |
| `POST /documents/{id}/index` returns 500 | Check the backend logs — commonly a missing/invalid `LLM_API_KEY` isn't the cause here (indexing doesn't call the LLM); more likely an embedding-model download failure (no network, or a very first run still downloading `sentence-transformers` weights). |
| Query always returns `INSUFFICIENT_EVIDENCE` | No documents are indexed yet (`status` must be `"indexed"`, not just `"chunking"`), or `MIN_RERANK_SCORE` is set above 0 and nothing retrieved scores high enough — try lowering it or check your documents actually cover the question. |
| `LLM_PROVIDER=anthropic` requests fail with 401 | `LLM_API_KEY` is missing or invalid in `backend/.env`. |
| First query is very slow | The embedding, reranking, and NLI models all lazy-load on first use (not at server startup, unless `PRELOAD_MODELS=true`) — the first request after a cold start pays that cost once. |
| `npm run dev` can't reach the backend | Check `VITE_API_BASE_URL` in `frontend/.env` (copy from `frontend/.env.example`) matches where the backend is actually running. |
| Docker: frontend container never goes healthy / times out waiting on backend | Check `docker compose logs backend` — usually a missing `backend/.env` (the `env_file` directive requires the file to exist) or an invalid `LLM_API_KEY` blocking startup checks if `PRELOAD_MODELS=true`. |
| `pytest` fails with import errors | Make sure you're running it from `backend/` (so `pytest.ini`'s `pythonpath = .` applies) and that `pip install -r requirements.txt` completed fully. |

## Future improvements

Not implemented, and not pretended to be — see
[`docs/security.md`](docs/security.md#known-deliberate-gaps) for the full
list (authentication/multi-tenancy, rate limiting, DOCX decompression
limits) plus Experiments 5 and 6 in `evaluation/evaluate.py`'s docstring
(chunk-size and top-K sweeps) as natural next steps once baseline numbers
exist from a real run.
#   G r o u n d e d R A G  
 