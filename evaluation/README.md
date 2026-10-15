# Evaluation

Measures retrieval and groundedness quality against a small sample corpus —
not a claim about performance on your real documents, which will differ.

## What's here

- `dataset/sample_docs/` — 4 short synthetic documents (an employee handbook,
  a product spec, an API integration guide, an incident report) with
  concrete, checkable facts (dates, numbers, exact terms).
- `dataset/qa_pairs.json` — 22 question/answer pairs against those documents.
  21 are answerable; 1 (`q22`) is deliberately unanswerable from the sample
  docs, to check that the pipeline correctly returns `INSUFFICIENT_EVIDENCE`
  instead of guessing.
- `evaluate.py` — ingests the sample docs into an **isolated** SQLite
  database and FAISS/BM25 index (never your real data), runs every question
  through four pipeline variants, and writes a timestamped JSON report to
  `results/`.
- `metrics.py` (in `backend/app/evaluation/`) — Recall@K, Precision@K, MRR,
  citation accuracy, unsupported-claim rate; unit-tested in
  `backend/tests/test_evaluation_metrics.py`.

## Running it

```bash
cd backend
# make sure backend/.env has a working LLM_PROVIDER + LLM_API_KEY,
# or set LLM_PROVIDER=extractive to skip real LLM calls (see note below)
python ../evaluation/evaluate.py
```

This prints a report to stdout and writes the full numbers to
`evaluation/results/eval_<timestamp>.json`.

## The four variants compared

1. **Baseline Vector RAG** — dense retrieval only (Recall@5, Precision@5, MRR)
2. **Hybrid RAG** — dense + BM25 fused with RRF
3. **Hybrid + CrossEncoder Reranking** — adds the reranking stage
4. **Verified RAG** — the full pipeline: generation + claim extraction +
   citation verification + groundedness + self-correction (citation
   accuracy, unsupported-claim rate, groundedness, regeneration count,
   latency, and insufficient-evidence detection on `q22`)

## A note on `LLM_PROVIDER=extractive`

Variants 1–3 only exercise retrieval and don't call the LLM at all, so they
work with any provider setting. Variant 4 calls the LLM for generation and
(in `CLAIM_EXTRACTION_MODE=llm`) for claim decomposition — with
`LLM_PROVIDER=extractive` those calls return a trivial placeholder answer,
so the citation-accuracy/groundedness numbers in that mode aren't
meaningful. Use a real provider for variant 4 to get real numbers.

## Important: numbers are never fabricated here

This script has been written and syntax-checked but **has not been run** in
the environment that built this repository — it depends on FAISS, torch,
sentence-transformers, and a working LLM API key, none of which were
available there. The first time you run it is the first time any number in
this report exists. Nowhere in this repository's README or docs are sample
or illustrative numbers presented as measured results — if you see a metric
quoted anywhere, it came from an actual run of this script.

## Extending it

- **Experiment 5 (chunk size)**: change `CHUNK_SIZE_TOKENS` in `.env`,
  delete `evaluation/results/eval_data/`, and re-run — the isolated index
  rebuilds from scratch.
- **Experiment 6 (top-K)**: change `DENSE_TOP_K` / `BM25_TOP_K` /
  `RERANK_TOP_M` and re-run; no index rebuild needed, since retrieval counts
  are read at query time.
- Add more `qa_pairs.json` entries (and matching sample docs) for a larger,
  more representative sample before drawing conclusions from the numbers.
