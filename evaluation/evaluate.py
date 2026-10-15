#!/usr/bin/env python3
"""
Evaluation harness for GroundedRAG.

Ingests the sample documents in dataset/sample_docs/, runs every question in
dataset/qa_pairs.json through four pipeline variants, and writes a report to
results/. Every number in that report is computed live by this script —
nothing here is a hardcoded or sample value. If you have not run this
script, there are no evaluation numbers for this project; don't claim any.

Run from the backend/ directory (so it picks up backend/.env) with:

    cd backend && python ../evaluation/evaluate.py

This uses an ISOLATED SQLite database and a separate FAISS/BM25 index under
evaluation/results/eval_data/ (set via environment variables before any app
module is imported), so running this script never touches your real
uploaded documents, your production Postgres database, or your production
vector/BM25 indexes.

Requires a working LLM_PROVIDER (set LLM_API_KEY in backend/.env, or set
LLM_PROVIDER=extractive to run the retrieval-only metrics without any LLM
calls -- generation/claim/groundedness metrics will be degenerate in that
mode since the extractive provider doesn't produce real prose answers).
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ISOLATED_DATA_DIR = RESULTS_DIR / "eval_data"
ISOLATED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# MUST be set before any `app.*` import: config.py and database.py read these
# at import time. This is what makes the evaluation run isolated from your
# real data.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{ISOLATED_DATA_DIR / 'eval.db'}")
os.environ.setdefault("DATA_DIR", str(ISOLATED_DATA_DIR))

from app.core.database import Base, SessionLocal, engine  # noqa: E402
from app.evaluation.metrics import (  # noqa: E402
    citation_accuracy,
    mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    stddev,
    unsupported_claim_rate,
)
from app.models.document import Chunk, Document  # noqa: E402
from app.services.ingestion.service import ingest_document  # noqa: E402
from app.services.rag.pipeline import run_query_pipeline  # noqa: E402
from app.services.reranking.service import rerank  # noqa: E402
from app.services.retrieval.dense import dense_search  # noqa: E402
from app.services.retrieval.hybrid import hybrid_search  # noqa: E402
from app.services.retrieval.document_indexer import index_document  # noqa: E402

SAMPLE_DOCS_DIR = EVAL_DIR / "dataset" / "sample_docs"
QA_PATH = EVAL_DIR / "dataset" / "qa_pairs.json"
TOP_K = 5
CANDIDATE_K = 20  # size of the ranked list MRR is computed over, and the hybrid pool reranking draws from


def setup_corpus(db):
    """Ingests + indexes every sample document. Safe to re-run: ingest_document dedupes by content hash."""
    print("Ingesting sample documents...")
    for path in sorted(SAMPLE_DOCS_DIR.glob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf", ".docx"):
            continue
        try:
            doc = ingest_document(db, path.name, path.read_bytes())
        except Exception as exc:  # DuplicateDocumentError on a second run -- fine, just skip indexing
            print(f"  = {path.name}: already ingested ({exc})")
            continue
        if doc.status == "failed":
            print(f"  ! {path.name}: parsing/chunking failed -- {doc.error_message}")
            continue
        index_document(db, doc)
        print(f"  + {path.name}: {doc.chunk_count} chunks, indexed")


def make_chunk_meta_fn(db):
    def chunk_meta_fn(chunk_ids):
        if not chunk_ids:
            return {}
        rows = (
            db.query(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Chunk.id.in_(chunk_ids))
            .all()
        )
        return {
            c.id: {
                "document_id": str(c.document_id), "filename": fn, "page_number": c.page_number,
                "section_title": c.section_title, "text": c.text,
            }
            for c, fn in rows
        }

    return chunk_meta_fn


def relevant_chunk_ids(db, keywords: list[str]) -> set[int]:
    """Ground truth: any indexed chunk whose text contains one of the given keyword phrases (case-insensitive)."""
    if not keywords:
        return set()
    all_chunks = db.query(Chunk.id, Chunk.text).all()
    lowered = [k.lower() for k in keywords]
    return {cid for cid, text in all_chunks if any(k in text.lower() for k in lowered)}


def evaluate_retrieval_variants(db, qa_pairs: list[dict]) -> dict:
    """Experiment 1-3: Dense-only vs Hybrid (RRF) vs Hybrid+CrossEncoder, Recall@5 / Precision@5 / MRR."""
    chunk_meta_fn = make_chunk_meta_fn(db)
    variants = {"dense_only": [], "hybrid": [], "hybrid_reranked": []}
    mrr_lists = {"dense_only": [], "hybrid": [], "hybrid_reranked": []}
    relevant_sets = []

    answerable = [qa for qa in qa_pairs if qa.get("relevant_keywords")]
    for qa in answerable:
        relevant = relevant_chunk_ids(db, qa["relevant_keywords"])
        relevant_sets.append(relevant)
        if not relevant:
            print(f"  ! WARNING: no chunk matched the keywords for {qa['id']} -- check qa_pairs.json")
            continue

        dense_ids = [r.chunk_id for r in dense_search(qa["question"], top_k=CANDIDATE_K)]
        hybrid_results = hybrid_search(qa["question"], hybrid_top_n=CANDIDATE_K)
        hybrid_ids = [r.chunk_id for r in hybrid_results]

        meta = chunk_meta_fn(hybrid_ids)
        chunk_texts = {cid: m["text"] for cid, m in meta.items()}
        reranked_ids = [r.chunk_id for r in rerank(qa["question"], hybrid_results, chunk_texts, top_m=CANDIDATE_K)]

        for name, ids in [("dense_only", dense_ids), ("hybrid", hybrid_ids), ("hybrid_reranked", reranked_ids)]:
            variants[name].append(
                {"recall": recall_at_k(ids, relevant, TOP_K), "precision": precision_at_k(ids, relevant, TOP_K)}
            )
            mrr_lists[name].append(ids)

    report = {}
    for name in variants:
        report[name] = {
            "recall_at_5_mean": mean([v["recall"] for v in variants[name]]),
            "recall_at_5_stddev": stddev([v["recall"] for v in variants[name]]),
            "precision_at_5_mean": mean([v["precision"] for v in variants[name]]),
            "mrr": mean_reciprocal_rank(mrr_lists[name], relevant_sets),
            "n_questions": len(variants[name]),
        }
    return report


def evaluate_verified_rag(db, qa_pairs: list[dict]) -> dict:
    """Experiment 4: full pipeline with claim extraction + citation verification + groundedness + self-correction."""
    chunk_meta_fn = make_chunk_meta_fn(db)
    answerable = [qa for qa in qa_pairs if qa.get("relevant_keywords")]
    unanswerable = [qa for qa in qa_pairs if not qa.get("relevant_keywords")]

    groundedness_scores, verdicts_per_query, latencies, regen_counts = [], [], [], []
    for qa in answerable:
        start = time.perf_counter()
        result = run_query_pipeline(qa["question"], chunk_meta_fn=chunk_meta_fn)
        latencies.append((time.perf_counter() - start) * 1000)
        regen_counts.append(result.regeneration_count)
        if result.groundedness.groundedness_score is not None:
            groundedness_scores.append(result.groundedness.groundedness_score)
        verdicts_per_query.append([v.verdict.value for v in result.claims])

    insufficient_evidence_correct = 0
    for qa in unanswerable:
        result = run_query_pipeline(qa["question"], chunk_meta_fn=chunk_meta_fn)
        if result.insufficient_evidence:
            insufficient_evidence_correct += 1

    return {
        "citation_accuracy": citation_accuracy(verdicts_per_query),
        "unsupported_claim_rate": unsupported_claim_rate(verdicts_per_query),
        "groundedness_mean": mean(groundedness_scores),
        "groundedness_stddev": stddev(groundedness_scores),
        "avg_latency_ms": mean(latencies),
        "avg_regeneration_count": mean([float(c) for c in regen_counts]),
        "n_answerable_questions": len(answerable),
        "insufficient_evidence_detection": f"{insufficient_evidence_correct}/{len(unanswerable)}",
    }


def print_report(retrieval_report: dict, verified_report: dict) -> None:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)

    print("\n" + "=" * 60)
    print("Baseline Vector RAG (dense retrieval only)")
    r = retrieval_report["dense_only"]
    print(f"  Recall@5:    {fmt(r['recall_at_5_mean'])} (stddev {fmt(r['recall_at_5_stddev'])})")
    print(f"  Precision@5: {fmt(r['precision_at_5_mean'])}")
    print(f"  MRR:         {fmt(r['mrr'])}")

    print("\nHybrid RAG (dense + BM25, Reciprocal Rank Fusion)")
    r = retrieval_report["hybrid"]
    print(f"  Recall@5:    {fmt(r['recall_at_5_mean'])} (stddev {fmt(r['recall_at_5_stddev'])})")
    print(f"  Precision@5: {fmt(r['precision_at_5_mean'])}")
    print(f"  MRR:         {fmt(r['mrr'])}")

    print("\nHybrid + CrossEncoder Reranking")
    r = retrieval_report["hybrid_reranked"]
    print(f"  Recall@5:    {fmt(r['recall_at_5_mean'])} (stddev {fmt(r['recall_at_5_stddev'])})")
    print(f"  Precision@5: {fmt(r['precision_at_5_mean'])}")
    print(f"  MRR:         {fmt(r['mrr'])}")

    print("\nVerified RAG (full pipeline: generation + claim extraction + citation verification)")
    print(f"  Citation accuracy:         {fmt(verified_report['citation_accuracy'])}")
    print(f"  Unsupported claim rate:    {fmt(verified_report['unsupported_claim_rate'])}")
    print(f"  Groundedness (mean):       {fmt(verified_report['groundedness_mean'])} "
          f"(stddev {fmt(verified_report['groundedness_stddev'])})")
    print(f"  Avg regeneration attempts: {fmt(verified_report['avg_regeneration_count'])}")
    print(f"  Avg latency:               {fmt(verified_report['avg_latency_ms'])} ms")
    print(f"  Insufficient-evidence detection: {verified_report['insufficient_evidence_detection']} correctly flagged")
    print("=" * 60)
    print("\nThese are measured values from this run against the sample documents in")
    print("dataset/sample_docs/ -- re-run this script any time the pipeline, config,")
    print("or dataset changes. Do not copy these numbers into the README without")
    print("re-running this script yourself first.\n")


def main():
    qa_pairs = json.loads(QA_PATH.read_text())
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)
        setup_corpus(db)

        print("\nRunning retrieval variant comparison (Experiments 1-3)...")
        retrieval_report = evaluate_retrieval_variants(db, qa_pairs)

        print("Running full pipeline with citation verification (Experiment 4)...")
        verified_report = evaluate_verified_rag(db, qa_pairs)
    finally:
        db.close()

    print_report(retrieval_report, verified_report)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {"timestamp": timestamp, "retrieval": retrieval_report, "verified_rag": verified_report}, indent=2
        )
    )
    print(f"Full report written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
