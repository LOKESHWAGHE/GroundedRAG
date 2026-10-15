import React, { useState } from "react";

// The "Retrieval Details" expandable section from the spec: for every query,
// shows each retrieved chunk's dense rank/score, BM25 rank/score, hybrid
// (RRF) rank, and reranked (CrossEncoder) rank/score side by side, plus
// whether it made it into the final LLM context after the word budget.
export default function RetrievalDebugPanel({ rows }) {
  const [open, setOpen] = useState(false);
  if (!rows || rows.length === 0) return null;

  const sorted = [...rows].sort((a, b) => (a.hybrid_rank ?? 999) - (b.hybrid_rank ?? 999));

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium text-slate-600 hover:bg-slate-50"
      >
        <span>Retrieval Details ({rows.length} chunks retrieved)</span>
        <span className="text-slate-400">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="scroll-thin overflow-x-auto border-t border-slate-100 px-1 py-2">
          <table className="w-full min-w-[640px] text-left text-xs">
            <thead>
              <tr className="text-slate-400">
                <th className="px-2 py-1">Chunk</th>
                <th className="px-2 py-1">Dense rank</th>
                <th className="px-2 py-1">Dense score</th>
                <th className="px-2 py-1">BM25 rank</th>
                <th className="px-2 py-1">BM25 score</th>
                <th className="px-2 py-1">Hybrid (RRF) rank</th>
                <th className="px-2 py-1">Rerank rank</th>
                <th className="px-2 py-1">Rerank score</th>
                <th className="px-2 py-1">In context</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((row) => (
                <tr
                  key={row.chunk_id}
                  className={`border-t border-slate-100 ${row.in_final_context ? "bg-emerald-50/50" : ""}`}
                >
                  <td className="px-2 py-1 font-mono text-slate-500">
                    #{row.chunk_id}
                    <div className="text-slate-400">
                      {row.filename} p.{row.page_number}
                    </div>
                  </td>
                  <td className="px-2 py-1">{row.dense_rank ?? "—"}</td>
                  <td className="px-2 py-1">{row.dense_score != null ? row.dense_score.toFixed(3) : "—"}</td>
                  <td className="px-2 py-1">{row.bm25_rank ?? "—"}</td>
                  <td className="px-2 py-1">{row.bm25_score != null ? row.bm25_score.toFixed(3) : "—"}</td>
                  <td className="px-2 py-1 font-medium">{row.hybrid_rank ?? "—"}</td>
                  <td className="px-2 py-1">{row.rerank_rank ?? "—"}</td>
                  <td className="px-2 py-1">{row.rerank_score != null ? row.rerank_score.toFixed(3) : "—"}</td>
                  <td className="px-2 py-1">{row.in_final_context ? "✓" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
