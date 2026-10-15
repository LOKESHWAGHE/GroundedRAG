import React from "react";

// Explicitly labels these as evaluation metrics, not a guarantee of factual
// correctness -- matching the backend's own documentation of the metric.
export default function GroundednessSummary({ groundedness }) {
  if (!groundedness || groundedness.total_claims === 0) {
    return null;
  }
  const pct = Math.round((groundedness.groundedness_score ?? 0) * 100);
  const coveragePct = Math.round((groundedness.citation_coverage ?? 0) * 100);

  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
      <span className="font-medium text-slate-700">Groundedness: {pct}%</span>
      <span>Citation coverage: {coveragePct}%</span>
      <span>
        {groundedness.supported} supported · {groundedness.partially_supported} partial ·{" "}
        {groundedness.unsupported} unsupported
      </span>
      <span className="italic">Evaluation metric, not a guarantee of factual correctness.</span>
    </div>
  );
}
