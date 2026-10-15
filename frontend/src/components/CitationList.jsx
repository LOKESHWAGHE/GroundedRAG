import React, { useState } from "react";
import VerdictBadge from "./VerdictBadge.jsx";

// Renders the numbered citation list under an answer, e.g.
//   [1] Company Handbook — Page 12  ✓ Supported
// Clicking a citation expands the supporting evidence: document, page,
// chunk text, support score, and which claim it backs.
export default function CitationList({ claims }) {
  const [openId, setOpenId] = useState(null);

  const citedClaims = claims.filter((c) => c.evidence);
  if (citedClaims.length === 0) return null;

  return (
    <div className="mt-3 space-y-1.5">
      {citedClaims.map((claim, i) => {
        const isOpen = openId === claim.claim_id;
        return (
          <div key={claim.claim_id} className="rounded-lg border border-slate-200 bg-white">
            <button
              type="button"
              onClick={() => setOpenId(isOpen ? null : claim.claim_id)}
              className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm hover:bg-slate-50"
            >
              <span className="flex min-w-0 items-center gap-2">
                <span className="shrink-0 font-mono text-slate-400">[{i + 1}]</span>
                <span className="truncate text-slate-700">
                  {claim.evidence.filename} — Page {claim.evidence.page_number}
                </span>
              </span>
              <span className="flex shrink-0 items-center gap-2">
                <VerdictBadge verdict={claim.verdict} />
                <span className="text-slate-400">{isOpen ? "−" : "+"}</span>
              </span>
            </button>
            {isOpen && (
              <div className="border-t border-slate-100 px-3 py-2 text-sm text-slate-600">
                <p className="mb-1 text-slate-500">
                  Claim: <span className="italic">&ldquo;{claim.text}&rdquo;</span>
                </p>
                <p className="mb-1">
                  Supporting text: <span className="bg-amber-50">{claim.evidence.quote}</span>
                </p>
                <p className="text-xs text-slate-400">
                  Support score: {claim.support_score.toFixed(2)} · Document ID: {claim.evidence.document_id}
                </p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
