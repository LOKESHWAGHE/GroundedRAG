import React from "react";
import CitationList from "./CitationList.jsx";
import GroundednessSummary from "./GroundednessSummary.jsx";
import RetrievalDebugPanel from "./RetrievalDebugPanel.jsx";

function AnswerText({ text }) {
  // Lightly render inline [n] citation markers as superscript-style badges
  // rather than leaving them as plain "[1]" text in the middle of a sentence.
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="whitespace-pre-wrap leading-relaxed text-slate-800">
      {parts.map((part, i) =>
        /^\[\d+\]$/.test(part) ? (
          <sup key={i} className="mx-0.5 font-semibold text-indigo-600">
            {part}
          </sup>
        ) : (
          <React.Fragment key={i}>{part}</React.Fragment>
        )
      )}
    </p>
  );
}

export default function ChatMessage({ message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] rounded-2xl rounded-br-sm bg-indigo-600 px-4 py-2 text-white">
          {message.text}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="flex justify-start">
        <div className="max-w-[75%] rounded-2xl rounded-bl-sm border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">
          {message.text}
        </div>
      </div>
    );
  }

  // assistant
  const { result } = message;
  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <AnswerText text={result.answer} />
        {result.insufficient_evidence && (
          <p className="mt-2 text-xs font-medium text-amber-600">
            The documents don't provide enough verified evidence to fully answer this.
          </p>
        )}
        {result.regeneration_count > 0 && (
          <p className="mt-1 text-xs text-slate-400">
            Answer was regenerated {result.regeneration_count}{" "}
            {result.regeneration_count === 1 ? "time" : "times"} after citation verification.
          </p>
        )}
        <GroundednessSummary groundedness={result.groundedness} />
        <CitationList claims={result.claims} />
        <RetrievalDebugPanel rows={result.retrieval_debug} />
      </div>
    </div>
  );
}
