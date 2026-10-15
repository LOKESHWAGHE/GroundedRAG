import React from "react";

// Document processing status badge: uploaded -> parsing -> parsed -> chunking -> embedding -> indexed -> failed
const STYLES = {
  uploaded: "bg-slate-100 text-slate-700",
  parsing: "bg-sky-100 text-sky-700",
  parsed: "bg-sky-100 text-sky-700",
  chunking: "bg-indigo-100 text-indigo-700",
  embedding: "bg-indigo-100 text-indigo-700",
  indexed: "bg-emerald-100 text-emerald-700",
  failed: "bg-rose-100 text-rose-700",
};

export default function StatusBadge({ status }) {
  const classes = STYLES[status] || STYLES.uploaded;
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ${classes}`}>
      {status}
    </span>
  );
}
