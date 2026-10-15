import React, { useState } from "react";
import StatusBadge from "./StatusBadge.jsx";
import { deleteDocument, extractErrorMessage, indexDocument } from "../services/api.js";

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentList({ documents, onChanged }) {
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState(null);

  async function handleIndex(id) {
    setBusyId(id);
    setError(null);
    try {
      await indexDocument(id);
      onChanged();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(id) {
    setBusyId(id);
    setError(null);
    try {
      await deleteDocument(id);
      onChanged();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setBusyId(null);
    }
  }

  if (documents.length === 0) {
    return <p className="mt-6 text-sm text-slate-400">No documents uploaded yet.</p>;
  }

  return (
    <div className="mt-6 overflow-hidden rounded-xl border border-slate-200 bg-white">
      {error && <p className="border-b border-rose-100 bg-rose-50 px-4 py-2 text-sm text-rose-600">{error}</p>}
      <table className="w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="px-4 py-2">Document</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2">Pages</th>
            <th className="px-4 py-2">Chunks</th>
            <th className="px-4 py-2">Size</th>
            <th className="px-4 py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id} className="border-t border-slate-100">
              <td className="px-4 py-2 font-medium text-slate-700">{doc.filename}</td>
              <td className="px-4 py-2">
                <StatusBadge status={doc.status} />
                {doc.status === "failed" && doc.error_message && (
                  <p className="mt-1 max-w-xs text-xs text-rose-500">{doc.error_message}</p>
                )}
              </td>
              <td className="px-4 py-2 text-slate-500">{doc.page_count}</td>
              <td className="px-4 py-2 text-slate-500">{doc.chunk_count}</td>
              <td className="px-4 py-2 text-slate-500">{formatSize(doc.size_bytes)}</td>
              <td className="px-4 py-2 text-right">
                <div className="flex justify-end gap-2">
                  {doc.status === "chunking" && (
                    <button
                      type="button"
                      disabled={busyId === doc.id}
                      onClick={() => handleIndex(doc.id)}
                      className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700 hover:bg-indigo-100 disabled:opacity-50"
                    >
                      {busyId === doc.id ? "Indexing…" : "Index"}
                    </button>
                  )}
                  <button
                    type="button"
                    disabled={busyId === doc.id}
                    onClick={() => handleDelete(doc.id)}
                    className="rounded-full bg-rose-50 px-3 py-1 text-xs font-medium text-rose-700 hover:bg-rose-100 disabled:opacity-50"
                  >
                    Delete
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
