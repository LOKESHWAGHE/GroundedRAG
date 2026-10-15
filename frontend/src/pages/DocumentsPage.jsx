import React, { useCallback, useEffect, useState } from "react";
import DocumentList from "../components/DocumentList.jsx";
import DocumentUpload from "../components/DocumentUpload.jsx";
import { extractErrorMessage, listDocuments } from "../services/api.js";

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setError(null);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className="mx-auto max-w-3xl px-4 py-6">
      <h1 className="text-lg font-semibold text-slate-800">Documents</h1>
      <p className="mt-1 text-sm text-slate-500">
        Upload PDF, TXT, Markdown, or Word documents. After a document finishes chunking,
        click <span className="font-medium">Index</span> to embed it and make it searchable.
      </p>

      <div className="mt-4">
        <DocumentUpload onUploaded={refresh} />
      </div>

      {error && <p className="mt-4 text-sm text-rose-600">{error}</p>}
      {loading ? (
        <p className="mt-6 text-sm text-slate-400">Loading…</p>
      ) : (
        <DocumentList documents={documents} onChanged={refresh} />
      )}
    </div>
  );
}
