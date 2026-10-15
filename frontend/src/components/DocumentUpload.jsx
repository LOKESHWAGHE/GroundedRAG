import React, { useRef, useState } from "react";
import { extractErrorMessage, uploadDocument } from "../services/api.js";

export default function DocumentUpload({ onUploaded }) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function handleFiles(files) {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of files) {
        await uploadDocument(file);
      }
      onUploaded();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragOver ? "border-indigo-400 bg-indigo-50" : "border-slate-300 bg-white"
        }`}
      >
        <p className="text-sm text-slate-500">
          {uploading ? "Uploading…" : "Drop a PDF, TXT, MD, or DOCX file here, or click to browse"}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,.md,.docx"
          multiple
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
      {error && <p className="mt-2 text-sm text-rose-600">{error}</p>}
    </div>
  );
}
