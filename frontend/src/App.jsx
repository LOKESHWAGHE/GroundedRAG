import React, { useState } from "react";
import ChatPage from "./pages/ChatPage.jsx";
import DocumentsPage from "./pages/DocumentsPage.jsx";

const TABS = [
  { id: "chat", label: "Chat" },
  { id: "documents", label: "Documents" },
];

export default function App() {
  const [tab, setTab] = useState("chat");

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-slate-200 bg-white px-4 py-3">
        <div>
          <h1 className="text-base font-semibold text-slate-800">GroundedRAG</h1>
          <p className="text-xs text-slate-400">Hybrid search RAG with citation verification</p>
        </div>
        <nav className="flex gap-1 rounded-full bg-slate-100 p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
                tab === t.id ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-h-0 flex-1">
        {tab === "chat" ? <ChatPage /> : <DocumentsPage />}
      </main>
    </div>
  );
}
