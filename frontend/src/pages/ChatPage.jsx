import React, { useEffect, useRef, useState } from "react";
import ChatMessage from "../components/ChatMessage.jsx";
import { extractErrorMessage, runQuery } from "../services/api.js";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const result = await runQuery(question);
      setMessages((prev) => [...prev, { role: "assistant", result }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "error", text: extractErrorMessage(err) }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="scroll-thin flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto flex max-w-3xl flex-col gap-4">
          {messages.length === 0 && (
            <div className="rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-400">
              Ask a question about your uploaded documents. Every answer is checked, claim by
              claim, against the retrieved evidence before it's shown to you.
            </div>
          )}
          {messages.map((m, i) => (
            <ChatMessage key={i} message={m} />
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-400 shadow-sm">
                Retrieving, generating, and verifying the answer…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <form onSubmit={handleSubmit} className="border-t border-slate-200 bg-white px-4 py-3">
        <div className="mx-auto flex max-w-3xl gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about your documents..."
            disabled={loading}
            className="flex-1 rounded-full border border-slate-300 px-4 py-2 text-sm focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400 disabled:bg-slate-50"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="rounded-full bg-indigo-600 px-5 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            Ask
          </button>
        </div>
      </form>
    </div>
  );
}
