import axios from "axios";

// Falls back to localhost:8000 so `npm run dev` works out of the box without
// a .env file, matching the backend's default port in docker-compose.yml.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000/api/v1";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // generation + verification can genuinely take a while on a cold model load
});

// Document endpoints
export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await client.post("/documents/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function listDocuments() {
  const { data } = await client.get("/documents");
  return data;
}

export async function getDocumentChunks(documentId) {
  const { data } = await client.get(`/documents/${documentId}/chunks`);
  return data;
}

export async function indexDocument(documentId) {
  const { data } = await client.post(`/documents/${documentId}/index`);
  return data;
}

export async function deleteDocument(documentId) {
  await client.delete(`/documents/${documentId}`);
}

// Chat / query endpoints
export async function runQuery(question) {
  const { data } = await client.post("/chat/query", { question });
  return data;
}

export async function getQueryHistory() {
  const { data } = await client.get("/chat/history");
  return data;
}

// Shared error-message extraction, since FastAPI's error shape varies
// (plain string detail vs. the {message, existing_document_id} shape
// the upload endpoint uses for 409s).
export function extractErrorMessage(error) {
  const detail = error?.response?.data?.detail;
  if (!detail) return error?.message || "Something went wrong.";
  if (typeof detail === "string") return detail;
  if (typeof detail === "object" && detail.message) return detail.message;
  return JSON.stringify(detail);
}

export default client;
