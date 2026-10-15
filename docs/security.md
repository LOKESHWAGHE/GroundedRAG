# Security notes

What's actually implemented, and what's a known, deliberate gap — written
honestly so this holds up under an interview question like "how would you
secure uploaded documents?"

## Implemented

- **Filenames are never trusted.** `sanitize_filename` strips directory
  components and unsafe characters; the file is stored on disk under a
  server-generated UUID, never the client's filename. A `../../etc/passwd`
  upload cannot write outside the upload directory.
- **Extension allow-list + magic-byte check.** A file must have an allowed
  extension (`.pdf`, `.txt`, `.md`, `.docx`) *and* its first bytes must
  match that format (e.g. `%PDF-` for PDF, the ZIP signature for DOCX). An
  `.exe` renamed to `.pdf` is rejected by the second check even though it
  passes the first.
- **Size capped while reading**, not after: `read_upload_limited` reads at
  most `MAX_UPLOAD_SIZE_MB + 1` bytes, so an oversized upload can't exhaust
  memory before being rejected.
- **No secrets in source control.** `LLM_API_KEY` and `DATABASE_URL` are
  read from environment variables / `backend/.env`; `.gitignore` excludes
  `.env` and `backend/.env`. `.env.example` ships with a placeholder key,
  never a real one.
- **SQL injection:** not applicable — every query goes through SQLAlchemy's
  ORM/query builder; there is no raw string-interpolated SQL anywhere in
  the codebase.
- **Server-side path never exposed.** `DocumentOut` (the API response
  schema) has no `stored_path` field — the on-disk location of a file is
  never returned to a client, even though it's stored in the database.
- **Unhandled exceptions never leak internals.** A global exception handler
  (`app/main.py`) catches anything that isn't already a deliberate
  `HTTPException`, logs the full exception server-side, and returns only a
  generic `"An internal error occurred."` to the client — never a raw
  traceback, file path, or library-specific error string (e.g. a database
  driver's connection-string error).
- **Config is the single source of truth for limits**, read from one place
  (`app/core/config.py`) rather than duplicated as magic numbers in
  individual request schemas — e.g. `QueryRequest.question`'s max length
  comes from `settings.MAX_QUESTION_CHARS`, not a hardcoded `1000`.

## Known, deliberate gaps

These are out of scope for this project but worth naming explicitly rather
than pretending they're handled:

- **No authentication or multi-tenancy.** Every document and every query is
  visible to every API client. This is a single-user/demo-scope project;
  adding auth (API keys, JWT, or session-based) would be the first thing
  to build before any real deployment.
- **No rate limiting.** A client can call `/chat/query` as fast as the LLM
  provider allows. Adding `slowapi` (or a reverse-proxy-level limiter) is
  a natural next step, not implemented here to keep the dependency list
  minimal.
- **DOCX parsing and zip-based decompression risk.** A DOCX file is a ZIP
  container; `python-docx` (via `lxml`) decompresses it without an
  explicit size cap, so a maliciously crafted "zip bomb" disguised as a
  `.docx` could consume excessive memory/CPU during parsing. Mitigating
  this would mean wrapping the ZIP read with a decompressed-size limit
  before handing it to `python-docx` — not implemented here.
- **Prompt injection from document content.** Retrieved chunk text is
  inserted directly into the LLM prompt. A document containing text like
  "ignore your instructions and..." is passed to the model as ordinary
  context. The system prompt constrains the model to answer only from
  context and cite every claim, which narrows (but doesn't eliminate) this
  risk — citation verification (Phase 10) is the real backstop: even if
  the model is influenced by injected text, a resulting claim still has to
  pass NLI verification against real evidence to be reported as supported.
- **CORS defaults to `localhost:5173`** for local development.
  `CORS_ORIGINS` must be set explicitly to your real frontend origin(s)
  before any non-local deployment.
