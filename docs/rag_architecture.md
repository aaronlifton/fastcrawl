# Fastcrawl RAG Architecture Plan

## Goals

1. Quantify retrieval quality before handing chunks to an LLM by replaying curated question/answer sets against `public.wiki_chunks`.
2. Keep embeddings fresh by diffing manifests and enqueueing only the chunks whose checksums changed since the last sweep.
3. Offer a thin HTTP retrieval API that wraps pgvector searches and ships structured context windows straight into the LLM prompt template.
4. Capture prompt templates that match the wiki chunk schema (URL, section_path, token_estimate, text) so experiments stay reproducible.

## Components

### Retrieval Evaluation Harness

- Binary: `fastcrawl-eval` (new CLI under `src/bin/vector_eval.rs`).
- Inputs: (a) evaluation JSONL containing `query`, `relevant_urls`, optional `notes`; (b) Postgres connection options identical to `fastcrawl-pgvector`; (c) OpenAI embedding configuration reused from the embedder.
- Flow: load eval cases → embed each query → fetch dense candidates (pgvector) + lexical candidates (Postgres `tsvector`) → fuse with Reciprocal Rank Fusion → score Recall@k / MRR / hit-rate vs. known relevant URLs.
- Output: table + JSON summary (precision-ish metrics plus list of misses) persisted to `--report-json`.
- Extensibility: plugging other embedding endpoints works automatically because we reuse `OpenAiEmbedder` traits.

### Embedding Freshness Job

- Binary: `fastcrawl-freshness` (new CLI under `src/bin/freshness.rs`).
- Inputs: manifests emitted by normalization (`data/wiki_manifest.jsonl`), optional previous manifest snapshot, and an embedding output dir.
- Flow: load manifests → compare current vs. prior checksums → emit `refresh_plan.jsonl` describing URLs needing re-embedding (new, changed, deleted) → optionally invoke `fastcrawl-embedder` via `--exec` hook.
- Side-effects: updates a sqlite-ish ledger (`data/embedding_ledger.jsonl`) recording plan decisions for observability.
- Purpose: scheduler/cron can call this job hourly to keep embeddings from drifting.

### Retrieval API Service

- Binary: `fastcrawl-retriever` (new CLI under `src/bin/retriever_api.rs`).
- Framework: `axum` + `tokio` (already in the tree) + `tokio_postgres` for DB pool.
- Hybrid stack:
  - Dense similarity from pgvector (OpenAI embeddings).
  - Lexical hits from Postgres full-text search backed by a generated `text_tsv` column + GIN index (created by `fastcrawl-pgvector` or the standalone `fastcrawl-fts-indexer` job).
  - Reciprocal Rank Fusion joins both streams, then optional token budgeting trims the context.
- Endpoints:
  - `POST /v1/query`: body `{ "query": "...", "top_k": 5 }` → returns JSON chunks annotated with `dense_distance`, `lexical_score`, `fused_score`, and rank metadata plus checksum/provenance.
  - `GET /healthz`: confirms DB + embedder readiness.
- Config: CLI flags for DB URL, pg schema/table, OpenAI credentials, candidate counts for dense/lexical streams, RRF constant, max chunk tokens per response, embedding-cache size, and rate limits (requests/minute + burst). Set cache or rate knobs to zero to disable protections when benchmarking.
- Usage: internal service used by eval harness and LLM prompt builder; does not expose OpenAI key to clients.

### Prompt Template Catalog

- Location: `personal_docs/prompt_templates/wiki_rag.md`.
- Contents: base system prompt, retrieval formatting instructions that spell out `section_path` breadcrumbs, and two prompt variants (few-shot Q&A and pure RAG answerer) referencing the new API payload.
- Maintains compatibility with later orchestration (LangChain, LlamaIndex) by prescribing JSON format for retrieved chunks.

## Data Flow Summary

Crawler → Normalizer (`NormalizedPage` JSONL) → Manifest (checksums) → Embedder (OpenAI) → `fastcrawl-pgvector` table `public.wiki_chunks` → Eval harness + Retrieval API. The freshness job watches manifests to decide when to kick the embedder and reload Postgres, while prompt templates and the API form the user-facing RAG surface.
