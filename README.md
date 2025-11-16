# Fastcrawl

Fastcrawl is a polite, configurable web-crawler focused on continuous streaming extraction. It ships with a minimal
Wikipedia example (`examples/wiki.rs`) that demonstrates how to plug custom link filters and crawl controls into the
core runtime.

Current fastest speed, with default controls of `max-depth` 4, `max-links-per-page` 16, `politeness-ms` 250,
`partition-strategy` 'wiki-prefix' (instead of 'hash'), `partition-buckets` 26, `remote-batch-size` 32, and
`duration-secs` 4 (it crawls for 4 seconds, but any enqued link is still awaited, so it ran for 26.61s) is **75.12
pages/sec**.

## Metrics

When running

```
  cargo run --example wiki --features multi_thread -- \
    --duration-secs 4 \
    --partition wiki-prefix \
    --partition-namespace \
    --partition-buckets 26 \
    --remote-batch-size 32
```

The crawl metrics were:

```
--- crawl metrics (26.61s) ---
pages fetched: 1999
urls fetched/sec: 75.12
urls discovered: 4517
urls enqueued: 1995
duplicate skips: 2522
frontier rejects: 0
http errors: 0
url parse errors: 0
local shard enqueues: 7889
remote shard links: 2739 (batches 344)
```

## Highlights

- **Streaming-first parsing.** The default build runs entirely on a single Tokio thread with `lol_html`, harvesting
  links as bytes arrive so memory stays bounded to the current response.
- **Sharded multi-thread mode.** Enable the `multi_thread` feature to spin up several single-thread runtimes in
  parallel. Each shard owns its own frontier and exchanges cross-shard links over Tokio channels, which keeps contention
  low while scaling to multiple cores.
- **Deterministic politeness.** `CrawlControls` exposes depth limits, per-domain allow lists, politeness delays, and
  other knobs so you never need to edit the example binary to tweak behavior.
- **Actionable metrics.** Every run prints pages fetched, URLs/sec, dedupe counts, and error totals so you can tune the
  pipeline quickly.

## Getting Started

```sh
git clone https://github.com/aaronlifton/fastcrawl.git
cd fastcrawl
cargo run --example wiki
```

That command launches the streaming single-thread runtime, seeded with a handful of Wikipedia URLs.

## Runtime Modes

### Single-thread (default)

```
cargo run --example wiki
```

- Builds without extra features.
- Uses `tokio::runtime::Builder::new_current_thread()` plus a `LocalSet`, meaning every worker can hold `lol_html`
  rewriters (which rely on `Rc<RefCell<_>>`).
- Starts extracting links while the response body is still in flight—ideal for tight politeness windows or
  memory-constrained environments.

### Multi-thread (sharded streaming)

```
cargo run --example wiki --features multi_thread -- --duration-secs 60
```

- Spawns one OS thread per shard (stack size bumped to 8 MB to support deep streaming stacks).
- Each shard runs the same streaming workers as single-thread mode but owns a unique frontier and Bloom filter.
- Cross-shard discoveries are routed through bounded mpsc channels, so enqueue contention happens on a single consumer
  instead of every worker.
- Pass `--partition wiki-prefix` (default: `hash`) to keep Wikipedia articles with similar prefixes on the same shard.
- Use `--partition-buckets <n>` (default `0`, meaning shard count) to control how many alphabetical buckets feed into
  the shards, and `--partition-namespace` to keep namespaces like `Talk:` or `Help:` on stable shards.
- Tune `--remote-batch-size <n>` (default 8) to control how many cross-shard links get buffered before the router sends
  them; higher values reduce channel wakeups at the cost of slightly delayed enqueues on the destination shard.
- Enable `--remote-channel-logs` only when debugging channel shutdowns; it reintroduces the verbose “remote shard …
  closed” logs.

## Customizing Crawls

- `CrawlControls` (exposed via CLI/env vars) manage maximum depth, per-domain filters, link-per-page caps, politeness
  delays, run duration, and more. See `src/controls.rs` for every option.
- `UrlFilter` lets you inject arbitrary site-specific logic—`examples/wiki.rs` filters out non-article namespaces and
  query patterns.
- Metrics live in `src/runtime.rs` and can be extended if you need additional counters or telemetry sinks. Multi-thread
  runs also report `local shard enqueues` vs `remote shard links (batches)` so you can gauge partition efficiency.

## Corpus Normalization

Pass `--normalize` to stream every fetched page through the new `Normalizer` service. The pipeline writes newline-
delimited JSON (metadata + cleaned text blocks + embedding-ready chunks) to `--normalize-jsonl` (default:
`normalized_pages.jsonl`) and respects additional knobs:

```
cargo run --example wiki \
  --features multi_thread -- \
  --duration-secs 30 \
  --partition wiki-prefix \
  --partition-namespace \
  --partition-buckets 26 \
  --remote-batch-size 32 \
  --normalize \
  --normalize-jsonl data/wiki.jsonl \
  --normalize-manifest-jsonl data/wiki_manifest.jsonl \
  --normalize-chunk-tokens 384 \
  --normalize-overlap-tokens 64
```

Chunk and block bounds can be tuned via `--normalize-chunk-tokens`, `--normalize-overlap-tokens`, and
`--normalize-max-blocks`. The JSON payload includes per-block heading context, content hashes, token estimates, and
metadata such as HTTP status, language hints, and shard ownership so downstream embedding/indexing jobs can ingest it
directly. When `--normalize-manifest-jsonl` is set, the runtime loads any existing manifest at that path before
overwriting it, then appends digest records (`url`, `checksum`, `last_seen_epoch_ms`, `changed`). Keeping that JSONL
file between runs unlocks true incremental diffs instead of just reporting changes that happened within a single
process.

## Embedding Pipeline

`fastcrawl-embedder` replaces the toy bag-of-words demo with real embedding providers. Point it at the normalized JSONL
stream and it batches chunk text into the configured backend (OpenAI by default):

First run:

-

```sh
cargo run --example wiki -- \
  --normalize \
  --normalize-jsonl data/wiki.jsonl \
  --normalize-manifest-jsonl data/wiki_manifest.jsonl \
  --duration-secs 4
```

Once that finishes, run the embedder command:

```sh
OPENAI_API_KEY=sk-yourkey \
cargo run --bin embedder -- \
  --input data/wiki.jsonl \
  --manifest data/wiki_manifest.jsonl \
  --output data/wiki_embeddings.jsonl \
  --batch-size 64 \
  --only-changed
```

To use Qdrant Cloud Inference instead of OpenAI:

```sh
QDRANT_API_KEY=secret \
cargo run --bin embedder -- \
  --provider qdrant \
  --qdrant-endpoint https://YOUR-CLUSTER.cloud.qdrant.io/inference/text \
  --qdrant-model qdrant/all-MiniLM-L6-v2 \
  --input data/wiki.jsonl \
  --manifest data/wiki_manifest.jsonl \
  --output data/wiki_embeddings.jsonl
```

Important flags/env vars:

- `--provider` selects `openai` (default) or `qdrant`.
- `OPENAI_API_KEY` must be set (or `--openai-api-key` passed) when `--provider openai`.
- `--openai-model` chooses any embedding-capable model (e.g. `text-embedding-3-large`).
- `--openai-dimensions` optionally asks OpenAI to project to a smaller dimension.
- `QDRANT_API_KEY`, `--qdrant-endpoint`, and `--qdrant-model` configure the Qdrant backend.
- `--batch-size` (env `FASTCRAWL_EMBED_BATCH`) controls request fan-out (default 32, retries/backoff handled automatically).
- `--openai-threads` (alias `--worker-threads`, or `FASTCRAWL_OPENAI_THREADS`) fans batches out to multiple worker
  threads so you can overlap network latency when OpenAI throttles.

The embedder still emits newline-delimited `EmbeddedChunkRecord`s compatible with downstream tooling. Set
`--only-changed` alongside the manifest produced by normalization to skip chunks whose manifest `changed` flag stayed
false, so re-embedding only happens when the crawler observed fresh content.

## pgvector Store

Ship the embeddings into Postgres with the bundled `fastcrawl-pgvector` binary. It ingests the JSONL produced above and
upserts into a `vector` table (creating the `vector` extension/table automatically unless disabled). The repo now ships
a `docker-compose.yml` that launches a local Postgres instance with the `pgvector` extension preinstalled:

```sh
docker compose up -d pgvector
```

Once the container is healthy, point `DATABASE_URL` at it and run the loader:

```fish
set -gx DATABASE_URL postgres://postgres:postgres@localhost:5432/fastcrawl
```

```sh
export DATABASE_URL=postgres://postgres:postgres@localhost:5432/fastcrawl
```

```sh
docker-compose up;

cargo run --bin pgvector_store -- \
  --input data/wiki_embeddings.jsonl \
  --schema public \
  --table wiki_chunks \
  --batch-size 256 \
  --upsert \
  --database-url=postgresql://postgres:postgres@localhost:5432
```

Stop the container with `docker compose down` (pass `-v` to remove the persisted volume if you want a clean slate).

`fastcrawl-pgvector` now provisions a generated `text_tsv` column plus a GIN index so Postgres full-text queries stay
fast. If you already had a table before this change, run the helper to retrofit the column/index (and optionally
ANALYZE) without reloading vectors:

```sh
cargo run --bin fts_indexer -- \
  --database-url $DATABASE_URL \
  --schema public \
  --table wiki_chunks
```

Columns created by default:

- `url TEXT`, `chunk_id BIGINT` primary key for provenance.
- `text`, `section_path JSONB`, `token_estimate`, `checksum`, `last_seen_epoch_ms` for metadata.
- `embedding VECTOR(<dims>)` where `<dims>` matches the first record’s vector length.
- `text_tsv TSVECTOR` generated from the chunk text, indexed for lexical search.

With vectors in pgvector you can run similarity search straight from SQL, plug it into RAG services, or join additional
metadata tables to restrict retrieval.

## Retrieval Evaluation Harness

Quantify how well pgvector surfaces the right chunks before plugging them into an LLM:

```
cargo run --bin vector_eval -- \
  --cases data/wiki_eval.jsonl \
  --database-url postgres://postgres:postgres@localhost:5432/fastcrawl \
  --schema public \
  --table wiki_chunks \
  --top-k 5 \
  --dense-candidates 32 \
  --lexical-candidates 64 \
  --rrf-k 60 \
  --report-json data/wiki_eval_report.json \
  --openai-api-key $OPENAI_API_KEY
```

The CLI embeds each query in `data/wiki_eval.jsonl`, performs the same dense+lexical fusion used by the HTTP retriever,
and prints hit-rate / MRR / recall metrics. Tweak fusion behaviour with the candidate and `--rrf-k` knobs. The optional
`--report-json` writes a structured summary for dashboards (including fused/dense/lexical scores per chunk).

## Embedding Freshness Planner

Detect drift between two manifest snapshots and emit a plan describing which URLs need re-embedding:

```
cargo run --bin freshness -- \
  --current-manifest data/wiki_manifest.jsonl \
  --previous-manifest data/wiki_manifest_prev.jsonl \
  --plan-output data/refresh_plan.jsonl \
  --ledger-output data/embedding_ledger.jsonl \
  --exec-plan 'cargo run --bin embedder -- --input data/wiki.jsonl --manifest data/wiki_manifest.jsonl --output data/wiki_embeddings.jsonl --only-changed --openai-api-key $OPENAI_API_KEY'
```

`fastcrawl-freshness` diffs the manifests, prints counts for new/changed/deleted URLs, writes JSONL plan entries,
appends an audit ledger, and (optionally) runs a shell hook once the plan exists. Pass `--dry-run` to see stats without
touching files or invoking the hook.

## Retrieval API

Expose pgvector retrieval over HTTP for downstream services:

```
cargo run --bin retriever_api -- \
  --database-url postgres://postgres:postgres@localhost:5432/fastcrawl \
  --schema public \
  --table wiki_chunks \
  --bind 0.0.0.0:8080 \
  --openai-api-key $OPENAI_API_KEY \
  --dense-candidates 32 \
  --lexical-candidates 64
```

Endpoints:

- `GET /healthz` – liveness probe.
- `POST /v1/query` – `{ "query": "When did Apollo 11 land?", "top_k": 6 }` returns scored chunks filtered by the token
  budget. Override the budget per request via `max_tokens`.

The retriever now performs hybrid search: dense candidates from pgvector plus lexical matches taken from Postgres full-
text search, fused via Reciprocal Rank Fusion, then trimmed by an optional token budget. Each chunk reports
`dense_distance`, optional `lexical_score`, and the final `fused_score`/rank so clients can understand why it surfaced.
Tune the hybrid behaviour with `--dense-candidates`, `--lexical-candidates`, and `--rrf-k`. By default the server also
caches 1,024 query embeddings and enforces 120 requests/minute with a burst of 12; adjust via `--embedding-cache-size`,
`--max-requests-per-minute`, and `--rate-limit-burst`.

Pair this API with the prompt templates in `personal_docs/prompt_templates/wiki_rag.md` to keep LLM formatting
consistent.

## LLM-Oriented Next Steps

Fastcrawl is already a solid content harvester for downstream ML pipelines. Future work aimed at LLM/RAG workflows
includes:

- [x] **Corpus normalization** – strip boilerplate, capture metadata, and chunk pages into consistent token windows.

2. **Embedding pipeline** – push cleaned chunks through an embedding model and store vectors (pgvector/Qdrant/Milvus)
   with provenance.
3. **Incremental refresh** – schedule revisits, diff pages, and update embeddings so the knowledge base stays current.
4. **Training data generation** – turn chunks into instruction/QA pairs or causal LM samples; track licensing for
   Wikipedia’s CC BY-SA requirements.
5. **Retrieval-augmented answering** – wire the crawler to trigger re-indexing as new pages stream in, then expose a
   lightweight API for LLMs to fetch relevant context on demand.
6. **Policy-aware agent** – use crawl metrics (latency, politeness) to drive an autonomous agent that decides which
   sections of the web to expand next based on embedding coverage gaps.

## License

Copyright © 2025 Aaron Lifton
