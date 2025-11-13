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

## LLM-Oriented Next Steps

Fastcrawl is already a solid content harvester for downstream ML pipelines. Future work aimed at LLM/RAG workflows
includes:

1. **Corpus normalization** – strip boilerplate, capture metadata, and chunk pages into consistent token windows.
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
