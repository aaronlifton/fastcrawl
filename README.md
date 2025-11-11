# Fastcrawl

## Features

### Single-thread (Uses `lol_html` streaming by default)

Usage: `cargo run --example wiki `

- Builds with the default Cargo feature set (no `--features multi_thread`).
- Uses `tokio::runtime::Builder::new_current_thread()` plus a `LocalSet`, allowing each worker to hold `lol_html`
  rewriters that rely on `Rc<RefCell<…>>`.
- Streams HTML as it arrives via `stream_links`, so memory stays bound to the current page and we can discover links
  before the entire body downloads.
- Best when you want polite, predictable crawling that doesn’t need to saturate multiple cores.

### Multi-thread (Uses `scraper`)

Usage:

```sh
cargo run --example wiki --features multi_thread -- --duration-secs 60
```

- lol_html is great for streaming, but its internals rely heavily on Rc<RefCell<…>>. That means any future which holds a
  HtmlRewriter is not Send, so Tokio refuses to spawn it onto a multi-thread runtime. In the single-thread path we solve
  that by running everything inside a LocalSet (spawn_local doesn’t require Send), so we get true streaming extraction.

- When you flip on the multi_thread Cargo feature, we want to exploit Tokio’s work-stealing executor and tokio::spawn,
  which does require every captured value to be Send. Rather than fight lol_html’s design, we switch to a buffering
  parser built on scraper: we download the full response (Response::text().await), then traverse it with
  scraper::Html/Selector. That approach is Send-friendly because the parsed DOM lives in owned Strings/Vecs, so each
  worker future can hop threads safely. It trades a bit of memory and latency (you wait for the full body) for
  compatibility with multi-threaded execution.

---

## Notes

- If we ever wanted streaming + multi-thread, we’d need either a different streaming parser with Send support or to keep
  each parser bound to a per-worker LocalSet and communicate with the main pool via channels. For now, tying scraper to
  the multi_thread feature is the most straightforward way to unlock higher throughput without rewriting lol_html.

### Comparison

- Single-thread mode wins here because its whole data flow stays on one Tokio thread and never leaves streaming mode:
  - spawn_streaming_worker uses tokio::task::spawn_local, so every worker stays on the same runtime thread as the
    frontier/metrics (src/runtime.rs:168- 191). That means no Send bounds, no cross-thread channels, and no cache-
    busting handoffs. In multi-thread mode we switch to tokio::spawn (src/ runtime.rs:194-218), so each worker hops
    across threads and pays the overhead of moving tasks, atomics, and wakeups through Tokio’s scheduler.
  - The single-thread handler calls stream_links directly and parses links as the response body arrives
    (src/runtime.rs:320-355). We never buffer the full HTML, so latency per page is basically “network + incremental
    parsing,” which keeps each worker busy.
  - The multi-thread handler must await extract_links_buffered, which first collects the whole response into memory
    before scraper can walk it (`src/runtime.rs:373-405`). That adds at least one extra copy of every page, more
    allocations, and more CPU per document just to reach the same set of links. On sites like Wikipedia, those extra
    milliseconds per page easily erase the theoretical parallelism benefit.
  - Because every buffered worker is a full Tokio task, they all contend on the shared Frontier and metrics atomics more
    aggressively (the hot loop in enqueue_discovered_links, `src/runtime.rs:431-460`). In single-thread mode those
    atomics are hit in-order on one core; in multi-thread mode, multiple cores fight over them and end up spinning.

- So even though the multi-thread feature adds more worker tasks, each task is heavier, and the extra synchronization
  ends up dominating. To make the multi-thread path competitive you’d need a parser that can stream without Rc (so it’s
  Send), or redesign the buffered extractor so it doesn’t have to materialize entire pages before handing links back to
  the frontier.

---

Copyright 2025 Aaron Lifton
