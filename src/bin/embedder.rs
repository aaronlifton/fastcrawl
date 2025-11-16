use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TryRecvError};
use fastcrawl::embedder::openai::OpenAiEmbedder;
use fastcrawl::embedder::qdrant::QdrantEmbedder;
use fastcrawl::{EmbeddedChunkRecord, ManifestRecord, NormalizedPage};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-embedder",
    about = "Prototype embedding pipeline for normalized pages"
)]
struct EmbedCli {
    /// Path to the normalized JSONL produced by `--normalize` runs
    #[arg(
        long,
        env = "FASTCRAWL_EMBED_INPUT",
        default_value = "normalized_pages.jsonl"
    )]
    input: PathBuf,

    /// Optional manifest JSONL to consult for incremental refresh decisions
    #[arg(long, env = "FASTCRAWL_EMBED_MANIFEST")]
    manifest: Option<PathBuf>,

    /// Output JSONL containing embedded chunk records
    #[arg(
        long,
        env = "FASTCRAWL_EMBED_OUTPUT",
        default_value = "embeddings.jsonl"
    )]
    output: PathBuf,

    /// Only emit embeddings for manifest entries marked as changed
    #[arg(long, default_value_t = false)]
    only_changed: bool,

    /// Embedding provider to use (openai or qdrant)
    #[arg(long, env = "FASTCRAWL_EMBED_PROVIDER", default_value = "openai")]
    provider: String,

    /// OpenAI API key used for embedding calls
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    /// Embedding model identifier (e.g. text-embedding-3-small)
    #[arg(
        long,
        env = "FASTCRAWL_OPENAI_MODEL",
        default_value = "text-embedding-3-small"
    )]
    openai_model: String,

    /// Optional dimension override when supported by the model
    #[arg(long, env = "FASTCRAWL_OPENAI_DIMENSIONS")]
    openai_dimensions: Option<usize>,

    /// Base URL for the OpenAI-compatible API
    #[arg(
        long,
        env = "FASTCRAWL_OPENAI_BASE",
        default_value = "https://api.openai.com/v1"
    )]
    openai_base_url: String,

    /// Max number of chunks to send per embedding request
    #[arg(long, env = "FASTCRAWL_EMBED_BATCH", default_value_t = 32)]
    batch_size: usize,

    /// Max seconds to wait for each embedding request
    #[arg(long, env = "FASTCRAWL_OPENAI_TIMEOUT_SECS", default_value_t = 30)]
    openai_timeout_secs: u64,

    /// Number of retries for rate limits or transient errors
    #[arg(long, env = "FASTCRAWL_OPENAI_MAX_RETRIES", default_value_t = 5)]
    max_retries: usize,

    /// Qdrant API key (falls back to QDRANT_API_KEY env var)
    #[arg(long, env = "QDRANT_API_KEY")]
    qdrant_api_key: Option<String>,

    /// Qdrant inference endpoint (e.g. https://cluster-id.cloud.qdrant.io/inference/text)
    #[arg(long, env = "FASTCRAWL_QDRANT_ENDPOINT")]
    qdrant_endpoint: Option<String>,

    /// Qdrant model identifier advertised by the inference cluster
    #[arg(long, env = "FASTCRAWL_QDRANT_MODEL")]
    qdrant_model: Option<String>,

    /// Number of concurrent embedding workers
    #[arg(
        long = "openai-threads",
        env = "FASTCRAWL_OPENAI_THREADS",
        default_value_t = 1,
        alias = "worker-threads"
    )]
    worker_threads: usize,
}

fn main() -> Result<()> {
    let cli = EmbedCli::parse();
    let manifest = if let Some(path) = &cli.manifest {
        Some(load_manifest(path).context("failed to read manifest")?)
    } else {
        None
    };

    let batch_size = cli.batch_size.max(1);
    let timeout = Duration::from_secs(cli.openai_timeout_secs.max(1));
    let max_retries = cli.max_retries.max(1);
    let provider = cli.provider.to_lowercase();
    let embedder = match provider.as_str() {
        "openai" => {
            let api_key = cli.openai_api_key.clone().ok_or_else(|| {
                anyhow!("--openai-api-key is required when using the OpenAI provider")
            })?;
            EmbeddingClient::OpenAi(OpenAiEmbedder::new(
                api_key,
                cli.openai_base_url.clone(),
                cli.openai_model.clone(),
                cli.openai_dimensions,
                timeout,
                max_retries,
                batch_size,
            )?)
        }
        "qdrant" => {
            let api_key = cli.qdrant_api_key.clone().ok_or_else(|| {
                anyhow!("--qdrant-api-key (or QDRANT_API_KEY) is required for the Qdrant provider")
            })?;
            let endpoint = cli
                .qdrant_endpoint
                .clone()
                .ok_or_else(|| anyhow!("--qdrant-endpoint is required for the Qdrant provider"))?;
            let model = cli
                .qdrant_model
                .clone()
                .ok_or_else(|| anyhow!("--qdrant-model is required for the Qdrant provider"))?;
            EmbeddingClient::Qdrant(QdrantEmbedder::new(
                api_key,
                endpoint,
                model,
                timeout,
                max_retries,
                batch_size,
            )?)
        }
        other => anyhow::bail!("unsupported embedding provider '{}'", other),
    };
    let input =
        File::open(&cli.input).with_context(|| format!("failed to open {:?}", cli.input))?;
    let reader = BufReader::new(input);
    let output =
        File::create(&cli.output).with_context(|| format!("failed to create {:?}", cli.output))?;
    let mut writer = BufWriter::new(output);

    process_stream(
        reader,
        &mut writer,
        &embedder,
        batch_size,
        manifest.as_ref(),
        cli.only_changed,
        cli.worker_threads,
    )?;
    writer.flush()?;
    Ok(())
}

fn process_stream<R: BufRead, W: Write>(
    reader: R,
    writer: &mut W,
    embedder: &EmbeddingClient,
    batch_size: usize,
    manifest: Option<&HashMap<String, bool>>,
    only_changed: bool,
    worker_threads: usize,
) -> Result<()> {
    let worker_threads = worker_threads.max(1);
    eprintln!(
        "launching embedder with batch size {} across {} worker(s)...",
        batch_size, worker_threads
    );
    let (task_tx, task_rx) = bounded::<EmbeddingTask>(worker_threads * 2);
    let (result_tx, result_rx) = bounded::<EmbeddingResult>(worker_threads * 2);

    for worker_id in 0..worker_threads {
        let worker_embedder = embedder.clone();
        let worker_rx = task_rx.clone();
        let worker_tx = result_tx.clone();
        thread::spawn(move || worker_loop(worker_id, worker_rx, worker_tx, worker_embedder));
    }
    drop(task_rx);
    drop(result_tx);

    let mut pending: Vec<EmbeddedChunkRecord> = Vec::with_capacity(batch_size);
    let mut pending_results: BTreeMap<usize, Vec<EmbeddedChunkRecord>> = BTreeMap::new();
    let mut embedded_chunks = 0usize;
    let mut total_pages = 0usize;
    let mut skipped_pages = 0usize;
    let mut next_batch_id = 0usize;
    let mut next_result_id = 0usize;
    let mut inflight_batches = 0usize;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let page: NormalizedPage = serde_json::from_str(&line)
            .with_context(|| format!("invalid normalized page at line {}", line_no + 1))?;
        total_pages += 1;
        if only_changed {
            let url = page.metadata.url.as_str();
            let changed = manifest
                .and_then(|records| records.get(url).copied())
                .unwrap_or(true);
            if !changed {
                skipped_pages += 1;
                continue;
            }
        }

        let page_url = page.metadata.url.to_string();
        let checksum = page.metadata.checksum;
        let last_seen_epoch_ms = page.metadata.fetched_epoch_ms();
        for chunk in page.chunks {
            let record = EmbeddedChunkRecord {
                url: page_url.clone(),
                chunk_id: chunk.chunk_id,
                text: chunk.text.clone(),
                section_path: chunk.section_path.clone(),
                token_estimate: chunk.token_estimate,
                embedding: Vec::new(),
                checksum,
                last_seen_epoch_ms,
            };
            pending.push(record);
            if pending.len() >= batch_size {
                if let Some(count) = dispatch_batch(
                    &mut pending,
                    &mut next_batch_id,
                    &mut inflight_batches,
                    &task_tx,
                )? {
                    eprintln!("queued embedding batch of {} chunks...", count);
                }
                drain_ready_results(
                    &result_rx,
                    &mut pending_results,
                    &mut next_result_id,
                    writer,
                    &mut embedded_chunks,
                    &mut inflight_batches,
                    total_pages,
                    skipped_pages,
                )?;
            }
        }
    }

    if let Some(count) = dispatch_batch(
        &mut pending,
        &mut next_batch_id,
        &mut inflight_batches,
        &task_tx,
    )? {
        eprintln!("queued embedding batch of {} chunks...", count);
    }
    drop(task_tx);

    while inflight_batches > 0 {
        let result = match result_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(res) => res,
            Err(RecvTimeoutError::Timeout) => {
                eprintln!(
                    "still waiting on {} embedding batch(es); oldest pending batch id {}...",
                    inflight_batches, next_result_id
                );
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                anyhow::bail!("embedding worker channel closed unexpectedly")
            }
        };
        match result {
            Ok(batch) => {
                inflight_batches -= 1;
                process_result(
                    batch,
                    &mut pending_results,
                    &mut next_result_id,
                    writer,
                    &mut embedded_chunks,
                    total_pages,
                    skipped_pages,
                )?;
            }
            Err(err) => return Err(err),
        }
    }

    eprintln!(
        "embedding complete: {} chunks written from {} pages ({} skipped).",
        embedded_chunks,
        total_pages - skipped_pages,
        skipped_pages
    );
    if embedded_chunks == 0 {
        eprintln!(
            "no chunks qualified for embedding; check manifest/filters if this is unexpected."
        );
    }
    Ok(())
}

fn load_manifest(path: &PathBuf) -> Result<HashMap<String, bool>> {
    let file = File::open(path).with_context(|| format!("failed to open manifest {:?}", path))?;
    let reader = BufReader::new(file);
    let mut entries = HashMap::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read manifest line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ManifestRecord = serde_json::from_str(&line)
            .with_context(|| format!("invalid manifest record at line {}", line_no + 1))?;
        entries.insert(record.url, record.changed);
    }
    Ok(entries)
}

fn dispatch_batch(
    pending: &mut Vec<EmbeddedChunkRecord>,
    next_batch_id: &mut usize,
    inflight: &mut usize,
    sender: &Sender<EmbeddingTask>,
) -> Result<Option<usize>> {
    if pending.is_empty() {
        return Ok(None);
    }

    let mut records = Vec::with_capacity(pending.len());
    records.append(pending);
    let chunk_count = records.len();
    let batch = EmbeddingTask {
        id: *next_batch_id,
        records,
    };
    *next_batch_id += 1;
    *inflight += 1;
    sender
        .send(batch)
        .map_err(|_| anyhow!("embedding worker channel closed"))?;
    Ok(Some(chunk_count))
}

fn drain_ready_results<W: Write>(
    result_rx: &Receiver<EmbeddingResult>,
    pending_results: &mut BTreeMap<usize, Vec<EmbeddedChunkRecord>>,
    next_result_id: &mut usize,
    writer: &mut W,
    embedded_chunks: &mut usize,
    inflight_batches: &mut usize,
    total_pages: usize,
    skipped_pages: usize,
) -> Result<()> {
    loop {
        let result = match result_rx.try_recv() {
            Ok(res) => res,
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                anyhow::bail!("embedding worker channel closed unexpectedly")
            }
        };
        *inflight_batches = inflight_batches.saturating_sub(1);
        process_result(
            result?,
            pending_results,
            next_result_id,
            writer,
            embedded_chunks,
            total_pages,
            skipped_pages,
        )?;
    }
    Ok(())
}

fn process_result<W: Write>(
    batch: EmbeddingBatchResult,
    pending_results: &mut BTreeMap<usize, Vec<EmbeddedChunkRecord>>,
    next_result_id: &mut usize,
    writer: &mut W,
    embedded_chunks: &mut usize,
    total_pages: usize,
    skipped_pages: usize,
) -> Result<()> {
    pending_results.insert(batch.id, batch.records);
    while let Some(records) = pending_results.remove(next_result_id) {
        let written = write_records(writer, records)?;
        if written > 0 {
            *embedded_chunks += written;
            eprintln!(
                "embedded {} chunks ({} pages processed, {} skipped)...",
                *embedded_chunks,
                total_pages - skipped_pages,
                skipped_pages
            );
        }
        *next_result_id += 1;
    }
    Ok(())
}

fn write_records<W: Write>(writer: &mut W, records: Vec<EmbeddedChunkRecord>) -> Result<usize> {
    let mut written = 0usize;
    for record in records {
        serde_json::to_writer(&mut *writer, &record)?;
        writer.write_all(b"\n")?;
        written += 1;
    }
    Ok(written)
}

fn worker_loop(
    worker_id: usize,
    receiver: Receiver<EmbeddingTask>,
    sender: Sender<EmbeddingResult>,
    embedder: EmbeddingClient,
) {
    for task in receiver.iter() {
        let EmbeddingTask {
            id: batch_id,
            mut records,
        } = task;
        let chunk_count = records.len();
        eprintln!(
            "worker {} embedding batch {} ({} chunks)...",
            worker_id, batch_id, chunk_count
        );
        let result = embed_records(&embedder, &mut records)
            .map(|_| {
                eprintln!(
                    "worker {} completed batch {} ({} chunks).",
                    worker_id, batch_id, chunk_count
                );
                EmbeddingBatchResult {
                    id: batch_id,
                    records,
                }
            })
            .map_err(|err| {
                anyhow!(
                    "worker {} failed batch {} ({} chunks): {}",
                    worker_id,
                    batch_id,
                    chunk_count,
                    err
                )
            });
        if sender.send(result).is_err() {
            break;
        }
    }
}

fn embed_records(embedder: &EmbeddingClient, records: &mut Vec<EmbeddedChunkRecord>) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    let inputs: Vec<&str> = records.iter().map(|record| record.text.as_str()).collect();
    let embeddings = embedder.embed_batch(&inputs)?;
    anyhow::ensure!(
        embeddings.len() == records.len(),
        "embedding count {} mismatched pending {}",
        embeddings.len(),
        records.len()
    );

    for (record, vector) in records.iter_mut().zip(embeddings.into_iter()) {
        record.embedding = vector;
    }
    Ok(())
}

struct EmbeddingTask {
    id: usize,
    records: Vec<EmbeddedChunkRecord>,
}

struct EmbeddingBatchResult {
    id: usize,
    records: Vec<EmbeddedChunkRecord>,
}

type EmbeddingResult = Result<EmbeddingBatchResult>;

#[derive(Clone)]
enum EmbeddingClient {
    OpenAi(OpenAiEmbedder),
    Qdrant(QdrantEmbedder),
}

impl EmbeddingClient {
    fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            EmbeddingClient::OpenAi(client) => client.embed_batch(inputs),
            EmbeddingClient::Qdrant(client) => client.embed_batch(inputs),
        }
    }
}
