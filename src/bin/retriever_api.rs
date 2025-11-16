use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use fastcrawl::embedder::openai::OpenAiEmbedder;
use fastcrawl::normalizer::SectionHeading;
use fastcrawl::TableName;
use lru::LruCache;
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_postgres::types::Json as PgJson;
use tokio_postgres::{Client, NoTls, Row};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-retriever",
    about = "HTTP API that wraps pgvector similarity search for Fastcrawl chunks"
)]
struct ApiCli {
    /// Address to bind the HTTP server to (host:port).
    #[arg(long, env = "FASTCRAWL_BIND", default_value = "127.0.0.1:8080")]
    bind: String,

    /// Postgres connection string (postgres://...).
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Schema for the pgvector table.
    #[arg(long, env = "FASTCRAWL_RETRIEVER_SCHEMA", default_value = "public")]
    schema: String,

    /// Table storing embedded chunks.
    #[arg(long, env = "FASTCRAWL_RETRIEVER_TABLE", default_value = "wiki_chunks")]
    table: String,

    /// Default top-k when the client does not override it.
    #[arg(long, default_value_t = 5)]
    default_top_k: usize,

    /// Maximum top-k allowed per request.
    #[arg(long, default_value_t = 12)]
    max_top_k: usize,

    /// Optional global token budget for the summed token_estimate returned.
    #[arg(long)]
    max_response_tokens: Option<u64>,

    /// OpenAI API key used for query embeddings.
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: String,

    /// Embedding model identifier.
    #[arg(
        long,
        env = "FASTCRAWL_OPENAI_MODEL",
        default_value = "text-embedding-3-small"
    )]
    openai_model: String,

    /// Optional embedding dimension override.
    #[arg(long, env = "FASTCRAWL_OPENAI_DIMENSIONS")]
    openai_dimensions: Option<usize>,

    /// Base URL for OpenAI-compatible endpoints.
    #[arg(
        long,
        env = "FASTCRAWL_OPENAI_BASE",
        default_value = "https://api.openai.com/v1"
    )]
    openai_base_url: String,

    /// Max inputs per embedding request.
    #[arg(long, env = "FASTCRAWL_OPENAI_BATCH", default_value_t = 32)]
    batch_size: usize,

    /// Seconds before OpenAI requests time out.
    #[arg(long, env = "FASTCRAWL_OPENAI_TIMEOUT_SECS", default_value_t = 30)]
    openai_timeout_secs: u64,

    /// Retry attempts for transient embedding errors.
    #[arg(long, env = "FASTCRAWL_OPENAI_MAX_RETRIES", default_value_t = 5)]
    max_retries: usize,

    /// Max cached query embeddings kept in-memory (0 disables caching).
    #[arg(long, default_value_t = 1024)]
    embedding_cache_size: usize,

    /// Max requests per minute allowed (0 disables rate limiting).
    #[arg(long, default_value_t = 120)]
    max_requests_per_minute: u32,

    /// Rate-limit burst size (tokens available instantly).
    #[arg(long, default_value_t = 12)]
    rate_limit_burst: u32,

    /// Dense candidate count fetched from pgvector before fusion.
    #[arg(long, default_value_t = 24)]
    dense_candidates: usize,

    /// Lexical candidate count fetched via Postgres full-text search before fusion.
    #[arg(long, default_value_t = 40)]
    lexical_candidates: usize,

    /// Reciprocal Rank Fusion constant (higher softens score differences).
    #[arg(long, default_value_t = 60.0)]
    rrf_k: f64,
}

#[derive(Clone)]
struct AppState {
    db: Arc<Client>,
    query_sql: Arc<String>,
    lexical_sql: Arc<String>,
    embedder: Arc<OpenAiEmbedder>,
    default_top_k: usize,
    max_top_k: usize,
    max_response_tokens: Option<u64>,
    embedding_cache: Option<Arc<Mutex<LruCache<String, Vec<f32>>>>>,
    rate_limiter: Option<RateLimiter>,
    dense_candidates: usize,
    lexical_candidates: usize,
    rrf_k: f64,
}

#[derive(Debug, Deserialize)]
struct QueryRequest {
    query: String,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    max_tokens: Option<u64>,
}

#[derive(Debug, Serialize)]
struct QueryResponse {
    chunks: Vec<ResponseChunk>,
    meta: ResponseMeta,
}

#[derive(Debug, Serialize)]
struct ResponseMeta {
    top_k: usize,
    latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_budget: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fusion: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResponseChunk {
    url: String,
    chunk_id: i64,
    section_path: Vec<SectionHeading>,
    token_estimate: i64,
    text: String,
    checksum: u32,
    last_seen_epoch_ms: i64,
    dense_distance: Option<f64>,
    lexical_score: Option<f64>,
    fused_score: f64,
    fused_rank: usize,
}

#[derive(Clone)]
struct ChunkData {
    url: String,
    chunk_id: i64,
    section_path: Vec<SectionHeading>,
    token_estimate: i64,
    text: String,
    checksum: u32,
    last_seen_epoch_ms: i64,
}

struct DenseRow {
    data: ChunkData,
    distance: f64,
}

struct LexicalRow {
    data: ChunkData,
    score: f64,
}

struct Candidate {
    data: ChunkData,
    dense_rank: Option<usize>,
    lexical_rank: Option<usize>,
    dense_distance: Option<f64>,
    lexical_score: Option<f64>,
    fused_score: f64,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = ApiCli::parse();
    let table = TableName::new(cli.schema, cli.table)?;
    let embedder = Arc::new(OpenAiEmbedder::new(
        cli.openai_api_key,
        cli.openai_base_url,
        cli.openai_model,
        cli.openai_dimensions,
        std::time::Duration::from_secs(cli.openai_timeout_secs.max(1)),
        cli.max_retries.max(1),
        cli.batch_size.max(1),
    )?);

    let (client, connection) = tokio_postgres::connect(&cli.database_url, NoTls)
        .await
        .with_context(|| format!("failed to connect to Postgres at {}", cli.database_url))?;
    tokio::spawn(async move {
        if let Err(err) = connection.await {
            eprintln!("postgres connection error: {err}");
        }
    });
    let query_sql = Arc::new(select_sql(&table));
    let lexical_sql = Arc::new(select_lexical_sql(&table));
    let cache = build_cache(cli.embedding_cache_size);
    let rate_limiter = RateLimiter::new(cli.max_requests_per_minute, cli.rate_limit_burst);
    let state = AppState {
        db: Arc::new(client),
        query_sql,
        lexical_sql,
        embedder,
        default_top_k: cli.default_top_k.max(1),
        max_top_k: cli.max_top_k.max(1),
        max_response_tokens: cli.max_response_tokens,
        embedding_cache: cache,
        rate_limiter,
        dense_candidates: cli.dense_candidates.max(1),
        lexical_candidates: cli.lexical_candidates,
        rrf_k: cli.rrf_k.max(1.0),
    };
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/query", post(query_handler))
        .with_state(state);

    let addr: SocketAddr = cli
        .bind
        .parse()
        .with_context(|| format!("invalid bind address {}", cli.bind))?;
    println!("fastcrawl-retriever listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;
    axum::serve(listener, app)
        .await
        .context("server shutdown")?;
    Ok(())
}

async fn healthz() -> StatusCode {
    StatusCode::OK
}

async fn query_handler(
    State(state): State<AppState>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<ErrorBody>)> {
    if request.query.trim().is_empty() {
        return Err(bad_request("query text must not be empty"));
    }
    if let Some(limiter) = &state.rate_limiter {
        if !limiter.acquire().await {
            return Err(too_many_requests("rate limit exceeded"));
        }
    }
    let top_k = request
        .top_k
        .unwrap_or(state.default_top_k)
        .clamp(1, state.max_top_k);
    let token_budget = request.max_tokens.or(state.max_response_tokens);
    let start = Instant::now();
    let embedding = embed_query(&state, request.query.clone())
        .await
        .map_err(internal_error)?;
    let dense_future = fetch_dense_rows(&state, &embedding, state.dense_candidates);
    let lexical_future = fetch_lexical_rows(&state, &request.query, state.lexical_candidates);
    let (dense_rows, lexical_rows) = tokio::join!(dense_future, lexical_future);
    let dense_rows = dense_rows.map_err(internal_error)?;
    let lexical_rows = lexical_rows.map_err(internal_error)?;
    let mut fused = fuse_candidates(dense_rows, lexical_rows, state.rrf_k);
    let keyword_tokens = tokenize_query(&request.query);
    let fallback_limit = if state.lexical_candidates > 0 {
        state.lexical_candidates
    } else {
        state.dense_candidates
    };
    apply_keyword_fallback(&mut fused, &keyword_tokens, state.rrf_k, fallback_limit);
    fused.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    fused.truncate(top_k);
    let response_chunks: Vec<ResponseChunk> = fused
        .into_iter()
        .enumerate()
        .map(|(idx, candidate)| candidate.into_chunk(idx + 1))
        .collect();
    let chunks = apply_token_budget(response_chunks, token_budget);
    let fusion_label = if state.lexical_candidates > 0 {
        Some("dense+lexical_rrf".to_string())
    } else {
        Some("dense_only".to_string())
    };
    let response = QueryResponse {
        chunks,
        meta: ResponseMeta {
            top_k,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            token_budget,
            fusion: fusion_label,
        },
    };
    Ok(Json(response))
}

fn bad_request(message: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorBody {
            message: message.into(),
        }),
    )
}

fn internal_error(err: anyhow::Error) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody {
            message: err.to_string(),
        }),
    )
}

fn too_many_requests(message: impl Into<String>) -> (StatusCode, Json<ErrorBody>) {
    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(ErrorBody {
            message: message.into(),
        }),
    )
}

async fn embed_query(state: &AppState, query: String) -> Result<Vec<f32>, anyhow::Error> {
    if let Some(cache) = &state.embedding_cache {
        if let Some(hit) = {
            let mut guard = cache.lock().await;
            guard.get(&query).cloned()
        } {
            return Ok(hit);
        }
    }

    let embedder = state.embedder.clone();
    let query_clone = query.clone();
    let embedding = tokio::task::spawn_blocking(move || {
        let batch = vec![query_clone];
        let refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();
        let mut embeddings = embedder.embed_batch(&refs)?;
        embeddings
            .pop()
            .ok_or_else(|| anyhow!("OpenAI returned no embedding"))
    })
    .await
    .map_err(|err| anyhow!("embedding task join error: {err}"))??;

    if let Some(cache) = &state.embedding_cache {
        let mut guard = cache.lock().await;
        guard.put(query, embedding.clone());
    }
    Ok(embedding)
}

async fn fetch_dense_rows(
    state: &AppState,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<DenseRow>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let vector = Vector::from(embedding.to_vec());
    let rows = state
        .db
        .query(state.query_sql.as_str(), &[&vector, &(limit as i64)])
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let data = chunk_data_from_row(&row)?;
        let distance: f64 = row.get("distance");
        out.push(DenseRow { data, distance });
    }
    Ok(out)
}

async fn fetch_lexical_rows(
    state: &AppState,
    query_text: &str,
    limit: usize,
) -> Result<Vec<LexicalRow>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let trimmed = query_text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let rows = state
        .db
        .query(state.lexical_sql.as_str(), &[&trimmed, &(limit as i64)])
        .await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let data = chunk_data_from_row(&row)?;
        let score: Option<f32> = row.get("lexical_score");
        out.push(LexicalRow {
            data,
            score: score.unwrap_or(0.0) as f64,
        });
    }
    Ok(out)
}

fn chunk_data_from_row(row: &Row) -> Result<ChunkData> {
    let url: String = row.get("url");
    let chunk_id: i64 = row.get("chunk_id");
    let text: String = row.get("text");
    let PgJson(section_path): PgJson<Vec<SectionHeading>> = row.get("section_path");
    let token_estimate: i64 = row.get("token_estimate");
    let checksum_raw: i64 = row.get("checksum");
    let checksum = u32::try_from(checksum_raw)
        .map_err(|_| anyhow!("checksum {} exceeds u32 range", checksum_raw))?;
    let last_seen_epoch_ms: i64 = row.get("last_seen_epoch_ms");

    Ok(ChunkData {
        url,
        chunk_id,
        section_path,
        token_estimate,
        text,
        checksum,
        last_seen_epoch_ms,
    })
}

fn fuse_candidates(dense: Vec<DenseRow>, lexical: Vec<LexicalRow>, rrf_k: f64) -> Vec<Candidate> {
    let mut map: HashMap<(String, i64), Candidate> = HashMap::new();
    let k = rrf_k.max(1.0);
    for (idx, row) in dense.into_iter().enumerate() {
        let key = (row.data.url.clone(), row.data.chunk_id);
        let entry = map
            .entry(key)
            .or_insert_with(|| Candidate::from_data(row.data));
        entry.dense_rank = Some(idx + 1);
        entry.dense_distance = Some(row.distance);
        entry.fused_score += rrf_contribution(k, idx + 1);
    }
    for (idx, row) in lexical.into_iter().enumerate() {
        let key = (row.data.url.clone(), row.data.chunk_id);
        let entry = map
            .entry(key)
            .or_insert_with(|| Candidate::from_data(row.data));
        entry.lexical_rank = Some(idx + 1);
        entry.lexical_score = Some(row.score);
        entry.fused_score += rrf_contribution(k, idx + 1);
        entry.fused_score += row.score * LEXICAL_SCORE_WEIGHT;
    }
    let mut candidates: Vec<Candidate> = map.into_values().collect();
    candidates.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
}

impl Candidate {
    fn from_data(data: ChunkData) -> Self {
        Self {
            data,
            dense_rank: None,
            lexical_rank: None,
            dense_distance: None,
            lexical_score: None,
            fused_score: 0.0,
        }
    }

    fn into_chunk(self, fused_rank: usize) -> ResponseChunk {
        ResponseChunk {
            url: self.data.url,
            chunk_id: self.data.chunk_id,
            section_path: self.data.section_path,
            token_estimate: self.data.token_estimate,
            text: self.data.text,
            checksum: self.data.checksum,
            last_seen_epoch_ms: self.data.last_seen_epoch_ms,
            dense_distance: self.dense_distance,
            lexical_score: self.lexical_score,
            fused_score: self.fused_score,
            fused_rank,
        }
    }
}

fn rrf_contribution(k: f64, rank: usize) -> f64 {
    1.0 / (k + rank as f64)
}

const LEXICAL_SCORE_WEIGHT: f64 = 0.05;

fn tokenize_query(query: &str) -> Vec<String> {
    let mut tokens = HashSet::new();
    for token in query
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|tok| tok.len() >= 3)
    {
        tokens.insert(token.to_lowercase());
    }
    tokens.into_iter().collect()
}

fn keyword_overlap(tokens: &[String], text: &str) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let haystack = text.to_lowercase();
    let mut hits = 0usize;
    for token in tokens {
        if haystack.contains(token) {
            hits += 1;
        }
    }
    hits as f64 / tokens.len() as f64
}

fn apply_keyword_fallback(
    candidates: &mut [Candidate],
    tokens: &[String],
    rrf_k: f64,
    max_results: usize,
) {
    if tokens.is_empty() || max_results == 0 {
        return;
    }
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, candidate)| {
            if candidate.lexical_score.is_some() {
                return None;
            }
            let score = keyword_overlap(tokens, &candidate.data.text);
            if score > 0.0 {
                Some((idx, score))
            } else {
                None
            }
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (rank, (idx, score)) in scored.into_iter().enumerate() {
        if rank >= max_results {
            break;
        }
        let candidate = &mut candidates[idx];
        candidate.lexical_score = Some(score);
        candidate.lexical_rank = Some(rank + 1);
        candidate.fused_score += rrf_contribution(rrf_k, rank + 1);
        candidate.fused_score += score * LEXICAL_SCORE_WEIGHT;
    }
}

fn apply_token_budget(chunks: Vec<ResponseChunk>, budget: Option<u64>) -> Vec<ResponseChunk> {
    if let Some(mut remaining) = budget {
        if remaining == 0 {
            return Vec::new();
        }
        let mut filtered = Vec::new();
        for chunk in chunks {
            let cost = chunk.token_estimate.max(0) as u64;
            if cost > remaining && filtered.is_empty() {
                filtered.push(chunk);
                break;
            }
            if cost > remaining {
                break;
            }
            remaining = remaining.saturating_sub(cost);
            filtered.push(chunk);
            if remaining == 0 {
                break;
            }
        }
        filtered
    } else {
        chunks
    }
}

fn select_sql(table: &TableName) -> String {
    format!(
        "SELECT \
            url, \
            chunk_id, \
            text, \
            section_path, \
            token_estimate, \
            checksum, \
            last_seen_epoch_ms, \
            embedding <=> $1 AS distance \
        FROM {} \
        ORDER BY embedding <=> $1 ASC \
        LIMIT $2",
        table.qualified()
    )
}

fn select_lexical_sql(table: &TableName) -> String {
    format!(
        "WITH query AS (SELECT plainto_tsquery('english', $1) AS q)
        SELECT
            url,
            chunk_id,
            text,
            section_path,
            token_estimate,
            checksum,
            last_seen_epoch_ms,
            ts_rank_cd(text_tsv, query.q) AS lexical_score
        FROM {table}
        CROSS JOIN query
        WHERE query.q <> to_tsquery('') AND text_tsv @@ query.q
        ORDER BY lexical_score DESC
        LIMIT $2",
        table = table.qualified()
    )
}

fn build_cache(size: usize) -> Option<Arc<Mutex<LruCache<String, Vec<f32>>>>> {
    NonZeroUsize::new(size).map(|capacity| Arc::new(Mutex::new(LruCache::new(capacity))))
}

#[derive(Clone)]
struct RateLimiter {
    state: Arc<Mutex<RateState>>,
    capacity: f64,
    refill_per_sec: f64,
}

struct RateState {
    tokens: f64,
    last_refill: Instant,
}

impl RateLimiter {
    fn new(max_per_minute: u32, burst: u32) -> Option<Self> {
        if max_per_minute == 0 || burst == 0 {
            return None;
        }
        let capacity = burst as f64;
        let refill_per_sec = max_per_minute as f64 / 60.0;
        Some(Self {
            state: Arc::new(Mutex::new(RateState {
                tokens: capacity,
                last_refill: Instant::now(),
            })),
            capacity,
            refill_per_sec,
        })
    }

    async fn acquire(&self) -> bool {
        let mut guard = self.state.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(guard.last_refill).as_secs_f64();
        guard.last_refill = now;
        guard.tokens = (guard.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        if guard.tokens >= 1.0 {
            guard.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}
