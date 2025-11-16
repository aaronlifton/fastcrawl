use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Write};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use fastcrawl::embedder::openai::OpenAiEmbedder;
use fastcrawl::normalizer::SectionHeading;
use fastcrawl::TableName;
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tokio_postgres::types::Json as PgJson;
use tokio_postgres::{Client, NoTls, Row, Statement};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-eval",
    about = "Evaluate pgvector retrieval quality against a labeled query set"
)]
struct EvalCli {
    /// Path to the evaluation JSONL file.
    #[arg(
        long,
        env = "FASTCRAWL_EVAL_DATA",
        default_value = "data/wiki_eval.jsonl"
    )]
    cases: PathBuf,

    /// Postgres connection string (postgres://...).
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Target schema for the vector table.
    #[arg(long, env = "FASTCRAWL_EVAL_SCHEMA", default_value = "public")]
    schema: String,

    /// Target table that stores chunk vectors.
    #[arg(long, env = "FASTCRAWL_EVAL_TABLE", default_value = "wiki_chunks")]
    table: String,

    /// Top-K candidates fetched per query.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Optional JSON report output path.
    #[arg(long, env = "FASTCRAWL_EVAL_REPORT")]
    report_json: Option<PathBuf>,

    /// OpenAI API key used to embed evaluation queries.
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

    /// Base URL for the OpenAI-compatible API.
    #[arg(
        long,
        env = "FASTCRAWL_OPENAI_BASE",
        default_value = "https://api.openai.com/v1"
    )]
    openai_base_url: String,

    /// Max chunks per embedding batch.
    #[arg(long, env = "FASTCRAWL_OPENAI_BATCH", default_value_t = 32)]
    batch_size: usize,

    /// Max seconds to wait for OpenAI before timing out.
    #[arg(long, env = "FASTCRAWL_OPENAI_TIMEOUT_SECS", default_value_t = 30)]
    openai_timeout_secs: u64,

    /// Number of retries for transient embedding errors.
    #[arg(long, env = "FASTCRAWL_OPENAI_MAX_RETRIES", default_value_t = 5)]
    max_retries: usize,

    /// Dense candidate count fetched from pgvector before fusion.
    #[arg(long, default_value_t = 24)]
    dense_candidates: usize,

    /// Lexical candidate count fetched via Postgres full-text search.
    #[arg(long, default_value_t = 40)]
    lexical_candidates: usize,

    /// Reciprocal Rank Fusion constant used to fuse dense + lexical rankings.
    #[arg(long, default_value_t = 60.0)]
    rrf_k: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct EvalCase {
    query: String,
    #[serde(default)]
    relevant_urls: Vec<String>,
    #[serde(default)]
    notes: Option<String>,
}

struct PreparedCase {
    case: EvalCase,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct RetrievedChunkReport {
    url: String,
    chunk_id: i64,
    dense_distance: Option<f64>,
    lexical_score: Option<f64>,
    fused_score: f64,
    token_estimate: i64,
    section_path: Vec<SectionHeading>,
    text_preview: String,
    rank: usize,
    hit: bool,
}

#[derive(Clone)]
struct ChunkData {
    url: String,
    chunk_id: i64,
    section_path: Vec<SectionHeading>,
    token_estimate: i64,
    text: String,
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
struct CaseReport {
    query: String,
    relevant_urls: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    notes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recall: Option<f64>,
    best_rank: Option<usize>,
    latency_ms: f64,
    retrieved: Vec<RetrievedChunkReport>,
}

#[derive(Debug, Serialize)]
struct EvalSummary {
    total_cases: usize,
    hit_rate: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_recall: Option<f64>,
    mean_reciprocal_rank: f64,
    avg_latency_ms: f64,
    top_k: usize,
}

#[derive(Debug, Serialize)]
struct EvalReport {
    summary: EvalSummary,
    cases: Vec<CaseReport>,
}

fn main() -> Result<()> {
    let cli = EvalCli::parse();
    let table = TableName::new(cli.schema.clone(), cli.table.clone())?;
    let dense_sql = select_dense_sql(&table);
    let lexical_sql = select_lexical_sql(&table);
    let cases = load_cases(&cli.cases)?;
    anyhow::ensure!(!cases.is_empty(), "evaluation file contains no cases");
    let embedder = OpenAiEmbedder::new(
        cli.openai_api_key,
        cli.openai_base_url,
        cli.openai_model,
        cli.openai_dimensions,
        Duration::from_secs(cli.openai_timeout_secs.max(1)),
        cli.max_retries.max(1),
        cli.batch_size.max(1),
    )?;
    let prepared = embed_cases(&cases, &embedder)?;
    let runtime = Runtime::new().context("failed to start tokio runtime")?;
    let report = runtime.block_on(run_evaluation(
        &cli.database_url,
        dense_sql,
        lexical_sql,
        cli.top_k.max(1),
        prepared,
        cli.dense_candidates.max(1),
        cli.lexical_candidates,
        cli.rrf_k.max(1.0),
    ))?;
    render_summary(&report.summary);
    if let Some(path) = cli.report_json {
        write_report(&report, &path)?;
        println!("wrote JSON report to {:?}", path);
    }
    Ok(())
}

fn load_cases(path: &PathBuf) -> Result<Vec<EvalCase>> {
    let file = File::open(path).with_context(|| format!("failed to open {:?}", path))?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read evaluation line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let case: EvalCase = serde_json::from_str(&line)
            .with_context(|| format!("invalid evaluation record at line {}", idx + 1))?;
        cases.push(case);
    }
    Ok(cases)
}

fn embed_cases(cases: &[EvalCase], embedder: &OpenAiEmbedder) -> Result<Vec<PreparedCase>> {
    let mut prepared = Vec::with_capacity(cases.len());
    for chunk in cases.chunks(embedder.batch_size()) {
        let inputs: Vec<&str> = chunk.iter().map(|case| case.query.as_str()).collect();
        let embeddings = embedder.embed_batch(&inputs)?;
        anyhow::ensure!(
            embeddings.len() == inputs.len(),
            "embedding batch returned mismatched length"
        );
        for (case, embedding) in chunk.iter().cloned().zip(embeddings.into_iter()) {
            prepared.push(PreparedCase { case, embedding });
        }
    }
    Ok(prepared)
}

async fn retrieve_hybrid(
    client: &Client,
    dense_stmt: &Statement,
    lexical_stmt: Option<&Statement>,
    embedding: &[f32],
    query_text: &str,
    dense_candidates: usize,
    lexical_candidates: usize,
    rrf_k: f64,
    top_k: usize,
) -> Result<Vec<RetrievedChunkReport>> {
    let dense_rows = fetch_dense_rows(client, dense_stmt, embedding, dense_candidates).await?;
    let lexical_rows =
        fetch_lexical_rows(client, lexical_stmt, query_text, lexical_candidates).await?;
    let mut fused = fuse_candidates(dense_rows, lexical_rows, rrf_k);
    let keyword_tokens = tokenize_query(query_text);
    let fallback_limit = if lexical_candidates > 0 {
        lexical_candidates
    } else {
        dense_candidates
    };
    apply_keyword_fallback(&mut fused, &keyword_tokens, rrf_k, fallback_limit);
    fused.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(Ordering::Equal)
    });
    fused.truncate(top_k);
    Ok(fused
        .into_iter()
        .enumerate()
        .map(|(idx, candidate)| candidate.into_report(idx + 1))
        .collect())
}

async fn fetch_dense_rows(
    client: &Client,
    stmt: &Statement,
    embedding: &[f32],
    limit: usize,
) -> Result<Vec<DenseRow>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let vector = Vector::from(embedding.to_vec());
    let rows = client.query(stmt, &[&vector, &(limit as i64)]).await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let data = chunk_data_from_row(&row);
        let distance: f64 = row.get("distance");
        out.push(DenseRow { data, distance });
    }
    Ok(out)
}

async fn fetch_lexical_rows(
    client: &Client,
    stmt: Option<&Statement>,
    query_text: &str,
    limit: usize,
) -> Result<Vec<LexicalRow>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let stmt = match stmt {
        Some(stmt) => stmt,
        None => return Ok(Vec::new()),
    };
    let trimmed = query_text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let rows = client.query(stmt, &[&trimmed, &(limit as i64)]).await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let data = chunk_data_from_row(&row);
        let score: Option<f32> = row.get("lexical_score");
        out.push(LexicalRow {
            data,
            score: score.unwrap_or(0.0) as f64,
        });
    }
    Ok(out)
}

fn chunk_data_from_row(row: &Row) -> ChunkData {
    let url: String = row.get("url");
    let chunk_id: i64 = row.get("chunk_id");
    let text: String = row.get("text");
    let PgJson(section_path): PgJson<Vec<SectionHeading>> = row.get("section_path");
    let token_estimate: i64 = row.get("token_estimate");
    ChunkData {
        url,
        chunk_id,
        section_path,
        token_estimate,
        text,
    }
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
        entry.fused_score += rrf_score(k, idx + 1);
    }
    for (idx, row) in lexical.into_iter().enumerate() {
        let key = (row.data.url.clone(), row.data.chunk_id);
        let entry = map
            .entry(key)
            .or_insert_with(|| Candidate::from_data(row.data));
        entry.lexical_rank = Some(idx + 1);
        entry.lexical_score = Some(row.score);
        entry.fused_score += rrf_score(k, idx + 1);
        entry.fused_score += row.score * LEXICAL_SCORE_WEIGHT;
    }
    let mut candidates: Vec<Candidate> = map.into_values().collect();
    candidates.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(Ordering::Equal)
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

    fn into_report(self, fused_rank: usize) -> RetrievedChunkReport {
        RetrievedChunkReport {
            url: self.data.url,
            chunk_id: self.data.chunk_id,
            dense_distance: self.dense_distance,
            lexical_score: self.lexical_score,
            fused_score: self.fused_score,
            token_estimate: self.data.token_estimate,
            section_path: self.data.section_path,
            text_preview: snippet(&self.data.text),
            rank: fused_rank,
            hit: false,
        }
    }
}

fn rrf_score(k: f64, rank: usize) -> f64 {
    1.0 / (k + rank as f64)
}

const LEXICAL_SCORE_WEIGHT: f64 = 0.05;

fn tokenize_query(query: &str) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
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
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    for (rank, (idx, score)) in scored.into_iter().enumerate() {
        if rank >= max_results {
            break;
        }
        let candidate = &mut candidates[idx];
        candidate.lexical_score = Some(score);
        candidate.lexical_rank = Some(rank + 1);
        candidate.fused_score += rrf_score(rrf_k, rank + 1);
        candidate.fused_score += score * LEXICAL_SCORE_WEIGHT;
    }
}

async fn run_evaluation(
    database_url: &str,
    dense_sql: String,
    lexical_sql: String,
    top_k: usize,
    cases: Vec<PreparedCase>,
    dense_candidates: usize,
    lexical_candidates: usize,
    rrf_k: f64,
) -> Result<EvalReport> {
    let (client, connection) = tokio_postgres::connect(database_url, NoTls)
        .await
        .with_context(|| format!("failed to connect to Postgres at {}", database_url))?;
    tokio::spawn(async move {
        if let Err(err) = connection.await {
            eprintln!("postgres connection error: {err}");
        }
    });
    let dense_statement = client.prepare(&dense_sql).await?;
    let lexical_statement = if lexical_candidates > 0 {
        Some(client.prepare(&lexical_sql).await?)
    } else {
        None
    };
    let total_cases = cases.len();
    let mut results = Vec::with_capacity(total_cases);
    let mut hits = 0usize;
    let mut reciprocal_rank_sum = 0f64;
    let mut total_latency_ms = 0f64;
    let mut recall_sum = 0f64;
    let mut recall_cases = 0usize;
    for (idx, prepared) in cases.into_iter().enumerate() {
        let start = Instant::now();
        let retrieved = retrieve_hybrid(
            &client,
            &dense_statement,
            lexical_statement.as_ref(),
            &prepared.embedding,
            &prepared.case.query,
            dense_candidates,
            lexical_candidates,
            rrf_k,
            top_k,
        )
        .await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let report = build_case_report(prepared.case, retrieved);
        total_latency_ms += latency_ms;
        if let Some(rank) = report.best_rank {
            hits += 1;
            reciprocal_rank_sum += 1.0 / rank as f64;
        }
        if let Some(recall) = report.recall {
            recall_sum += recall;
            recall_cases += 1;
        }
        results.push(CaseReport {
            latency_ms,
            ..report
        });
        render_eval_progress(idx + 1, total_cases);
    }
    if total_cases > 0 {
        println!();
    }

    let total_cases = results.len();
    let summary = EvalSummary {
        total_cases,
        hit_rate: if total_cases == 0 {
            0.0
        } else {
            hits as f64 / total_cases as f64
        },
        mean_recall: if recall_cases == 0 {
            None
        } else {
            Some(recall_sum / recall_cases as f64)
        },
        mean_reciprocal_rank: if total_cases == 0 {
            0.0
        } else {
            reciprocal_rank_sum / total_cases as f64
        },
        avg_latency_ms: if total_cases == 0 {
            0.0
        } else {
            total_latency_ms / total_cases as f64
        },
        top_k,
    };

    Ok(EvalReport {
        summary,
        cases: results,
    })
}

fn build_case_report(case: EvalCase, mut retrieved: Vec<RetrievedChunkReport>) -> CaseReport {
    let relevant: HashSet<String> = case
        .relevant_urls
        .iter()
        .map(|url| url.to_lowercase())
        .collect();
    let mut matched_urls = HashSet::new();
    let mut best_rank = None;
    let mut hit_counts = 0usize;
    for entry in retrieved.iter_mut() {
        let key = entry.url.to_lowercase();
        if relevant.contains(&key) {
            entry.hit = true;
            if matched_urls.insert(key) {
                hit_counts += 1;
                if best_rank.is_none() {
                    best_rank = Some(entry.rank);
                }
            }
        }
    }
    let recall = if case.relevant_urls.is_empty() {
        None
    } else {
        Some(hit_counts as f64 / case.relevant_urls.len() as f64)
    };

    CaseReport {
        query: case.query,
        relevant_urls: case.relevant_urls,
        notes: case.notes,
        recall,
        best_rank,
        retrieved,
        latency_ms: 0.0, // filled by caller
    }
}

fn snippet(text: &str) -> String {
    const MAX_CHARS: usize = 200;
    if text.len() <= MAX_CHARS {
        return text.to_string();
    }
    let mut snippet = text.chars().take(MAX_CHARS).collect::<String>();
    snippet.push('…');
    snippet
}

fn select_dense_sql(table: &TableName) -> String {
    format!(
        "SELECT url, chunk_id, text, section_path, token_estimate, embedding <=> $1 AS distance \
         FROM {} ORDER BY embedding <=> $1 ASC LIMIT $2",
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
            ts_rank_cd(text_tsv, query.q) AS lexical_score
         FROM {table}
         CROSS JOIN query
         WHERE query.q <> to_tsquery('') AND text_tsv @@ query.q
         ORDER BY lexical_score DESC
         LIMIT $2",
        table = table.qualified()
    )
}

fn render_summary(summary: &EvalSummary) {
    println!("--- Retrieval Evaluation Summary ---");
    println!("cases: {}", summary.total_cases);
    println!("top_k: {}", summary.top_k);
    println!("hit rate: {:.3}", summary.hit_rate);
    if let Some(recall) = summary.mean_recall {
        println!("mean recall: {:.3}", recall);
    } else {
        println!("mean recall: n/a");
    }
    println!("mean reciprocal rank: {:.3}", summary.mean_reciprocal_rank);
    println!("avg db latency (ms): {:.2}", summary.avg_latency_ms);
}

fn write_report(report: &EvalReport, path: &PathBuf) -> Result<()> {
    let file = File::create(path).with_context(|| format!("failed to create {:?}", path))?;
    serde_json::to_writer_pretty(file, report).context("failed to write JSON report")?;
    Ok(())
}

fn render_eval_progress(done: usize, total: usize) {
    if total == 0 {
        return;
    }
    let pct = (done as f64 / total as f64) * 100.0;
    print!("\rEvaluating {done}/{total} ({pct:.1}%)");
    let _ = io::stdout().flush();
}
