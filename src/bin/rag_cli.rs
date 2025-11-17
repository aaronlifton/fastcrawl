use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[path = "rag_cli/providers/mod.rs"]
mod providers;

use providers::{AnthropicProvider, LlmProvider, OpenAiProvider, ProviderRequest};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-rag",
    about = "Query wiki chunks via fastcrawl-retriever and stream them into an LLM answer"
)]
struct RagCli {
    /// Question to answer using the wiki chunk store
    #[arg(long)]
    query: String,

    /// Retriever HTTP endpoint
    #[arg(
        long,
        env = "FASTCRAWL_RETRIEVER_URL",
        default_value = "http://127.0.0.1:8080/v1/query"
    )]
    retriever_url: String,

    /// Number of chunks requested from the retriever
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Optional token budget enforced by the retriever
    #[arg(long)]
    max_tokens: Option<u64>,

    /// OpenAI API key for the answering model
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    /// OpenAI chat model used for synthesis
    #[arg(long, env = "FASTCRAWL_RAG_MODEL", default_value = "gpt-4o-mini")]
    openai_model: String,

    /// Optional max words constraint included in the prompt
    #[arg(long)]
    max_words: Option<usize>,

    /// Sampling temperature for the answer model
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    /// Maximum tokens to request from the completion model
    #[arg(long, default_value_t = 400)]
    max_completion_tokens: usize,

    /// Target LLM provider (openai or anthropic)
    #[arg(long, env = "FASTCRAWL_RAG_PROVIDER", default_value = "openai")]
    llm_provider: String,

    /// Anthropic API key (required when --llm-provider anthropic)
    #[arg(long, env = "ANTHROPIC_API_KEY")]
    anthropic_api_key: Option<String>,

    /// Anthropic model identifier
    #[arg(
        long,
        env = "FASTCRAWL_ANTHROPIC_MODEL",
        default_value = "claude-3-sonnet-20240229"
    )]
    anthropic_model: String,

    /// Only print the rendered context (skip LLM call)
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let cli = RagCli::parse();
    let retriever_client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("failed to build retriever HTTP client")?;
    let retriever_response = retrieve_chunks(
        &retriever_client,
        &cli.retriever_url,
        &cli.query,
        cli.top_k,
        cli.max_tokens,
    )?;
    if retriever_response.chunks.is_empty() {
        bail!("retriever returned zero chunks; is the server running?");
    }
    let context_block = render_context(&retriever_response.chunks);
    println!("--- Retrieved Context ---\n{context_block}\n");
    if cli.dry_run {
        println!("dry-run enabled; skipping LLM call.");
        return Ok(());
    }

    let prompt = build_prompt(&cli.query, &context_block, cli.max_words);
    let llm_request = ProviderRequest {
        prompt: &prompt,
        temperature: cli.temperature,
        max_tokens: cli.max_completion_tokens,
    };
    let provider = cli.llm_provider.to_lowercase();
    let answer = match provider.as_str() {
        "openai" => {
            let key = cli
                .openai_api_key
                .clone()
                .ok_or_else(|| anyhow!("OPENAI_API_KEY must be set for the OpenAI provider"))?;
            let provider = OpenAiProvider::new(key, cli.openai_model.clone())?;
            provider.answer(&llm_request)?
        }
        "anthropic" => {
            let key = cli.anthropic_api_key.clone().ok_or_else(|| {
                anyhow!("ANTHROPIC_API_KEY must be set for the Anthropic provider")
            })?;
            let provider = AnthropicProvider::new(key, cli.anthropic_model.clone())?;
            provider.answer(&llm_request)?
        }
        other => bail!(
            "unsupported llm provider '{}'; use openai or anthropic",
            other
        ),
    };
    println!("--- Answer ---\n{answer}");
    Ok(())
}

fn retrieve_chunks(
    client: &Client,
    url: &str,
    query: &str,
    top_k: usize,
    max_tokens: Option<u64>,
) -> Result<RetrieverResponse> {
    let request = RetrieverRequest {
        query,
        top_k,
        max_tokens,
    };
    let resp = client
        .post(url)
        .json(&request)
        .send()
        .with_context(|| format!("failed to call retriever at {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp
            .text()
            .unwrap_or_else(|_| "<body unavailable>".to_string());
        bail!("retriever returned {}: {}", status, body);
    }
    let parsed: RetrieverResponse = resp.json().context("failed to parse retriever response")?;
    Ok(parsed)
}

fn build_prompt(question: &str, context_block: &str, max_words: Option<usize>) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are FastcrawlWiki, a precise assistant that must only use the supplied wiki chunks. Cite your sources as [^chunk_id].\n\n");
    prompt.push_str("Context:\n");
    prompt.push_str(context_block);
    prompt.push_str("\n\nQuestion:\n");
    prompt.push_str(question);
    prompt.push_str("\n\nInstructions:\n1. Give a direct answer in 1-2 sentences referencing citations.\n2. Add a short bullet list of supporting facts with citations.\n3. State a confidence level (High/Med/Low).\n");
    if let Some(limit) = max_words {
        prompt.push_str(&format!(
            "4. Limit the answer to roughly {} words.\n",
            limit
        ));
    }
    prompt
}

fn render_context(chunks: &[RetrieverChunk]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        out.push_str(&format!(
            "URL: {}\nChunk ID: {}\nSection: {}\nFused Score: {:.4} | Lexical: {:?} | Dense: {:?}\n{}\n---\n",
            chunk.url,
            chunk.chunk_id,
            render_section_path(&chunk.section_path),
            chunk.fused_score,
            chunk.lexical_score,
            chunk.dense_distance,
            chunk.text.trim()
        ));
    }
    out
}

fn render_section_path(path: &[SectionHeading]) -> String {
    if path.is_empty() {
        return String::from("(none)");
    }
    path.iter()
        .map(|heading| heading.title.as_str())
        .collect::<Vec<_>>()
        .join(" > ")
}

#[derive(Serialize)]
struct RetrieverRequest<'a> {
    query: &'a str,
    #[serde(rename = "top_k")]
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RetrieverResponse {
    chunks: Vec<RetrieverChunk>,
    #[allow(dead_code)]
    meta: RetrieverMeta,
}

#[derive(Debug, Deserialize)]
struct RetrieverChunk {
    url: String,
    chunk_id: i64,
    text: String,
    section_path: Vec<SectionHeading>,
    #[allow(dead_code)]
    token_estimate: i64,
    fused_score: f64,
    lexical_score: Option<f64>,
    dense_distance: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SectionHeading {
    title: String,
    #[allow(dead_code)]
    level: u8,
}

#[derive(Debug, Deserialize)]
struct RetrieverMeta {
    #[allow(dead_code)]
    top_k: usize,
    #[allow(dead_code)]
    latency_ms: f64,
    #[allow(dead_code)]
    token_budget: Option<u64>,
}
