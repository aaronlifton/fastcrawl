//! OpenAI-based embedding client implementation.

use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

/// Blocking embeddings client that talks to OpenAI-compatible endpoints.
#[derive(Clone)]
pub struct OpenAiEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    dimensions: Option<usize>,
    max_retries: usize,
    batch_size: usize,
}

impl OpenAiEmbedder {
    /// Builds a new OpenAI embeddings client.
    pub fn new(
        api_key: String,
        base_url: String,
        model: String,
        dimensions: Option<usize>,
        timeout: Duration,
        max_retries: usize,
        batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(!api_key.trim().is_empty(), "missing OpenAI API key");
        anyhow::ensure!(!model.trim().is_empty(), "missing OpenAI model name");
        let mut headers = reqwest::header::HeaderMap::new();
        let auth = format!("Bearer {}", api_key.trim());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth).context("invalid OpenAI API key")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let client = Client::builder()
            .timeout(timeout)
            .default_headers(headers)
            .build()
            .context("failed to build OpenAI HTTP client")?;
        let endpoint = format!("{}/embeddings", base_url.trim_end_matches('/'));
        Ok(Self {
            client,
            endpoint,
            model,
            dimensions,
            max_retries,
            batch_size,
        })
    }

    /// Maximum batch size configured for this client.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Sends a batch of strings to OpenAI and returns embedding vectors.
    pub fn embed_batch(&self, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            inputs.len() <= self.batch_size,
            "batch of {} exceeds configured max {}",
            inputs.len(),
            self.batch_size
        );

        let mut attempt = 0usize;
        loop {
            let request = EmbeddingRequest {
                model: &self.model,
                input: inputs,
                dimensions: self.dimensions,
            };
            let response = self.client.post(&self.endpoint).json(&request).send();
            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let mut parsed: EmbeddingResponse = resp
                            .json()
                            .context("failed to parse OpenAI embedding response")?;
                        parsed.data.sort_by_key(|entry| entry.index);
                        anyhow::ensure!(
                            parsed.data.len() == inputs.len(),
                            "OpenAI returned {} embeddings for {} inputs",
                            parsed.data.len(),
                            inputs.len()
                        );
                        return Ok(parsed
                            .data
                            .into_iter()
                            .map(|entry| entry.embedding)
                            .collect());
                    }

                    let body = resp
                        .text()
                        .unwrap_or_else(|_| "<body unavailable>".to_string());
                    if self.should_retry(status) && attempt + 1 < self.max_retries {
                        attempt += 1;
                        thread::sleep(self.retry_backoff(attempt));
                        continue;
                    }
                    anyhow::bail!("OpenAI embeddings request failed ({}): {}", status, body);
                }
                Err(err) => {
                    if self.is_retryable_error(&err) && attempt + 1 < self.max_retries {
                        attempt += 1;
                        thread::sleep(self.retry_backoff(attempt));
                        continue;
                    }
                    return Err(err.into());
                }
            }
        }
    }

    fn should_retry(&self, status: StatusCode) -> bool {
        status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
    }

    fn is_retryable_error(&self, err: &reqwest::Error) -> bool {
        err.is_timeout() || err.is_connect() || err.is_body() || err.is_request() || err.is_decode()
    }

    fn retry_backoff(&self, attempt: usize) -> Duration {
        // parallelize embedder and ensure manifest writes finish
        let capped = attempt.min(5) as u32;
        Duration::from_millis(500 * (1 << capped))
    }
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    #[serde(borrow)]
    input: &'a [&'a str],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}
