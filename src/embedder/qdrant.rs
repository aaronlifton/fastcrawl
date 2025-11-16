//! Qdrant Cloud Inference embedding client.

use std::thread;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderValue, CONTENT_TYPE};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

/// Blocking embeddings client that talks to Qdrant Cloud Inference endpoints.
#[derive(Clone)]
pub struct QdrantEmbedder {
    client: Client,
    endpoint: String,
    model: String,
    max_retries: usize,
    batch_size: usize,
}

impl QdrantEmbedder {
    /// Builds a new Qdrant embeddings client.
    ///
    /// # Arguments
    /// * `api_key` - Value for the `api-key` header (usually from `QDRANT_API_KEY` env var)
    /// * `endpoint` - Full inference endpoint, e.g. `https://cluster-id.cloud.qdrant.io/inference/text`
    /// * `model` - Model identifier advertised by the cluster (e.g. `qdrant/all-MiniLM-L6-v2`)
    pub fn new(
        api_key: String,
        endpoint: String,
        model: String,
        timeout: Duration,
        max_retries: usize,
        batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(!api_key.trim().is_empty(), "missing Qdrant API key");
        anyhow::ensure!(
            endpoint.starts_with("http://") || endpoint.starts_with("https://"),
            "Qdrant endpoint must be an http(s) URL"
        );
        anyhow::ensure!(!model.trim().is_empty(), "missing Qdrant model name");
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "api-key",
            HeaderValue::from_str(api_key.trim()).context("invalid Qdrant API key")?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let client = Client::builder()
            .timeout(timeout)
            .default_headers(headers)
            .build()
            .context("failed to build Qdrant HTTP client")?;
        Ok(Self {
            client,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model,
            max_retries: max_retries.max(1),
            batch_size: batch_size.max(1),
        })
    }

    /// Maximum batch size configured for this client.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Sends a batch of strings to Qdrant Cloud Inference and returns embedding vectors.
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
            let request = InferenceRequest {
                model: &self.model,
                inputs,
            };
            let response = self.client.post(&self.endpoint).json(&request).send();
            match response {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let payload: InferenceResponse = resp
                            .json()
                            .context("failed to parse Qdrant inference response")?;
                        let embeddings = payload.into_embeddings(inputs.len())?;
                        return Ok(embeddings);
                    }
                    let body = resp
                        .text()
                        .unwrap_or_else(|_| "<body unavailable>".to_string());
                    if should_retry(status) && attempt + 1 < self.max_retries {
                        attempt += 1;
                        thread::sleep(retry_backoff(attempt));
                        continue;
                    }
                    anyhow::bail!("Qdrant inference request failed ({}): {}", status, body);
                }
                Err(err) => {
                    if err.is_connect() || err.is_timeout() || err.is_request() || err.is_body() {
                        if attempt + 1 < self.max_retries {
                            attempt += 1;
                            thread::sleep(retry_backoff(attempt));
                            continue;
                        }
                    }
                    return Err(err.into());
                }
            }
        }
    }
}

fn should_retry(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn retry_backoff(attempt: usize) -> Duration {
    let capped = attempt.min(5) as u32;
    Duration::from_millis(500 * (1 << capped))
}

#[derive(Serialize)]
struct InferenceRequest<'a> {
    model: &'a str,
    #[serde(rename = "input")]
    inputs: &'a [&'a str],
}

#[derive(Debug, Deserialize)]
struct InferenceResponse {
    #[serde(default)]
    data: Vec<InferenceData>,
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
}

impl InferenceResponse {
    fn into_embeddings(self, expected_len: usize) -> Result<Vec<Vec<f32>>> {
        if !self.data.is_empty() {
            anyhow::ensure!(
                self.data.len() == expected_len,
                "Qdrant returned {} embeddings for {} inputs",
                self.data.len(),
                expected_len
            );
            let mut data = self.data;
            data.sort_by_key(|d| d.index.unwrap_or(0));
            return Ok(data.into_iter().map(|d| d.embedding).collect());
        }
        if !self.embeddings.is_empty() {
            anyhow::ensure!(
                self.embeddings.len() == expected_len,
                "Qdrant returned {} embeddings for {} inputs",
                self.embeddings.len(),
                expected_len
            );
            return Ok(self.embeddings);
        }
        Err(anyhow!("Qdrant response missing embedding payloads"))
    }
}

#[derive(Debug, Deserialize)]
struct InferenceData {
    embedding: Vec<f32>,
    #[serde(default)]
    index: Option<usize>,
}
