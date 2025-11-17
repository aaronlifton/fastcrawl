use std::time::Duration;

use anyhow::{bail, Context, Result};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::{LlmProvider, ProviderRequest};

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("failed to build Anthropic HTTP client")?;
        Ok(Self {
            api_key,
            model,
            client,
        })
    }
}

impl LlmProvider for AnthropicProvider {
    fn answer(&self, request: &ProviderRequest) -> Result<String> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(self.api_key.trim()).context("invalid Anthropic API key")?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            messages: vec![AnthropicMessage {
                role: "user",
                content: vec![AnthropicContentBlock {
                    kind: "text",
                    text: request.prompt,
                }],
            }],
        };
        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .headers(headers)
            .json(&body)
            .send()
            .context("failed to call Anthropic messages API")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp
                .text()
                .unwrap_or_else(|_| "<body unavailable>".to_string());
            bail!("Anthropic returned {}: {}", status, text);
        }
        let parsed: AnthropicResponse =
            resp.json().context("failed to parse Anthropic response")?;
        let answer = parsed
            .content
            .into_iter()
            .filter_map(|block| match block {
                AnthropicResponseBlock::Text { text } => Some(text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if answer.is_empty() {
            bail!("Anthropic response missing text content");
        }
        Ok(answer)
    }
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    temperature: f32,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: Vec<AnthropicContentBlock<'a>>,
}

#[derive(Serialize)]
struct AnthropicContentBlock<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseBlock {
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}
